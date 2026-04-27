"""Laminar tracing plugin for Hermes Agent.

Wires Hermes's plugin hooks (``pre_llm_call``, ``pre_tool_call``,
``post_tool_call``, ``on_session_end``, ``subagent_stop``) into Laminar
spans so every conversation turn, tool call, and subagent run shows up
as a proper nested trace in the Laminar UI.

Installation (local dev, no pip publish needed)::

    pip install -e path/to/lmnr-python/examples/hermes-plugin
    hermes plugins enable lmnr-hermes

Or drop the package under ``~/.hermes/plugins/lmnr-hermes/`` and the
directory loader will pick it up.

Initialization reads ``LMNR_PROJECT_API_KEY`` / ``LMNR_BASE_URL`` from the
environment. Raw provider instrumentors (OpenAI, Anthropic, Bedrock) are
auto-enabled by :func:`lmnr.Laminar.initialize` and will nest their spans
under the current Hermes turn automatically.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from lmnr import Laminar
from lmnr.sdk.types import LaminarSpanContext
from opentelemetry import context as context_api
from opentelemetry import trace

logger = logging.getLogger(__name__)

_init_lock = threading.Lock()
_initialized = False
_init_failed = False

# (session_id, tool_call_id) -> active tool span.  Entries are inserted in
# pre_tool_call and popped in post_tool_call.  Same-thread access within a
# single tool dispatch, so no extra locking needed.
_tool_spans: dict[tuple[str, str], Any] = {}

# session_id -> {"span": LaminarSpan, "ctx_token": OTel context token,
#                "lmnr_ctx": LaminarSpanContext serialized for cross-thread reuse}
_turn_state: dict[str, dict[str, Any]] = {}
_turn_lock = threading.Lock()


def _ensure_initialized() -> bool:
    """Initialize Laminar once. Returns True if tracing is active.

    A failed initialization (missing key or exception) is cached in
    ``_init_failed`` so subsequent hook invocations short-circuit without
    re-acquiring the lock or re-running expensive setup.
    """
    global _initialized, _init_failed
    if _initialized:
        return True
    if _init_failed:
        return False
    with _init_lock:
        if _initialized:
            return True
        if _init_failed:
            return False
        if Laminar.is_initialized():
            _initialized = True
            return True
        api_key = os.environ.get("LMNR_PROJECT_API_KEY")
        if not api_key and not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            logger.info(
                "lmnr-hermes: LMNR_PROJECT_API_KEY not set; tracing disabled."
            )
            _init_failed = True
            return False
        try:
            Laminar.initialize()
        except Exception as exc:
            logger.warning("lmnr-hermes: Laminar.initialize() failed: %s", exc)
            _init_failed = True
            return False
        _initialized = True
        return True


def _safe_str(value: Any, limit: int = 4000) -> str:
    try:
        s = value if isinstance(value, str) else str(value)
    except Exception:
        return ""
    return s if len(s) <= limit else s[:limit] + "...[truncated]"


# ---------------------------------------------------------------------------
# Hook callbacks
# ---------------------------------------------------------------------------


def _on_session_start(
    session_id: str = "",
    model: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    if not _ensure_initialized():
        return
    # Nothing to span here — the first real work is pre_llm_call on the same
    # turn.  Keep this hook registered so downstream metadata (e.g. platform)
    # is available for logging and so future plugin revisions can use it.
    logger.debug(
        "lmnr-hermes: session start session_id=%s model=%s platform=%s",
        session_id, model, platform,
    )


def _on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    conversation_history: list | None = None,
    is_first_turn: bool = False,
    model: str = "",
    platform: str = "",
    sender_id: str = "",
    **_: Any,
) -> None:
    if not _ensure_initialized():
        return
    if not session_id:
        return

    # If a prior turn's span was never closed (on_session_end missed, crash,
    # etc.), close it before starting a new one so we don't leak spans.
    _close_turn(session_id, reason="reenter")

    try:
        span = Laminar.start_span(
            name="hermes.turn",
            input={
                "user_message": _safe_str(user_message),
                "model": model,
                "platform": platform,
                "is_first_turn": is_first_turn,
                "history_len": len(conversation_history or []),
            },
            session_id=session_id,
            user_id=sender_id or None,
            tags=[t for t in ["hermes", platform or None] if t],
            attributes={
                "hermes.model": model or "",
                "hermes.platform": platform or "",
                "hermes.is_first_turn": bool(is_first_turn),
            },
        )
    except Exception as exc:
        logger.warning("lmnr-hermes: failed to start turn span: %s", exc)
        return

    # Attach span to the OTel current context so raw provider instrumentors
    # (OpenAI, Anthropic, Bedrock) running on the main thread parent under it.
    try:
        ctx = trace.set_span_in_context(span)
        ctx_token = context_api.attach(ctx)
    except Exception as exc:
        logger.warning("lmnr-hermes: attach context failed: %s", exc)
        ctx_token = None

    try:
        lmnr_ctx = Laminar.get_laminar_span_context_dict(span)
    except Exception:
        lmnr_ctx = None

    with _turn_lock:
        _turn_state[session_id] = {
            "span": span,
            "ctx_token": ctx_token,
            "lmnr_ctx": lmnr_ctx,
        }


def _on_post_api_request(
    task_id: str = "",
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    api_mode: str = "",
    api_call_count: int = 0,
    api_duration: float | None = None,
    finish_reason: str = "",
    usage: dict | None = None,
    response_model: str | None = None,
    assistant_content_chars: int = 0,
    assistant_tool_call_count: int = 0,
    **_: Any,
) -> None:
    """Attach per-API-call attributes to the turn span so the UI shows token
    usage and finish reason even when provider instrumentors are disabled."""
    if not _ensure_initialized():
        return
    with _turn_lock:
        state = _turn_state.get(session_id)
    if not state:
        return
    span = state["span"]
    try:
        attrs: dict[str, Any] = {
            "hermes.provider": provider or "",
            "hermes.api_mode": api_mode or "",
            "hermes.api_call_count": int(api_call_count or 0),
            "hermes.finish_reason": finish_reason or "",
            "hermes.assistant_tool_call_count": int(assistant_tool_call_count or 0),
        }
        if api_duration is not None:
            attrs["hermes.api_duration_ms"] = int(float(api_duration) * 1000)
        if response_model:
            attrs["hermes.response_model"] = response_model
        if isinstance(usage, dict):
            for k, v in usage.items():
                if isinstance(v, (int, float, str, bool)):
                    attrs[f"hermes.usage.{k}"] = v
        for k, v in attrs.items():
            span.set_attribute(k, v)
    except Exception as exc:
        logger.debug("lmnr-hermes: set post_api_request attrs failed: %s", exc)


def _on_pre_tool_call(
    tool_name: str = "",
    args: dict | None = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> None:
    if not _ensure_initialized():
        return
    if not session_id or not tool_call_id:
        return

    with _turn_lock:
        state = _turn_state.get(session_id)
    parent_ctx = state.get("lmnr_ctx") if state else None

    try:
        span = Laminar.start_span(
            name=f"tool.{tool_name}" if tool_name else "tool",
            input={"tool_name": tool_name, "args": args or {}},
            span_type="TOOL",
            parent_span_context=parent_ctx,
            session_id=session_id or None,
            attributes={
                "hermes.tool_name": tool_name or "",
                "hermes.tool_call_id": tool_call_id or "",
                "hermes.task_id": task_id or "",
            },
        )
    except Exception as exc:
        logger.debug("lmnr-hermes: start tool span failed: %s", exc)
        return

    _tool_spans[(session_id, tool_call_id)] = span


def _on_post_tool_call(
    tool_name: str = "",
    args: dict | None = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    duration_ms: int = 0,
    **_: Any,
) -> None:
    if not _ensure_initialized():
        return
    span = _tool_spans.pop((session_id, tool_call_id), None)
    if span is None:
        return
    try:
        span.set_attribute("hermes.tool_duration_ms", int(duration_ms or 0))
        span.set_output(_safe_str(result, limit=16000))
    except Exception:
        pass
    try:
        span.end()
    except Exception:
        pass


def _close_turn(session_id: str, *, reason: str = "session_end") -> None:
    with _turn_lock:
        state = _turn_state.pop(session_id, None)
    if not state:
        return
    span = state.get("span")
    ctx_token = state.get("ctx_token")
    if ctx_token is not None:
        try:
            context_api.detach(ctx_token)
        except Exception:
            pass
    if span is None:
        return
    try:
        span.set_attribute("hermes.turn_end_reason", reason)
    except Exception:
        pass
    try:
        span.end()
    except Exception:
        pass


def _on_post_llm_call(
    session_id: str = "",
    user_message: str = "",
    assistant_response: str = "",
    conversation_history: list | None = None,
    model: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    if not _ensure_initialized():
        return
    with _turn_lock:
        state = _turn_state.get(session_id)
    if not state:
        return
    span = state["span"]
    try:
        span.set_output(_safe_str(assistant_response, limit=16000))
    except Exception:
        pass


def _on_session_end(
    session_id: str = "",
    completed: bool = True,
    interrupted: bool = False,
    model: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    if not _ensure_initialized():
        return
    with _turn_lock:
        state = _turn_state.get(session_id)
    if state is not None:
        span = state.get("span")
        try:
            if span is not None:
                span.set_attribute("hermes.completed", bool(completed))
                span.set_attribute("hermes.interrupted", bool(interrupted))
        except Exception:
            pass
    _close_turn(
        session_id,
        reason="interrupted" if interrupted else ("completed" if completed else "ended"),
    )


def _on_subagent_stop(
    parent_session_id: str = "",
    child_role: str = "",
    child_summary: str = "",
    child_status: str = "",
    duration_ms: int = 0,
    **_: Any,
) -> None:
    """Emit a retrospective span for a finished subagent delegation.

    Hermes fires this on the parent thread after the child agent has already
    finished, so we can't span the child's execution itself — we just record
    a short instantaneous span with the child's metadata nested under the
    parent's turn span.
    """
    if not _ensure_initialized():
        return
    with _turn_lock:
        state = _turn_state.get(parent_session_id)
    parent_ctx = state.get("lmnr_ctx") if state else None
    try:
        span = Laminar.start_span(
            name=f"subagent.{child_role}" if child_role else "subagent",
            input={"role": child_role},
            parent_span_context=parent_ctx,
            session_id=parent_session_id or None,
            attributes={
                "hermes.subagent.role": child_role or "",
                "hermes.subagent.status": child_status or "",
                "hermes.subagent.duration_ms": int(duration_ms or 0),
            },
        )
        span.set_output(_safe_str(child_summary, limit=8000))
        span.end()
    except Exception as exc:
        logger.debug("lmnr-hermes: subagent span failed: %s", exc)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point called by Hermes's PluginManager."""
    # Initialize eagerly so the Laminar SDK can start its background exporter
    # before the first turn — cleaner startup logs than lazy-initializing in
    # the first hook call.
    _ensure_initialized()
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("subagent_stop", _on_subagent_stop)


__all__ = ["register"]

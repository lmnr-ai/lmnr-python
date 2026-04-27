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

# (session_id, api_call_count) -> active LLM span.  Opened in pre_api_request
# and closed in post_api_request.  Hermes drives its own HTTP client so the
# raw anthropic/openai instrumentors never see these calls — we have to emit
# LLM spans ourselves to get token usage and cost into the Laminar UI.
_api_spans: dict[tuple[str, int], Any] = {}

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
        init_kwargs: dict[str, Any] = {}
        http_port = os.environ.get("LMNR_HTTP_PORT")
        grpc_port = os.environ.get("LMNR_GRPC_PORT")
        if http_port:
            init_kwargs["http_port"] = int(http_port)
        if grpc_port:
            init_kwargs["grpc_port"] = int(grpc_port)
        try:
            Laminar.initialize(**init_kwargs)
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


def _on_pre_api_request(
    task_id: str = "",
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    api_call_count: int = 0,
    message_count: int = 0,
    tool_count: int = 0,
    approx_input_tokens: int = 0,
    request_char_count: int = 0,
    max_tokens: int | None = None,
    **_: Any,
) -> None:
    """Open an LLM span for each provider API call.

    Hermes drives its own HTTP client (not the ``anthropic`` / ``openai``
    Python SDKs) so the raw-SDK instrumentors Laminar auto-enables never see
    these calls.  We emit the span ourselves and close it in
    ``post_api_request`` where the response usage dict is available.
    """
    if not _ensure_initialized():
        return
    if not session_id:
        return

    with _turn_lock:
        state = _turn_state.get(session_id)
    parent_ctx = state.get("lmnr_ctx") if state else None

    # Short human-readable model slug for the span name
    span_name = f"llm.{provider}" if provider else "llm"
    if model:
        span_name = f"{span_name}.{model}"

    try:
        span = Laminar.start_span(
            name=span_name,
            span_type="LLM",
            parent_span_context=parent_ctx,
            session_id=session_id or None,
            attributes={
                "gen_ai.system": provider or "",
                "gen_ai.request.model": model or "",
                "hermes.api_mode": api_mode or "",
                "hermes.api_call_count": int(api_call_count or 0),
                "hermes.task_id": task_id or "",
                "hermes.message_count": int(message_count or 0),
                "hermes.tool_count": int(tool_count or 0),
                "hermes.approx_input_tokens": int(approx_input_tokens or 0),
                "hermes.request_char_count": int(request_char_count or 0),
            },
        )
        if base_url:
            try:
                span.set_attribute("gen_ai.request.server_address", base_url)
            except Exception:
                pass
        if max_tokens is not None:
            try:
                span.set_attribute("gen_ai.request.max_tokens", int(max_tokens))
            except Exception:
                pass
    except Exception as exc:
        logger.debug("lmnr-hermes: start llm span failed: %s", exc)
        return

    _api_spans[(session_id, int(api_call_count or 0))] = span


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
    """Close the LLM span opened in pre_api_request with usage and output."""
    if not _ensure_initialized():
        return
    span = _api_spans.pop((session_id, int(api_call_count or 0)), None)
    if span is None:
        return

    try:
        if isinstance(usage, dict):
            # Laminar derives cost from these exact attribute keys (+ model).
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            cache_read = int(usage.get("cache_read_tokens") or 0)
            cache_write = int(usage.get("cache_write_tokens") or 0)
            reasoning = int(usage.get("reasoning_tokens") or 0)
            total = int(
                usage.get("total_tokens")
                or (input_tokens + output_tokens + cache_read + cache_write)
            )
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            if cache_read:
                span.set_attribute("gen_ai.usage.cache_read_input_tokens", cache_read)
            if cache_write:
                span.set_attribute(
                    "gen_ai.usage.cache_creation_input_tokens", cache_write
                )
            if reasoning:
                span.set_attribute("gen_ai.usage.reasoning_tokens", reasoning)
            if total:
                span.set_attribute("llm.usage.total_tokens", total)

        attrs: dict[str, Any] = {
            "hermes.finish_reason": finish_reason or "",
            "hermes.assistant_content_chars": int(assistant_content_chars or 0),
            "hermes.assistant_tool_call_count": int(assistant_tool_call_count or 0),
        }
        if api_duration is not None:
            attrs["hermes.api_duration_ms"] = int(float(api_duration) * 1000)
        if response_model:
            attrs["gen_ai.response.model"] = response_model
            attrs["hermes.response_model"] = response_model
        for k, v in attrs.items():
            span.set_attribute(k, v)
    except Exception as exc:
        logger.debug("lmnr-hermes: set llm span attrs failed: %s", exc)

    try:
        # A brief summary is more useful than a raw JSON dump of the whole
        # response body — we don't receive the body in the hook payload
        # anyway.  Surface finish reason + tool-call count as the "output".
        output = {
            "finish_reason": finish_reason or "",
            "assistant_content_chars": int(assistant_content_chars or 0),
            "assistant_tool_call_count": int(assistant_tool_call_count or 0),
        }
        if response_model:
            output["response_model"] = response_model
        span.set_output(output)
    except Exception:
        pass

    try:
        span.end()
    except Exception:
        pass


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


def _cleanup_api_spans(session_id: str) -> None:
    """Close any LLM spans opened in ``pre_api_request`` but never closed.

    A provider call that raises before ``post_api_request`` fires would leave
    the span dangling; sweep it when the turn ends so the trace is never
    missing its closing event.
    """
    leaked = [k for k in _api_spans if k[0] == session_id]
    for k in leaked:
        span = _api_spans.pop(k, None)
        if span is None:
            continue
        try:
            span.set_attribute("hermes.llm_end_reason", "turn_closed")
        except Exception:
            pass
        try:
            span.end()
        except Exception:
            pass


def _finish_turn_state(
    state: dict[str, Any] | None, *, reason: str
) -> None:
    """End a turn span from an already-popped state entry.

    Split out so callers that need to atomically pop-and-mutate the state
    (e.g. ``_on_session_end``) can do so under a single lock acquisition
    instead of a get-then-pop that another thread could interleave.
    """
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


def _close_turn(session_id: str, *, reason: str = "session_end") -> None:
    with _turn_lock:
        state = _turn_state.pop(session_id, None)
    _cleanup_api_spans(session_id)
    _finish_turn_state(state, reason=reason)


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
    # Atomically pop the state so a concurrent pre_llm_call that installs a
    # new turn for this session_id can't have its state ended here by mistake.
    with _turn_lock:
        state = _turn_state.pop(session_id, None)
    if state is not None:
        span = state.get("span")
        try:
            if span is not None:
                span.set_attribute("hermes.completed", bool(completed))
                span.set_attribute("hermes.interrupted", bool(interrupted))
        except Exception:
            pass
    _cleanup_api_spans(session_id)
    _finish_turn_state(
        state,
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
    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("subagent_stop", _on_subagent_stop)


__all__ = ["register"]

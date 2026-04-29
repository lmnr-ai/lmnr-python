"""LaminarMiddleware - an `AgentMiddleware` that emits Laminar spans."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from opentelemetry import context as context_api
from opentelemetry.trace import Status, StatusCode

from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.opentelemetry_lib.tracing.attributes import SPAN_TYPE
from lmnr.opentelemetry_lib.tracing.context import attach_context, detach_context
from lmnr.opentelemetry_lib.tracing.span import LaminarSpan
from lmnr.opentelemetry_lib.tracing.tracer import get_tracer_with_context
from lmnr.opentelemetry_lib.tracing.utils import set_association_props_in_context
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import json_dumps

try:
    from langchain.agents.middleware.types import AgentMiddleware
except ImportError:  # pragma: no cover - guarded by the instrumentor
    AgentMiddleware = object  # type: ignore[assignment,misc]

logger = get_default_logger(__name__)


def _summarize_messages(messages: Any) -> Any:
    """Extract role + content pairs from langchain/langgraph message objects."""
    if not isinstance(messages, list):
        return messages
    out = []
    for m in messages:
        role = getattr(m, "type", None) or getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role is None and isinstance(m, dict):
            role = m.get("role") or m.get("type")
            content = m.get("content", content)
        out.append({"role": role, "content": content} if role is not None else m)
    return out


def _tool_result_to_json(result: Any) -> Any:
    """Return a JSON-friendly view of a ToolMessage / Command."""
    if hasattr(result, "content"):
        return getattr(result, "content", None)
    if hasattr(result, "update"):
        try:
            return {"update": getattr(result, "update", None)}
        except Exception:
            return repr(result)
    return result


class _SpanHandle:
    """Context manager that opens a Laminar span and cleans up on exit."""

    def __init__(self, name: str, span_type: str):
        self.name = name
        self.span_type = span_type
        self.span: LaminarSpan | None = None
        self._wrapper: TracerWrapper | None = None
        self._ctx_token = None
        self._iso_token = None
        self._assoc_token = None
        self._did_push = False

    def __enter__(self) -> "_SpanHandle":
        if not TracerWrapper.verify_initialized():
            return self
        try:
            self._wrapper = TracerWrapper()
        except Exception:
            logger.debug("TracerWrapper unavailable; skipping span", exc_info=True)
            return self
        try:
            with get_tracer_with_context() as (tracer, isolated_context):
                raw = tracer.start_span(
                    self.name,
                    context=isolated_context,
                    attributes={SPAN_TYPE: self.span_type},
                )
            self.span = raw if isinstance(raw, LaminarSpan) else LaminarSpan(raw)
            self._assoc_token = set_association_props_in_context(self.span)
            new_ctx = self._wrapper.push_span_context(self.span)
            self._did_push = True
            self._ctx_token = context_api.attach(new_ctx)
            self._iso_token = attach_context(new_ctx)
        except Exception:
            logger.debug("Failed to open Laminar span", exc_info=True)
            self.span = None
        return self

    def set_input(self, value: Any) -> None:
        if self.span is None or value is None:
            return
        try:
            self.span.set_input(value)
        except Exception:
            logger.debug("Failed to set span input", exc_info=True)

    def set_output(self, value: Any) -> None:
        if self.span is None or value is None:
            return
        try:
            self.span.set_output(value)
        except Exception:
            logger.debug("Failed to set span output", exc_info=True)

    def set_attribute(self, key: str, value: Any) -> None:
        if self.span is None:
            return
        try:
            if isinstance(value, (str, int, float, bool)):
                self.span.set_attribute(key, value)
            else:
                self.span.set_attribute(key, json_dumps(value))
        except Exception:
            logger.debug("Failed to set attribute %s", key, exc_info=True)

    def record_exception(self, exc: BaseException) -> None:
        if self.span is None:
            return
        try:
            self.span.record_exception(exc, escaped=True)
            self.span.set_status(Status(StatusCode.ERROR, str(exc)))
        except Exception:
            logger.debug("Failed to record exception", exc_info=True)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.span is None:
            return
        try:
            self.span.end()
        except Exception:
            logger.debug("Failed to end span", exc_info=True)
        try:
            if self._ctx_token is not None:
                context_api.detach(self._ctx_token)
        except Exception:
            logger.debug("Failed to detach global context", exc_info=True)
        try:
            if self._iso_token is not None:
                detach_context(self._iso_token)
        except Exception:
            logger.debug("Failed to detach isolated context", exc_info=True)
        if self._did_push and self._wrapper is not None:
            try:
                self._wrapper.pop_span_context()
            except Exception:
                logger.debug("Failed to pop span context", exc_info=True)


class LaminarMiddleware(AgentMiddleware):  # type: ignore[misc,valid-type]
    """Emits Laminar TOOL spans around every agent tool call.

    Injected automatically by `DeepagentsInstrumentor` into every agent
    built via `deepagents.create_deep_agent`. Safe to add manually — it's a
    no-op when Laminar isn't initialised.

    The matching DEFAULT root span that parents these tool spans is opened
    by `DeepagentsInstrumentor._wrap_graph_methods` around the compiled
    graph's `invoke`/`stream`, not by `before_agent` / `after_agent` hooks:
    LangGraph runs middleware hooks as separate graph nodes, so OTel
    context attached in `before_agent` doesn't survive into later tool
    nodes. Wrapping `invoke` instead keeps the root span's context active
    for the entire graph execution.
    """

    def __init__(self) -> None:
        super().__init__()

    # ---- tool span ----

    @staticmethod
    def _tool_span_name(request: Any) -> str:
        call = getattr(request, "tool_call", None) or {}
        if isinstance(call, dict):
            name = call.get("name")
            if name:
                return f"{name}"
        tool = getattr(request, "tool", None)
        return getattr(tool, "name", None) or "tool"

    @staticmethod
    def _tool_span_input(request: Any) -> Any:
        call = getattr(request, "tool_call", None) or {}
        if isinstance(call, dict):
            return call.get("args")
        return None

    def _wrap_tool(self, request, handler):
        name = self._tool_span_name(request)
        with _SpanHandle(name, "TOOL") as handle:
            handle.set_input(self._tool_span_input(request))
            try:
                result = handler(request)
            except BaseException as e:
                handle.record_exception(e)
                raise
            handle.set_output(_tool_result_to_json(result))
            return result

    async def _awrap_tool(self, request, handler):
        name = self._tool_span_name(request)
        with _SpanHandle(name, "TOOL") as handle:
            handle.set_input(self._tool_span_input(request))
            try:
                result = await handler(request)
            except BaseException as e:
                handle.record_exception(e)
                raise
            handle.set_output(_tool_result_to_json(result))
            return result

    def wrap_tool_call(  # type: ignore[override]
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        if not TracerWrapper.verify_initialized():
            return handler(request)
        return self._wrap_tool(request, handler)

    async def awrap_tool_call(  # type: ignore[override]
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        if not TracerWrapper.verify_initialized():
            return await handler(request)
        return await self._awrap_tool(request, handler)

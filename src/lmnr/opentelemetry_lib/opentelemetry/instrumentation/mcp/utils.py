"""Request / result parsing and W3C trace-context carriage for MCP spans."""

from __future__ import annotations

from typing import Any, cast

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.types import AttributeValue

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    set_span_attribute,
    should_send_prompts,
)
from lmnr.opentelemetry_lib.tracing.attributes import SPAN_INPUT, SPAN_OUTPUT
from lmnr.opentelemetry_lib.tracing.context import get_event_attributes_from_context
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import LaminarSpanType
from lmnr.sdk.utils import json_dumps

logger = get_default_logger(__name__)

TOOLS_CALL = "tools/call"
TOOL_ERROR = "tool_error"

# the W3C headers `TraceContextTextMapPropagator` reads and writes
_TRACE_CONTEXT_KEYS = ("traceparent", "tracestate")

_propagator = TraceContextTextMapPropagator()


def _dump(model: Any) -> Any:
    # the same shape mcp puts on the wire
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=True, mode="json", exclude_none=True)
    return model


def request_root(request: Any) -> Any:
    # `ClientRequest` / `ServerRequest` are `RootModel`s over the concrete request
    return getattr(request, "root", request)


def request_method(request: Any) -> str:
    method = getattr(request_root(request), "method", None)
    return method if isinstance(method, str) else "unknown"


def tool_name(request: Any) -> str | None:
    root = request_root(request)
    if getattr(root, "method", None) != TOOLS_CALL:
        return None
    name = getattr(getattr(root, "params", None), "name", None)
    return name if isinstance(name, str) else None


def span_name_and_type(request: Any) -> tuple[str, LaminarSpanType]:
    name = tool_name(request)
    if name is not None:
        return f"{name}.tool", "TOOL"
    return f"{request_method(request)}.mcp", "DEFAULT"


def span_attributes(request: Any) -> dict[str, AttributeValue]:
    attributes: dict[str, AttributeValue] = {"mcp.method.name": request_method(request)}
    name = tool_name(request)
    if name is not None:
        attributes["gen_ai.tool.name"] = name
    return attributes


def set_input(span: Span, request: Any) -> None:
    if not should_send_prompts():
        return
    root = request_root(request)
    name = tool_name(request)
    span_input: Any
    if name is not None:
        span_input = {
            "tool_name": name,
            "arguments": getattr(getattr(root, "params", None), "arguments", None) or {},
        }
    else:
        span_input = _dump(root)
    set_span_attribute(span, SPAN_INPUT, json_dumps(span_input))


def set_output(span: Span, result: Any) -> None:
    if not should_send_prompts():
        return
    set_span_attribute(span, SPAN_OUTPUT, json_dumps(_dump(result)))


def record_error(span: Span, error: Exception, error_type: str | None = None) -> None:
    # the `exception` event (not the status) is what marks the span failed in Laminar
    error_type = error_type or error.__class__.__name__
    span.set_attribute(ERROR_TYPE, error_type)
    span.record_exception(
        error,
        attributes={**get_event_attributes_from_context(), "exception.type": error_type},
    )
    span.set_status(Status(StatusCode.ERROR, str(error)))


def _first_text(result: Any) -> str | None:
    content: list[Any] = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
    return None


def record_tool_error(span: Span, result: Any) -> None:
    """Mark a span failed for a `CallToolResult` with `isError` set.

    The tool's own error text is tool output, so it is withheld when content
    capture is off; the `tool_error` type still marks the failure.
    """
    message = (_first_text(result) if should_send_prompts() else None) or "Tool returned an error"
    record_error(span, Exception(message), TOOL_ERROR)


def is_tool_error(result: Any) -> bool:
    return getattr(result, "isError", False) is True


def with_trace_context(request: Any, span: Span) -> Any:
    """A copy of `request` whose `params._meta` carries `span`'s trace context.

    The receiving server reads it back in its `_handle_request` wrapper, so its
    span (and everything its handler does) joins this trace. A `traceparent`
    the caller set themselves is left alone. The caller's request is never
    mutated: on any failure the original is returned as-is.
    """
    try:
        data: Any = _dump(request)
        if not isinstance(data, dict):
            return request
        params: Any = cast(dict[str, Any], data).setdefault("params", {})
        if not isinstance(params, dict):
            return request
        meta: Any = cast(dict[str, Any], params).setdefault("_meta", {})
        if not isinstance(meta, dict) or "traceparent" in meta:
            return request
        carrier: dict[str, str] = {}
        _propagator.inject(carrier, context=trace.set_span_in_context(span))
        if "traceparent" not in carrier:
            return request
        cast(dict[str, Any], meta).update(carrier)
        request_type: Any = type(request)
        return request_type.model_validate(data)
    except Exception:
        logger.debug("Failed to inject trace context into MCP request", exc_info=True)
        return request


def extract_trace_context(meta: Any, base: Context | None = None) -> Context | None:
    """The remote parent carried in a request's `_meta`, or None without one.

    `RequestParams.Meta` allows extra fields, so the W3C headers the client
    injected land in its `model_extra`.
    """
    if meta is None:
        return None
    extra: dict[str, Any] = getattr(meta, "model_extra", None) or {}
    carrier: dict[str, str] = {}
    for key in _TRACE_CONTEXT_KEYS:
        value: Any = extra.get(key, getattr(meta, key, None))
        if isinstance(value, str):
            carrier[key] = value
    if "traceparent" not in carrier:
        return None
    try:
        context = _propagator.extract(carrier, context=base)
    except Exception:
        logger.debug("Failed to extract trace context from MCP request", exc_info=True)
        return None
    if not trace.get_current_span(context).get_span_context().is_valid:
        return None
    return context

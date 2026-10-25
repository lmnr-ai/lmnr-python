"""Span helpers, request / result parsing and W3C trace-context carriage for MCP spans.

Shared by the mcp 1.x (`v1.py`) and 2.x (`v2.py`) wrappers. Both majors hand
the client a typed request model (1.x wraps it in a `RootModel`) and a typed
result; 2.x servers see the raw method name and params instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from contextvars import ContextVar
from typing import Any, cast

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Span, SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.types import AttributeValue

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    safe_start_span,
    set_span_attribute,
    should_send_prompts,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    stamp_instrumentation_scope,
)
from lmnr.opentelemetry_lib.tracing.attributes import SPAN_INPUT, SPAN_OUTPUT
from lmnr.opentelemetry_lib.tracing.context import get_event_attributes_from_context
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import LaminarSpanType
from lmnr.sdk.utils import json_dumps

logger = get_default_logger(__name__)

TOOLS_CALL = "tools/call"
TOOL_ERROR = "tool_error"
RPC_STATUS_CODE = "rpc.response.status_code"

# the W3C headers `TraceContextTextMapPropagator` reads and writes
_TRACE_CONTEXT_KEYS = ("traceparent", "tracestate")

_propagator = TraceContextTextMapPropagator()

# The client span `wrap_send_request` is running in, so the mcp 2.x dispatcher
# reuses it instead of opening its own native span (see `v2.py`)
ACTIVE_CLIENT_SPAN: ContextVar[Span | None] = ContextVar("lmnr_mcp_client_span", default=None)


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


def name_and_type(method: str, tool: str | None) -> tuple[str, LaminarSpanType]:
    if tool is not None:
        return f"{tool}.tool", "TOOL"
    return f"{method}.mcp", "DEFAULT"


def attributes_for(method: str, tool: str | None) -> dict[str, AttributeValue]:
    attributes: dict[str, AttributeValue] = {"mcp.method.name": method}
    if tool is not None:
        attributes["gen_ai.tool.name"] = tool
    return attributes


def span_name_and_type(request: Any) -> tuple[str, LaminarSpanType]:
    return name_and_type(request_method(request), tool_name(request))


def span_attributes(request: Any) -> dict[str, AttributeValue]:
    return attributes_for(request_method(request), tool_name(request))


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


def set_raw_input(span: Span, method: str, params: Mapping[str, Any] | None) -> None:
    """Input for a request seen as a method name and its JSON params (mcp 2.x servers)."""
    if not should_send_prompts():
        return
    span_input: Any
    if method == TOOLS_CALL and isinstance((params or {}).get("name"), str):
        span_input = {
            "tool_name": (params or {})["name"],
            "arguments": (params or {}).get("arguments") or {},
        }
    else:
        span_input = {"method": method, "params": dict(params)} if params else {"method": method}
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
    set_status_code(span, error_code(error))


def error_code(error: Exception) -> Any:
    # `McpError` (1.x) / `MCPError` (2.x) carry the JSON-RPC `ErrorData`
    return getattr(getattr(error, "error", None), "code", None)


def set_status_code(span: Span, code: Any) -> None:
    if isinstance(code, int):
        span.set_attribute(RPC_STATUS_CODE, str(code))


def _get(obj: Any, key: str) -> Any:
    # results are models, or plain dicts a 2.x handler returned as-is
    if isinstance(obj, Mapping):
        return cast(Mapping[str, Any], obj).get(key)
    return getattr(obj, key, None)


def _first_text(result: Any) -> str | None:
    content: Any = _get(result, "content") or []
    if not isinstance(content, Sequence):
        return None
    for block in cast(Sequence[Any], content):
        text = _get(block, "text")
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
    # `isError` on 1.x models and on raw dicts, `is_error` on 2.x models
    if isinstance(result, Mapping):
        return cast(Mapping[str, Any], result).get("isError") is True
    return getattr(result, "isError", None) is True or getattr(result, "is_error", None) is True


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

    On 1.x `RequestParams.Meta` allows extra fields, so the W3C headers the
    client injected land in its `model_extra`; on 2.x `_meta` is a plain dict.
    """
    if meta is None:
        return None
    carrier: dict[str, str] = {}
    for key in _TRACE_CONTEXT_KEYS:
        value: Any
        if isinstance(meta, Mapping):
            value = cast(Mapping[str, Any], meta).get(key)
        else:
            extra: dict[str, Any] = getattr(meta, "model_extra", None) or {}
            value = extra.get(key, getattr(meta, key, None))
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


def start_span(
    to_wrap: WrappedFunctionSpec,
    request: Any,
    kind: SpanKind,
    context: Context | None = None,
) -> Span | None:
    name, span_type = span_name_and_type(request)
    span = safe_start_span(
        name=name,
        context=context,
        attributes=span_attributes(request),
        span_type=span_type,
        kind=kind,
    )
    if span is None:
        return None
    stamp_instrumentation_scope(span, to_wrap)
    dont_throw(set_input)(span, request)
    return span


def use_span(span: Span) -> AbstractContextManager[Any]:
    # exceptions are recorded by `record_error`, not by `use_span`
    return Laminar.use_span(span, record_exception=False, set_status_on_exception=False)


@dont_throw
def record_result(span: Span, result: Any) -> None:
    set_output(span, result)
    if is_tool_error(result):
        record_tool_error(span, result)


def replace_arg(
    args: Sequence[Any], kwargs: dict[str, Any], index: int, name: str, value: Any
) -> tuple[Sequence[Any], dict[str, Any]]:
    if len(args) > index:
        return (*args[:index], value, *args[index + 1 :]), kwargs
    return args, {**kwargs, name: value}


def get_arg(args: Sequence[Any], kwargs: dict[str, Any], index: int, name: str) -> Any:
    return args[index] if len(args) > index else kwargs.get(name)

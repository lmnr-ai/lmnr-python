"""Wrappers for the MCP Python SDK 2.x.

mcp 2.x ships its own OpenTelemetry spans, but they carry no input / output or
span type, mark tool errors with a status only (which Laminar does not read as
a failure), and are opened with `start_as_current_span`, which sets only the
global OTel context, so Laminar spans the tool's code opens don't nest under
them. Instead of enriching them, these wrappers take their place:

- `OpenTelemetryMiddleware.__call__`, the server middleware every `Server`
  installs by default, is replaced (the native span is never opened) with one
  that opens a Laminar SERVER span and runs the rest of the chain inside it.
  Servers that removed the middleware get no server spans, as without us.
- `ClientSession.send_request` / `ServerSession.send_request` get the same
  CLIENT span wrapper as 1.x (`v1.wrap_send_request`). The typed request and
  result give the span its input / output, and it covers the in-process
  `DirectDispatcher` too, which has no native spans. `ClientSession.send_discover`,
  the connect-time `server/discover` probe, bypasses `send_request` and gets
  its own CLIENT span.
- The dispatcher's `otel_span` reuses that CLIENT span instead of opening a
  native one, so the `traceparent` it injects into `_meta` is ours. The native
  span is not dropped in the span processor instead: the server span would
  then point at a parent that never arrives.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind
from pydantic import ValidationError

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.utils import (
    ACTIVE_CLIENT_SPAN,
    TOOLS_CALL,
    attributes_for,
    error_code,
    extract_trace_context,
    get_arg,
    name_and_type,
    record_error,
    record_result,
    set_raw_input,
    set_status_code,
    use_span,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.v1 import wrap_send_request
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
from lmnr.opentelemetry_lib.tracing.context import get_current_context

# JSON-RPC "Invalid params", what mcp answers a `ValidationError` with
_INVALID_PARAMS = -32602
SERVER_DISCOVER = "server/discover"


def _start_server_span(to_wrap: WrappedFunctionSpec, ctx: Any) -> Span | None:
    method: Any = getattr(ctx, "method", None)
    if not isinstance(method, str):
        return None
    params: Any = getattr(ctx, "params", None) or {}
    name: Any = params.get("name") if method == TOOLS_CALL else None
    tool = name if isinstance(name, str) else None
    span_name, span_type = name_and_type(method, tool)
    # The server handles each request in a task of its own, so the client's
    # context only arrives through `_meta`
    parent = extract_trace_context(getattr(ctx, "meta", None), get_current_context())
    span = safe_start_span(
        name=span_name,
        context=parent,
        attributes={**attributes_for(method, tool), "mcp.request.id": str(ctx.request_id)},
        span_type=span_type,
        kind=SpanKind.SERVER,
    )
    if span is None:
        return None
    stamp_instrumentation_scope(span, to_wrap)
    dont_throw(set_raw_input)(span, method, params)
    return span


def _record_server_error(span: Span, error: Exception) -> None:
    if isinstance(error, ValidationError):
        # pydantic messages quote the client's input
        if not should_send_prompts():
            error = Exception("Invalid request parameters")
        record_error(span, error, "ValidationError")
        set_status_code(span, _INVALID_PARAMS)
    else:
        # `MCPError` for a JSON-RPC error response, anything else a handler raised
        record_error(span, error)


async def _wrap_server_middleware(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[Any]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    """`OpenTelemetryMiddleware.__call__(ctx, call_next)`.

    Wraps every inbound request, `initialize` included, and every notification.
    """
    ctx = get_arg(args, kwargs, 0, "ctx")
    call_next = get_arg(args, kwargs, 1, "call_next")
    if (
        ctx is None
        or call_next is None
        or context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY)
    ):
        return await wrapped(*args, **kwargs)
    if getattr(ctx, "request_id", None) is None:
        # notifications are not traced, as on 1.x
        return await call_next(ctx)

    span = _start_server_span(to_wrap, ctx)
    if span is None:
        # Laminar is not initialized: leave mcp's own span in place
        return await wrapped(*args, **kwargs)
    try:
        with use_span(span):
            result = await call_next(ctx)
        # a `CallToolResult` or a plain dict, `isError` included
        record_result(span, result)
    except Exception as e:
        dont_throw(_record_server_error)(span, e)
        raise
    finally:
        span.end()
    return result


async def _wrap_send_discover(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[Any]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    """`ClientSession.send_discover(version)`, the connect-time probe.

    It builds its `server/discover` request itself and sends it straight to the
    dispatcher, bypassing `send_request`. A JSON-RPC error here is how a legacy
    server answers (the client falls back to `initialize`), so it gets
    `rpc.response.status_code` but no `exception` event.
    """
    if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)
    span_name, span_type = name_and_type(SERVER_DISCOVER, None)
    span = safe_start_span(
        name=span_name,
        attributes=attributes_for(SERVER_DISCOVER, None),
        span_type=span_type,
        kind=SpanKind.CLIENT,
    )
    if span is None:
        return await wrapped(*args, **kwargs)
    stamp_instrumentation_scope(span, to_wrap)
    version = get_arg(args, kwargs, 0, "version")
    if isinstance(version, str):
        set_span_attribute(span, "mcp.protocol.version", version)
    dont_throw(set_raw_input)(span, SERVER_DISCOVER, None)

    token = ACTIVE_CLIENT_SPAN.set(span)
    try:
        with use_span(span):
            result = await wrapped(*args, **kwargs)
            record_result(span, result)
    except Exception as e:
        code = error_code(e)
        if isinstance(code, int):
            # `MCPError`
            set_status_code(span, code)
        else:
            # timeouts, closed streams
            record_error(span, e)
        raise
    finally:
        ACTIVE_CLIENT_SPAN.reset(token)
        span.end()
    return result


@contextmanager
def _reuse_span(span: Span, attributes: dict[str, Any] | None) -> Generator[Span]:
    if attributes:
        # `jsonrpc.request.id`, which only the dispatcher knows
        span.set_attributes(attributes)
    yield span


def _wrap_otel_span(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Any],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    """`mcp.shared.jsonrpc_dispatcher.otel_span(name, *, kind, attributes, ...)`.

    Only the dispatcher's own binding is wrapped, so the outbound request span
    is the only native span affected. The error status it sets on a JSON-RPC
    error lands on our span, which then records the `MCPError` it raises.
    """
    span = ACTIVE_CLIENT_SPAN.get()
    # only when nothing was opened in between (`use_span` attached it globally)
    if span is None or trace.get_current_span() is not span:
        return wrapped(*args, **kwargs)
    return _reuse_span(span, kwargs.get("attributes"))


V2_WRAPPED_FUNCTIONS: list[WrappedFunctionSpec] = [
    WrappedFunctionSpec(
        package_name="mcp.client.session",
        object_name="ClientSession",
        method_name="send_request",
        is_async=True,
        wrapper_function=wrap_send_request,
    ),
    WrappedFunctionSpec(
        package_name="mcp.client.session",
        object_name="ClientSession",
        method_name="send_discover",
        is_async=True,
        wrapper_function=_wrap_send_discover,
    ),
    WrappedFunctionSpec(
        package_name="mcp.server.session",
        object_name="ServerSession",
        method_name="send_request",
        is_async=True,
        wrapper_function=wrap_send_request,
    ),
    WrappedFunctionSpec(
        package_name="mcp.server._otel",
        object_name="OpenTelemetryMiddleware",
        method_name="__call__",
        is_async=True,
        wrapper_function=_wrap_server_middleware,
    ),
    WrappedFunctionSpec(
        package_name="mcp.shared.jsonrpc_dispatcher",
        object_name=None,
        method_name="otel_span",
        is_async=False,
        wrapper_function=_wrap_otel_span,
    ),
]

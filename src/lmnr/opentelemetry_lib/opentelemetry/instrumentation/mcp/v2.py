"""Wrappers for the MCP Python SDK 2.x.

mcp 2.x ships its own OpenTelemetry spans, but they carry no input / output or
span type, mark tool errors with a status only (which Laminar does not read as
a failure), and are opened with `start_as_current_span`, which sets only the
global OTel context, so Laminar spans the tool's code opens don't nest under
them. Instead of enriching them, these wrappers take their place:

- `OpenTelemetryMiddleware.__call__`, the server middleware every `Server`
  installs by default, is replaced (the native span is never opened) by the
  shared server span (`server.py`) around the rest of the chain. Servers that
  removed the middleware get no server spans, as without us.
- In MCP observability mode (`Laminar.initialize(mcp=...)`) the span wraps the
  whole middleware chain instead (`ServerRunner._compose_server_middleware`,
  which knows its server; `ctx` does not), so it runs before params validation
  and also covers servers without the middleware (FastMCP 4 replaces it with
  its own). The middleware then just calls on.
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

import dataclasses
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, cast

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.context_field import (
    split_context,
    tools_with_context,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.fastmcp_compat import (
    native_span_suppression,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.server import ServerSpan
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.settings import (
    TOOLS_LIST,
    McpRequest,
    Mode,
    as_dict,
    traces,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.utils import (
    ACTIVE_CLIENT_SPAN,
    attributes_for,
    error_code,
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
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    stamp_instrumentation_scope,
)
from lmnr.opentelemetry_lib.tracing.mcp_options import get_mcp_options
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

CallNext = Callable[[Any], Awaitable[Any]]

SERVER_DISCOVER = "server/discover"


def _headers(ctx: Any) -> Mapping[str, str] | None:
    headers: Any = getattr(getattr(ctx, "request", None), "headers", None)
    if headers is None:
        # the transport metadata of the request's dispatch context
        outbound: Any = getattr(getattr(ctx, "session", None), "_request_outbound", None)
        headers = getattr(getattr(outbound, "transport", None), "headers", None)
    return cast(Mapping[str, str], headers) if isinstance(headers, Mapping) else None


def _request_from_context(ctx: Any) -> McpRequest:
    params = as_dict(getattr(ctx, "params", None))
    params.pop("_meta", None)
    headers = _headers(ctx)
    session: Any = getattr(ctx, "session", None)
    session_id: Any = getattr(getattr(session, "_connection", None), "session_id", None)
    if not isinstance(session_id, str) and headers is not None:
        session_id = headers.get("mcp-session-id")
    client_params: Any = getattr(session, "client_params", None)
    client_info: Any = getattr(client_params, "client_info", None)
    request_id: Any = getattr(ctx, "request_id", None)
    protocol_version: Any = getattr(ctx, "protocol_version", None)
    return McpRequest(
        method=str(getattr(ctx, "method", "")),
        params=params,
        meta=as_dict(getattr(ctx, "meta", None)),
        request_id=str(request_id) if request_id is not None else None,
        transport_session_id=session_id if isinstance(session_id, str) else None,
        protocol_version=protocol_version if isinstance(protocol_version, str) else None,
        http=headers is not None or None,
        client_name=getattr(client_info, "name", None),
        client_version=getattr(client_info, "version", None),
    )


async def _observe(
    to_wrap: WrappedFunctionSpec,
    mode: Mode,
    ctx: Any,
    call_next: CallNext,
    server: Any = None,
    suppress_native: Callable[[], AbstractContextManager[Any]] = nullcontext,
) -> Any:
    method: Any = getattr(ctx, "method", None)
    if (
        not isinstance(method, str)
        or getattr(ctx, "request_id", None) is None
        or not traces(mode, method)
    ):
        # notifications, and what observability mode doesn't trace
        result = await call_next(ctx)
        return tools_with_context(mode, result) if method == TOOLS_LIST else result
    try:
        request = _request_from_context(ctx)
        arguments, context_text = await split_context(mode, request, server)
    except Exception:
        logger.debug("Failed to read the MCP request", exc_info=True)
        return await call_next(ctx)
    if arguments is not None:
        # before params validation, so strict schemas don't reject `context`
        ctx = dataclasses.replace(ctx, params={**ctx.params, "arguments": arguments})
    span = ServerSpan.start(to_wrap, mode, request, context_text, suppress_native)
    return await span.run(call_next(ctx), record_result=True)


def _wrap_compose_server_middleware(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., CallNext],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> CallNext:
    """`ServerRunner._compose_server_middleware(inner)`, per inbound message.

    Only in MCP observability mode: the span goes around the whole chain.
    """
    chain = wrapped(*args, **kwargs)
    mode = get_mcp_options()
    if mode is None:
        return chain
    server: Any = getattr(instance, "server", None)
    suppress_native = native_span_suppression(server)

    async def observed(ctx: Any) -> Any:
        # also around FastMCP 4's own dispatch middleware, whose spans it turns off
        with suppress_native():
            return await _observe(to_wrap, mode, ctx, chain, server, suppress_native)

    return observed


async def _wrap_server_middleware(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[Any]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    """`OpenTelemetryMiddleware.__call__(ctx, call_next)`, for every inbound message."""
    ctx = get_arg(args, kwargs, 0, "ctx")
    call_next = get_arg(args, kwargs, 1, "call_next")
    if ctx is None or call_next is None:
        return await wrapped(*args, **kwargs)
    if get_mcp_options() is not None:
        # the runner hook already opened the span around the chain
        return await call_next(ctx)
    if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)
    if getattr(ctx, "request_id", None) is None:
        # notifications are not traced, as on 1.x
        return await call_next(ctx)
    if not Laminar.is_initialized():
        # leave mcp's own span in place
        return await wrapped(*args, **kwargs)
    return await _observe(to_wrap, None, ctx, call_next)


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
        package_name="mcp.server.runner",
        object_name="ServerRunner",
        method_name="_compose_server_middleware",
        is_async=False,
        wrapper_function=_wrap_compose_server_middleware,
    ),
    WrappedFunctionSpec(
        package_name="mcp.shared.jsonrpc_dispatcher",
        object_name=None,
        method_name="otel_span",
        is_async=False,
        wrapper_function=_wrap_otel_span,
    ),
]

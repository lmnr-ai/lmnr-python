"""Wrappers for the MCP Python SDK 1.x.

Two wrappers cover every transport (stdio, SSE, streamable HTTP, in-memory)
and every client or server built on the SDK, FastMCP included:

- `BaseSession.send_request`: one CLIENT span per outgoing request, whose
  trace context rides in the request's `params._meta` (W3C `traceparent`).
- `Server._handle_request`: the shared server span (`server.py`) per incoming
  request a low-level `Server` handles. The handler (and so the tool's own
  code) runs inside it.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, cast

from opentelemetry import context as context_api
from opentelemetry.trace import SpanKind
from typing_extensions import override
from wrapt import ObjectProxy

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
    as_dict,
    traces,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.utils import (
    ACTIVE_CLIENT_SPAN,
    get_arg,
    record_error,
    record_result,
    replace_arg,
    start_span,
    use_span,
    with_trace_context,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.tracing.mcp_options import get_mcp_options
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


async def wrap_send_request(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[Any]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    """`BaseSession.send_request(request, result_type, ...)` (1.x), or the
    same-shaped `ClientSession` / `ServerSession.send_request` (2.x).

    Covers the client side (`ClientSession`) and requests a server sends back
    to its client (`ServerSession`: sampling, elicitation, roots), which then
    nest under the server span of the request being handled.
    """
    request = get_arg(args, kwargs, 0, "request")
    if request is None or context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = start_span(to_wrap, request, SpanKind.CLIENT)
    if span is None:
        return await wrapped(*args, **kwargs)

    args, kwargs = replace_arg(args, kwargs, 0, "request", with_trace_context(request, span))
    token = ACTIVE_CLIENT_SPAN.set(span)
    try:
        with use_span(span):
            result = await wrapped(*args, **kwargs)
            record_result(span, result)
    except Exception as e:
        # `McpError` for a JSON-RPC error response, timeouts, closed streams
        record_error(span, e)
        raise
    finally:
        ACTIVE_CLIENT_SPAN.reset(token)
        span.end()
    return result


if TYPE_CHECKING:
    _ResponderBase = ObjectProxy[Any]
else:
    # wrapt 1.x proxies are not subscriptable at runtime
    _ResponderBase = ObjectProxy


class _Responder(_ResponderBase):
    """A `RequestResponder` that passes the server's response through `on_response`.

    `Server._handle_request` turns handler failures into an `ErrorData` and
    sends it with `respond` instead of raising, so this is the one place both
    results and errors are visible.
    """

    @override
    def __init__(self, wrapped: Any, on_response: Callable[[Any], Any]):
        super().__init__(wrapped)
        self._self_on_response: Callable[[Any], Any] = on_response

    async def respond(self, response: Any) -> None:
        await self.__wrapped__.respond(self._self_on_response(response))


def _negotiated_version(requested: Any) -> str | None:
    if not isinstance(requested, str):
        return None
    try:
        # mcp 1.x names (basedpyright checks against 2.x)
        supported: Any = importlib.import_module("mcp.shared.version").SUPPORTED_PROTOCOL_VERSIONS
        latest: Any = importlib.import_module("mcp.types").LATEST_PROTOCOL_VERSION
    except Exception:
        return requested
    # what `ServerSession` answers `initialize` with
    return requested if requested in supported else str(latest)


def _request_from(message: Any, req: Any, session: Any) -> McpRequest:
    params = as_dict(getattr(req, "params", None))
    # `RequestParams.Meta` allows extra fields, so the dump keeps `traceparent` & co.
    meta = as_dict(params.pop("_meta", None))
    # the HTTP request, on HTTP transports
    http_request: Any = getattr(getattr(message, "message_metadata", None), "request_context", None)
    raw_headers: Any = getattr(http_request, "headers", None)
    headers = cast(Mapping[str, str], raw_headers) if isinstance(raw_headers, Mapping) else None
    client_params: Any = getattr(session, "client_params", None)
    client_info: Any = getattr(client_params, "clientInfo", None)
    request_id: Any = getattr(message, "request_id", None)
    return McpRequest(
        method=str(getattr(req, "method", "")),
        params=params,
        meta=meta,
        request_id=str(request_id) if request_id is not None else None,
        transport_session_id=(headers or {}).get("mcp-session-id"),
        protocol_version=(headers or {}).get("mcp-protocol-version")
        or _negotiated_version(getattr(client_params, "protocolVersion", None)),
        http=headers is not None or None,
        client_name=getattr(client_info, "name", None),
        client_version=getattr(client_info, "version", None),
    )


def _recording(span: ServerSpan) -> Callable[[Any], Any]:
    def on_response(response: Any) -> Any:
        span.record_result(response)
        return response

    return on_response


async def _wrap_handle_request(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[Any]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    """`Server._handle_request(message, req, session, lifespan_context, raise_exceptions)`.

    `initialize` is answered by `ServerSession` itself and never reaches here.
    """
    message = get_arg(args, kwargs, 0, "message")
    req = get_arg(args, kwargs, 1, "req")
    if message is None or req is None:
        return await wrapped(*args, **kwargs)
    mode = get_mcp_options()
    suppress_native: Callable[[], AbstractContextManager[Any]] = nullcontext
    if mode is not None:
        suppress_native = native_span_suppression(instance)
    elif context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)
    with suppress_native():
        method: Any = getattr(req, "method", None)
        if not isinstance(method, str) or not traces(mode, method):
            if method == TOOLS_LIST:
                on_response = functools.partial(tools_with_context, mode)
                args, kwargs = replace_arg(
                    args, kwargs, 0, "message", _Responder(message, on_response)
                )
            return await wrapped(*args, **kwargs)
        try:
            request = _request_from(message, req, get_arg(args, kwargs, 2, "session"))
            arguments, context_text = await split_context(mode, request, instance)
            if arguments is not None:
                # before the handler, which is where 1.x validates arguments
                params = req.params.model_copy(update={"arguments": arguments})
                args, kwargs = replace_arg(
                    args, kwargs, 1, "req", req.model_copy(update={"params": params})
                )
        except Exception:
            logger.debug("Failed to read the MCP request", exc_info=True)
            return await wrapped(*args, **kwargs)
        span = ServerSpan.start(to_wrap, mode, request, context_text, suppress_native)
        args, kwargs = replace_arg(
            args, kwargs, 0, "message", _Responder(message, _recording(span))
        )
        # with `raise_exceptions=True` the handler's exception propagates here
        return await span.run(wrapped(*args, **kwargs), record_result=False)


V1_WRAPPED_FUNCTIONS: list[WrappedFunctionSpec] = [
    WrappedFunctionSpec(
        package_name="mcp.shared.session",
        object_name="BaseSession",
        method_name="send_request",
        is_async=True,
        wrapper_function=wrap_send_request,
    ),
    WrappedFunctionSpec(
        package_name="mcp.server.lowlevel.server",
        object_name="Server",
        method_name="_handle_request",
        is_async=True,
        wrapper_function=_wrap_handle_request,
    ),
]

"""OpenTelemetry instrumentation for the MCP Python SDK (`mcp` 1.x).

Two wrappers cover every transport (stdio, SSE, streamable HTTP, in-memory)
and every client or server built on the SDK, FastMCP included:

- `BaseSession.send_request`: one CLIENT span per outgoing request, whose
  trace context rides in the request's `params._meta` (W3C `traceparent`).
- `Server._handle_request`: one SERVER span per incoming request handled by a
  low-level `Server`, parented to the `traceparent` in the request's `_meta`.
  The handler (and so the tool's own code) runs inside it.

mcp 2.x ships its own OpenTelemetry spans and drops `BaseSession`, so neither
target exists there and this instrumentor installs nothing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Sequence
from contextlib import AbstractContextManager
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from opentelemetry import context as context_api
from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind
from typing_extensions import override
from wrapt import ObjectProxy

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.utils import (
    extract_trace_context,
    is_tool_error,
    record_error,
    record_tool_error,
    set_input,
    set_output,
    span_attributes,
    span_name_and_type,
    with_trace_context,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
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
from lmnr.opentelemetry_lib.tracing.context import get_current_context
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

_instruments = ("mcp >= 1.6.0",)


def _start_span(
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


def _use_span(span: Span) -> AbstractContextManager[Any]:
    # exceptions are recorded by `record_error`, not by `use_span`
    return Laminar.use_span(span, record_exception=False, set_status_on_exception=False)


@dont_throw
def _record_result(span: Span, result: Any) -> None:
    set_output(span, result)
    if is_tool_error(result):
        record_tool_error(span, result)


def _replace_arg(
    args: Sequence[Any], kwargs: dict[str, Any], index: int, name: str, value: Any
) -> tuple[Sequence[Any], dict[str, Any]]:
    if len(args) > index:
        return (*args[:index], value, *args[index + 1 :]), kwargs
    return args, {**kwargs, name: value}


def _get_arg(args: Sequence[Any], kwargs: dict[str, Any], index: int, name: str) -> Any:
    return args[index] if len(args) > index else kwargs.get(name)


async def _wrap_send_request(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[Any]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    """`BaseSession.send_request(request, result_type, ...)`.

    Covers the client side (`ClientSession`) and requests a server sends back
    to its client (`ServerSession`: sampling, elicitation, roots), which then
    nest under the server span of the request being handled.
    """
    request = _get_arg(args, kwargs, 0, "request")
    if request is None or context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = _start_span(to_wrap, request, SpanKind.CLIENT)
    if span is None:
        return await wrapped(*args, **kwargs)

    args, kwargs = _replace_arg(args, kwargs, 0, "request", with_trace_context(request, span))
    try:
        with _use_span(span):
            result = await wrapped(*args, **kwargs)
            _record_result(span, result)
    except Exception as e:
        # `McpError` for a JSON-RPC error response, timeouts, closed streams
        record_error(span, e)
        raise
    finally:
        span.end()
    return result


if TYPE_CHECKING:
    _ResponderBase = ObjectProxy[Any]
else:
    # wrapt 1.x proxies are not subscriptable at runtime
    _ResponderBase = ObjectProxy


class _TracedResponder(_ResponderBase):
    """A `RequestResponder` that records the server's response on its span.

    `Server._handle_request` turns handler failures into an `ErrorData` and
    sends it with `respond` instead of raising, so this is the one place both
    results and errors are visible.
    """

    @override
    def __init__(self, wrapped: Any, span: Span):
        super().__init__(wrapped)
        self._self_span: Span = span

    async def respond(self, response: Any) -> None:
        _record_response(self._self_span, response)
        await self.__wrapped__.respond(response)


@dont_throw
def _record_response(span: Span, response: Any) -> None:
    # a `ServerResult` (a `RootModel` over the concrete result) or an `ErrorData`
    if type(response).__name__ == "ErrorData":
        set_output(span, response)
        error_type = "McpError"
        record_error(span, Exception(getattr(response, "message", None) or error_type), error_type)
        return
    _record_result(span, getattr(response, "root", response))


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
    message = _get_arg(args, kwargs, 0, "message")
    request = _get_arg(args, kwargs, 1, "req")
    if (
        message is None
        or request is None
        or context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY)
    ):
        return await wrapped(*args, **kwargs)

    # The server handles each request in a task spawned from its own run loop,
    # so the client's context only arrives through `_meta`.
    meta = getattr(message, "request_meta", None) or getattr(
        getattr(request, "params", None), "meta", None
    )
    parent = extract_trace_context(meta, get_current_context())
    span = _start_span(to_wrap, request, SpanKind.SERVER, context=parent)
    if span is None:
        return await wrapped(*args, **kwargs)
    request_id = getattr(message, "request_id", None)
    if isinstance(request_id, (str, int)):
        set_span_attribute(span, "mcp.request.id", str(request_id))

    args, kwargs = _replace_arg(args, kwargs, 0, "message", _TracedResponder(message, span))
    try:
        with _use_span(span):
            return await wrapped(*args, **kwargs)
    except Exception as e:
        # only with `raise_exceptions=True`; otherwise the error is responded
        record_error(span, e)
        raise
    finally:
        span.end()


WRAPPED_FUNCTIONS: list[WrappedFunctionSpec] = [
    WrappedFunctionSpec(
        package_name="mcp.shared.session",
        object_name="BaseSession",
        method_name="send_request",
        is_async=True,
        wrapper_function=_wrap_send_request,
    ),
    WrappedFunctionSpec(
        package_name="mcp.server.lowlevel.server",
        object_name="Server",
        method_name="_handle_request",
        is_async=True,
        wrapper_function=_wrap_handle_request,
    ),
]


class McpInstrumentor(BaseLaminarInstrumentor):
    """An instrumentor for the MCP Python SDK."""

    _scope: LaminarInstrumentationScopeAttributes | None = None

    @override
    def __init__(self):
        super().__init__()
        self.instrumentor_config: LaminarInstrumentorConfig = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                mcp_version = version("mcp")
            except Exception:
                logger.debug("Failed to get mcp version", exc_info=True)
                mcp_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="mcp",
                version=mcp_version,
            )
        return self._scope

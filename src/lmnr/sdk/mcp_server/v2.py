"""`wrap_mcp_server` on mcp 2.x: a context-tier server middleware.

`Server.middleware` runs outermost-first around every inbound message, before
params validation, handler lookup and the handler. The wrapper puts
`ObservedServerMiddleware` first and drops the SDK's `OpenTelemetryMiddleware`
(the native span, and the auto-instrumentor's replacement of it), so a traced
request gets exactly one span: ours. Covers the low-level `Server`, `MCPServer`
(`_lowlevel_server`) and FastMCP 4 (`_mcp_server`, which already drops the
native middleware for its own seam span; that one is suppressed per request).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.observed import OBSERVED_REQUEST
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.mcp_server.config import McpServerObservability
from lmnr.sdk.mcp_server.observe import (
    TOOLS_CALL,
    TOOLS_LIST,
    TRACED_METHODS,
    McpCall,
    McpRequest,
    as_dict,
    split_context,
    tools_with_context,
)

logger = get_default_logger(__name__)

CallNext = Callable[[Any], Awaitable[Any]]

_NATIVE_MIDDLEWARE = ("mcp.server._otel", "OpenTelemetryMiddleware")


def _is_native_middleware(middleware: object) -> bool:
    cls = type(middleware)
    return (cls.__module__, cls.__name__) == _NATIVE_MIDDLEWARE


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


class ObservedServerMiddleware:
    """Opens the span of every traced request and maintains the `context` field."""

    def __init__(self, config: McpServerObservability) -> None:
        self.config: McpServerObservability = config

    async def __call__(self, ctx: Any, call_next: CallNext) -> Any:
        token = OBSERVED_REQUEST.set(getattr(ctx, "session", None))
        try:
            with self.config.suppress_native():
                method = getattr(ctx, "method", None)
                if getattr(ctx, "request_id", None) is None or method not in TRACED_METHODS:
                    result = await call_next(ctx)
                    if method == TOOLS_LIST and self.config.context_field():
                        return tools_with_context(self.config, result) or result
                    return result
                return await self._traced(ctx, call_next)
        finally:
            OBSERVED_REQUEST.reset(token)

    async def _traced(self, ctx: Any, call_next: CallNext) -> Any:
        forwarded = ctx
        request: McpRequest | None = None
        context_text: str | None = None
        try:
            request = _request_from_context(ctx)
            arguments, context_text = await split_context(self.config, request)
            if arguments is not None and request.method == TOOLS_CALL:
                # before params validation, so strict schemas don't reject it
                params = {**cast(Mapping[str, Any], ctx.params), "arguments": arguments}
                forwarded = dataclasses.replace(ctx, params=params)
        except Exception:
            logger.debug("Failed to prepare the MCP request span", exc_info=True)
            forwarded = ctx
        if request is None:
            return await call_next(ctx)

        call = McpCall.start(self.config, request, context_text)
        try:
            with call.activate():
                result = await call_next(forwarded)
        except BaseException as e:
            call.record_exception(e)
            raise
        else:
            call.record_result(result)
        finally:
            call.end()
        return result


def install(lowlevel: Any, config: McpServerObservability) -> None:
    middleware: list[Any] = lowlevel.middleware
    middleware[:] = [
        ObservedServerMiddleware(config),
        *(m for m in middleware if not _is_native_middleware(m)),
    ]
    schema_of: Any = getattr(lowlevel, "get_tool_input_schema", None)
    if callable(schema_of) and config.lookup_tool_schema is None:

        async def lookup(name: str) -> Mapping[str, Any] | None:
            schema: Any = schema_of(name)
            return cast(Mapping[str, Any], schema) if isinstance(schema, Mapping) else None

        config.lookup_tool_schema = lookup

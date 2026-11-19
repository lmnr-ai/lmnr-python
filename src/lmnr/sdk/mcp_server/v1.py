"""`wrap_mcp_server` on mcp 1.x: observed request handlers.

mcp 1.x has no middleware. `Server._handle_request` looks the handler up in
`Server.request_handlers` (keyed by request type), sets the request context and
awaits it; a raised exception becomes an `ErrorData` response. The wrapper
swaps `request_handlers` for `ObservedHandlers`, a dict that hands out traced
versions of the handlers it holds:

- handlers registered after wrapping (`@server.call_tool()` on a low-level
  server) are covered, since the lookup, not the registration, is wrapped;
- the traced handler sees the raised exception itself (its class name is the
  `error.type`), and the request context `_handle_request` set;
- `tools/call` arguments are rewritten before the handler, which is where 1.x's
  input validation runs (`Server.call_tool(validate_input=True)`).

`McpInstrumentor`'s `_handle_request` wrapper stands aside for a wrapped server
(`is_observed_server`), so a request still gets one span.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from typing_extensions import override

from lmnr.sdk.log import get_default_logger
from lmnr.sdk.mcp_server.config import McpServerObservability
from lmnr.sdk.mcp_server.observe import (
    PROMPTS_GET,
    RESOURCES_READ,
    TOOLS_CALL,
    TOOLS_LIST,
    McpCall,
    McpRequest,
    as_dict,
    split_context,
    tools_with_context,
)

logger = get_default_logger(__name__)

Handler = Callable[[Any], Awaitable[Any]]

# request type name -> method
_METHODS = {
    "CallToolRequest": TOOLS_CALL,
    "GetPromptRequest": PROMPTS_GET,
    "ReadResourceRequest": RESOURCES_READ,
    "ListToolsRequest": TOOLS_LIST,
}
_METHOD_NOT_FOUND = -32601


def _negotiated_version(requested: Any) -> str | None:
    if not isinstance(requested, str):
        return None
    try:
        # mcp 1.x names (the optional package; basedpyright checks against 2.x)
        supported: Any = importlib.import_module("mcp.shared.version").SUPPORTED_PROTOCOL_VERSIONS
        latest: Any = importlib.import_module("mcp.types").LATEST_PROTOCOL_VERSION
    except Exception:
        return requested
    # what `ServerSession` answers `initialize` with
    return requested if requested in supported else str(latest)


def request_from(server: Any, method: str, req: Any) -> McpRequest:
    params = as_dict(getattr(req, "params", None))
    meta = as_dict(params.pop("_meta", None))
    try:
        # set by `_handle_request` before it awaits the handler
        context: Any = server.request_context
    except Exception:
        context = None
    request_id: Any = getattr(context, "request_id", None)
    session: Any = getattr(context, "session", None)
    http_request: Any = getattr(context, "request", None)
    raw_headers: Any = getattr(http_request, "headers", None)
    headers = cast(Mapping[str, str], raw_headers) if isinstance(raw_headers, Mapping) else None
    client_params: Any = getattr(session, "client_params", None)
    client_info: Any = getattr(client_params, "clientInfo", None)
    protocol_version = (headers or {}).get("mcp-protocol-version") or _negotiated_version(
        getattr(client_params, "protocolVersion", None)
    )
    return McpRequest(
        method=method,
        params=params,
        meta=meta,
        request_id=str(request_id) if request_id is not None else None,
        transport_session_id=(headers or {}).get("mcp-session-id"),
        protocol_version=protocol_version,
        http=headers is not None or None,
        client_name=getattr(client_info, "name", None),
        client_version=getattr(client_info, "version", None),
    )


def _with_arguments(req: Any, arguments: dict[str, Any]) -> Any:
    params = req.params.model_copy(update={"arguments": arguments})
    return req.model_copy(update={"params": params})


async def _method_not_found(_req: Any) -> Any:
    # what `_handle_request` answers when no handler is registered (mcp 1.x names)
    error_type: Any = importlib.import_module("mcp.shared.exceptions").McpError
    error_data: Any = importlib.import_module("mcp.types").ErrorData
    raise error_type(error_data(code=_METHOD_NOT_FOUND, message="Method not found"))


class ObservedHandlers(dict[Any, Any]):
    """`Server.request_handlers` whose lookups return observed handlers."""

    def __init__(self, handlers: Mapping[Any, Any], server: Any, config: McpServerObservability):
        super().__init__(handlers)
        self.server: Any = server
        self.config: McpServerObservability = config

    @override
    def get(self, key: Any, default: Any = None) -> Any:
        handler = super().get(key, default)
        if handler is None and _METHODS.get(getattr(key, "__name__", "")) in (
            TOOLS_CALL,
            PROMPTS_GET,
            RESOURCES_READ,
        ):
            # a JSON-RPC error no handler saw is still a traced request
            handler = _method_not_found
        return self.observed(key, handler)

    @override
    def __getitem__(self, key: Any) -> Any:
        return self.observed(key, super().__getitem__(key))

    def observed(self, key: Any, handler: Any) -> Any:
        if not callable(handler):
            return handler
        method = _METHODS.get(getattr(key, "__name__", ""))
        config = self.config
        server = self.server
        handler = cast(Handler, handler)

        if method == TOOLS_LIST:

            async def list_tools(req: Any) -> Any:
                with config.suppress_native():
                    result = await handler(req)
                if not config.context_field():
                    return result
                return tools_with_context(config, result) or result

            return list_tools

        if method is None:

            async def untraced(req: Any) -> Any:
                with config.suppress_native():
                    return await handler(req)

            return untraced

        async def traced(req: Any) -> Any:
            with config.suppress_native():
                forwarded = req
                request: McpRequest | None = None
                context_text: str | None = None
                try:
                    request = request_from(server, method, req)
                    arguments, context_text = await split_context(config, request)
                    if arguments is not None:
                        forwarded = _with_arguments(req, arguments)
                except Exception:
                    logger.debug("Failed to prepare the MCP request span", exc_info=True)
                    forwarded = req
                if request is None:
                    return await handler(req)

                call = McpCall.start(config, request, context_text)
                try:
                    with call.activate():
                        result = await handler(forwarded)
                except BaseException as e:
                    call.record_exception(e)
                    raise
                else:
                    call.record_result(result)
                finally:
                    call.end()
                return result

        return traced


def install(lowlevel: Any, high_level: Any, config: McpServerObservability) -> None:
    handlers: Any = lowlevel.request_handlers
    lowlevel.request_handlers = ObservedHandlers(
        cast(Mapping[Any, Any], handlers), lowlevel, config
    )
    if config.lookup_tool_schema is None:
        config.lookup_tool_schema = _schema_lookup(lowlevel, high_level)


def _schema_lookup(
    lowlevel: Any, high_level: Any
) -> Callable[[str], Awaitable[Mapping[str, Any] | None]] | None:
    tool_manager: Any = getattr(high_level, "_tool_manager", None)
    get_tool: Any = getattr(high_level, "get_tool", None)
    cached: Any = getattr(lowlevel, "_get_cached_tool_definition", None)

    async def lookup(name: str) -> Mapping[str, Any] | None:
        schema: Any = None
        if tool_manager is not None and callable(getattr(tool_manager, "get_tool", None)):
            # mcp 1.x `FastMCP`
            schema = getattr(tool_manager.get_tool(name), "parameters", None)
        elif callable(get_tool):
            # FastMCP 3: `await server.get_tool(name)`
            tool: Any = get_tool(name)
            if isinstance(tool, Awaitable):
                tool = await cast(Awaitable[Any], tool)
            schema = getattr(tool, "parameters", None)
        elif callable(cached):
            # low-level `Server`: its `tools/list` cache (refreshed when missing)
            definition: Any = await cast(Any, cached)(name)
            schema = getattr(definition, "inputSchema", None)
        return cast(Mapping[str, Any], schema) if isinstance(schema, Mapping) else None

    return lookup

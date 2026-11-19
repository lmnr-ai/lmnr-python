"""`wrap_mcp_server`: attach MCP-observability settings to a server.

The spans come from `McpInstrumentor`'s server hooks, which read these
settings (`instrumentation/mcp/settings.py`): on mcp 1.x from the low-level
`Server` (`_handle_request`), on mcp 2.x from the `OpenTelemetryMiddleware`
instance this puts first in the server's middleware (its `ctx` has no
reference back to the server).
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Iterable, Mapping
from typing import Any, TypeVar, cast

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.redaction import normalize_keys
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.settings import (
    SETTINGS_ATTR,
    McpServerSettings,
    SchemaLookup,
    settings_of,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.turns import (
    DEFAULT_TURN_GAP_SECONDS,
    MAX_TURN_GAP_SECONDS,
    TurnTracker,
)
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.mcp_server.fastmcp_compat import native_span_suppression

logger = get_default_logger(__name__)

ServerT = TypeVar("ServerT")


def _low_level_server(server: Any) -> Any:
    """The mcp low-level `Server` behind `server` (or `server` itself), or None."""
    for attr in ("_lowlevel_server", "_mcp_server"):
        # `MCPServer` (mcp 2.x); mcp 1.x `FastMCP` and standalone FastMCP
        inner = getattr(server, attr, None)
        if inner is not None:
            return inner
    if hasattr(server, "middleware") or hasattr(server, "request_handlers"):
        return server
    return None


def _clamped_turn_gap(turn_gap_seconds: float) -> float:
    if turn_gap_seconds > MAX_TURN_GAP_SECONDS:
        logger.warning(
            "wrap_mcp_server: turn_gap_seconds=%s is above the %ss maximum (Signals analyse "
            + "a trace once it has been quiet for 30s); using %ss.",
            turn_gap_seconds,
            MAX_TURN_GAP_SECONDS,
            MAX_TURN_GAP_SECONDS,
        )
        return MAX_TURN_GAP_SECONDS
    return max(0.0, turn_gap_seconds)


def _schema_lookup(lowlevel: Any, high_level: Any) -> SchemaLookup:
    """The framework's own answer to "what is this tool's input schema?" (None: unknown)."""
    # mcp 2.x `Server` (set by `MCPServer` and FastMCP 4)
    schema_of: Any = getattr(lowlevel, "get_tool_input_schema", None)
    tool_manager: Any = getattr(high_level, "_tool_manager", None)
    get_tool: Any = getattr(high_level, "get_tool", None)
    cached: Any = getattr(lowlevel, "_get_cached_tool_definition", None)

    async def lookup(name: str) -> Mapping[str, Any] | None:
        schema: Any = None
        if callable(schema_of):
            schema = schema_of(name)
        elif tool_manager is not None and callable(getattr(tool_manager, "get_tool", None)):
            # mcp 1.x `FastMCP`
            schema = getattr(tool_manager.get_tool(name), "parameters", None)
        elif callable(get_tool):
            # FastMCP 3: `await server.get_tool(name)`
            tool: Any = get_tool(name)
            if isinstance(tool, Awaitable):
                tool = await cast(Awaitable[Any], tool)
            schema = getattr(tool, "parameters", None)
        elif callable(cached):
            # 1.x low-level `Server`: its `tools/list` cache (refreshed when missing)
            definition: Any = await cast(Any, cached)(name)
            schema = getattr(definition, "inputSchema", None)
        return cast(Mapping[str, Any], schema) if isinstance(schema, Mapping) else None

    return lookup


def _put_settings_middleware_first(middleware: list[Any], settings: McpServerSettings) -> None:
    """mcp 2.x: one `OpenTelemetryMiddleware` carrying `settings`, outermost.

    Outermost so it runs before params validation (the `context` argument is
    stripped there) and around FastMCP 4's own dispatch middleware (whose spans
    it suppresses). Existing instances are dropped: each would open a span.
    """
    otel_middleware: Any = importlib.import_module("mcp.server._otel").OpenTelemetryMiddleware
    instance = otel_middleware()
    setattr(instance, SETTINGS_ATTR, settings)
    middleware[:] = [instance, *(m for m in middleware if type(m) is not otel_middleware)]


def wrap_mcp_server(
    server: ServerT,
    *,
    context: bool = True,
    record_payloads: bool = True,
    redact_keys: Iterable[str] | None = None,
    turn_gap_seconds: float = DEFAULT_TURN_GAP_SECONDS,
) -> ServerT:
    """Trace every request an MCP server handles in Laminar.

    Each `tools/call`, `prompts/get` and `resources/read` becomes a root
    SERVER span (failures included); listing, `initialize`, `ping` and
    notifications are not traced. Calls of one user turn share a trace. Spans
    the handler opens (LLM calls and so on) nest under the request's span.

    Supports the official `mcp` SDK (1.x `FastMCP` and low-level `Server`,
    2.x `MCPServer` and low-level `Server`) and standalone FastMCP 3 and 4.
    Call it once, before the server starts serving; `Laminar.initialize()` may
    come before or after (the spans come from its MCP instrumentation, which is
    on by default). Returns the same server object; wrapping twice is a no-op.

    Args:
        server: The MCP server object.
        context: Add an optional `context` argument to every tool, asking the
            model for one sentence on the user's overall goal. It is removed
            before your tool runs and recorded as `lmnr.mcp.context`. Tools
            that already have a `context` argument are left alone.
        record_payloads: Record arguments, results and `context`. False keeps
            only names, counts, timings and errors (and turns `context` off).
        redact_keys: Extra argument / result keys whose values are replaced
            with `[REDACTED]`, on top of the built-in secret-looking keys.
        turn_gap_seconds: Calls of one session at most this far apart join the
            same trace. At most 25 (larger values are clamped).
    """
    try:
        lowlevel = _low_level_server(server)
        if lowlevel is None:
            logger.warning(
                "wrap_mcp_server: %s is not a supported MCP server; it is left unwrapped.",
                type(server).__name__,
            )
            return server
        if settings_of(lowlevel) is not None:
            return server

        settings = McpServerSettings(
            turns=TurnTracker(_clamped_turn_gap(turn_gap_seconds)),
            context=context,
            record_payloads=record_payloads,
            redact_keys=normalize_keys(redact_keys),
            lookup_tool_schema=_schema_lookup(lowlevel, server),
        )
        suppression = native_span_suppression(server)
        if suppression is not None:
            settings.suppress_native = suppression

        middleware: Any = getattr(lowlevel, "middleware", None)
        if isinstance(middleware, list):
            _put_settings_middleware_first(cast(list[Any], middleware), settings)
        elif not isinstance(getattr(lowlevel, "request_handlers", None), dict):
            logger.warning(
                "wrap_mcp_server: unsupported mcp version for %s; it is left unwrapped.",
                type(server).__name__,
            )
            return server
        setattr(lowlevel, SETTINGS_ATTR, settings)
    except Exception:
        logger.warning("wrap_mcp_server: failed to wrap the server", exc_info=True)
    return server

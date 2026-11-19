"""`wrap_mcp_server`: one Laminar span per MCP request a server handles."""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import version
from typing import Any, TypeVar

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.observed import (
    OBSERVED_SERVER_ATTR,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
)
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.mcp_server.config import McpServerObservability
from lmnr.sdk.mcp_server.fastmcp_compat import native_span_suppression
from lmnr.sdk.mcp_server.redaction import normalize_keys
from lmnr.sdk.mcp_server.turns import (
    DEFAULT_TURN_GAP_SECONDS,
    MAX_TURN_GAP_SECONDS,
    TurnTracker,
)
from lmnr.sdk.mcp_server.v1 import install as install_v1
from lmnr.sdk.mcp_server.v2 import install as install_v2

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


def _scope() -> LaminarInstrumentationScopeAttributes:
    try:
        mcp_version = version("mcp")
    except Exception:
        mcp_version = "unknown"
    return LaminarInstrumentationScopeAttributes(name="mcp", version=mcp_version)


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
    come before or after. Returns the same server object, wrapped in place;
    wrapping twice is a no-op.

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
        if getattr(lowlevel, OBSERVED_SERVER_ATTR, None) is not None:
            return server

        config = McpServerObservability(
            context=context,
            record_payloads=record_payloads,
            redact_keys=normalize_keys(redact_keys),
            turns=TurnTracker(_clamped_turn_gap(turn_gap_seconds)),
            scope=_scope(),
        )
        suppression = native_span_suppression(server)
        if suppression is not None:
            config.suppress_native = suppression

        middleware: Any = getattr(lowlevel, "middleware", None)
        if isinstance(middleware, list):
            install_v2(lowlevel, config)
        elif isinstance(getattr(lowlevel, "request_handlers", None), dict):
            install_v1(lowlevel, server, config)
        else:
            logger.warning(
                "wrap_mcp_server: unsupported mcp version for %s; it is left unwrapped.",
                type(server).__name__,
            )
            return server
        setattr(lowlevel, OBSERVED_SERVER_ATTR, config)
        if lowlevel is not server:
            try:
                setattr(server, OBSERVED_SERVER_ATTR, config)
            except Exception:
                # e.g. `__slots__`; the low-level server's marker is what counts
                logger.debug("Failed to mark the MCP server object", exc_info=True)
    except Exception:
        logger.warning("wrap_mcp_server: failed to wrap the server", exc_info=True)
    return server

"""The `mcp` option of `Laminar.initialize`: MCP observability mode.

Leaf store (like `instrument_options.py`): `Laminar.initialize` sets it, the MCP
instrumentation reads it per request. None means the mode is off.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import McpOptions

logger = get_default_logger(__name__)

DEFAULT_TURN_GAP_SECONDS = 20.0
# Signals wait until a trace has been quiet for 30 s before analysing it, so a
# turn gap at or above that would let a Signal run mid-turn.
MAX_TURN_GAP_SECONDS = 25.0


@dataclass(frozen=True)
class ResolvedMcpOptions:
    context: bool = True
    record_payloads: bool = True
    redact_keys: tuple[str, ...] = ()
    turn_gap_seconds: float = DEFAULT_TURN_GAP_SECONDS


_resolved: ResolvedMcpOptions | None = None


def _turn_gap(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        logger.warning(
            "Ignoring mcp turn_gap_seconds=%r; using %ss.", value, DEFAULT_TURN_GAP_SECONDS
        )
        return DEFAULT_TURN_GAP_SECONDS
    if value > MAX_TURN_GAP_SECONDS:
        logger.warning(
            "mcp turn_gap_seconds=%s is above the %ss maximum (Signals analyse a trace "
            + "once it has been quiet for 30s); using %ss.",
            value,
            MAX_TURN_GAP_SECONDS,
            MAX_TURN_GAP_SECONDS,
        )
        return MAX_TURN_GAP_SECONDS
    return float(value)


def resolve_mcp_options(mcp: bool | McpOptions | None) -> ResolvedMcpOptions | None:
    if mcp is None or mcp is False:
        return None
    if mcp is True:
        return ResolvedMcpOptions()
    # a dict from user code may hold anything
    redact_keys = cast(Iterable[object], mcp.get("redact_keys") or ())
    return ResolvedMcpOptions(
        context=mcp.get("context", True) is not False,
        record_payloads=mcp.get("record_payloads", True) is not False,
        redact_keys=tuple(k for k in redact_keys if isinstance(k, str)),
        turn_gap_seconds=_turn_gap(mcp.get("turn_gap_seconds", DEFAULT_TURN_GAP_SECONDS)),
    )


def set_mcp_options(mcp: bool | McpOptions | None) -> None:
    global _resolved
    _resolved = resolve_mcp_options(mcp)


def reset_mcp_options() -> None:
    global _resolved
    _resolved = None


def get_mcp_options() -> ResolvedMcpOptions | None:
    return _resolved

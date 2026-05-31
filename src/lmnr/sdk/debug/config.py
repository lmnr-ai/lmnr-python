"""Debug-mode environment parsing.

Reads the `LMNR_DEBUG*` env vars once at SDK init and produces an immutable
`DebugConfig`. See the shared debugger spec (§4, §5) for the authoritative
contract. This file is part of the cross-language parity surface — keep its
logic line-comparable with the TS `config.ts`.
"""

import os
import uuid
from dataclasses import dataclass

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

# Truthy set for LMNR_DEBUG, case-insensitive. Must match the TS SDK exactly.
_TRUTHY = {"true", "1", "yes", "on"}


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUTHY


def _parse_cache_until(value: str | None) -> int:
    """Parse N: clamp <0 to 0, non-numeric to 0 (with a warning)."""
    if value is None or value == "":
        return 0
    try:
        n = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "LMNR_DEBUG_CACHE_UNTIL=%r is not an integer; defaulting to 0", value
        )
        return 0
    return n if n > 0 else 0


@dataclass(frozen=True)
class DebugConfig:
    """Immutable debug configuration, built once at process start."""

    session_id: str
    replay_trace_id: str | None
    cache_until: int

    @property
    def replay_enabled(self) -> bool:
        """True when a source trace is set and at least one call should replay."""
        return self.replay_trace_id is not None and self.cache_until > 0


def build_debug_config() -> DebugConfig | None:
    """Build the debug config from the environment.

    Returns None when debug mode is disabled (LMNR_DEBUG falsey/absent) — the
    caller treats None as "everything inert".
    """
    if not _is_truthy(os.environ.get("LMNR_DEBUG")):
        return None

    session_id = os.environ.get("LMNR_DEBUG_SESSION_ID") or str(uuid.uuid4())
    replay_trace_id = os.environ.get("LMNR_DEBUG_REPLAY_TRACE_ID") or None
    cache_until = _parse_cache_until(os.environ.get("LMNR_DEBUG_CACHE_UNTIL"))

    return DebugConfig(
        session_id=session_id,
        replay_trace_id=replay_trace_id,
        cache_until=cache_until,
    )

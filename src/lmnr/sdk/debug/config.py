"""Debug-mode environment parsing.

Reads the `LMNR_DEBUG*` env vars once at SDK init and produces an immutable
`DebugConfig`. See the shared debugger spec (§4, §5) for the authoritative
contract. This file is part of the cross-language parity surface — keep its
logic line-comparable with the TS `config.ts`.
"""

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any

from lmnr.sdk.debug.pointer import POINTER_DIR, POINTER_FILE
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


def _load_last_run() -> dict[str, Any]:
    """Read `${CWD}/.lmnr/last-run.json` (the previous run's pointer).

    Best-effort: a missing / unreadable / malformed file returns {} so the
    caller silently falls back to env vars. Keep line-comparable with the TS
    `loadLastRun`.
    """
    try:
        path = os.path.join(os.getcwd(), POINTER_DIR, POINTER_FILE)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("Could not read debug pointer file: %s", exc)
        return {}


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

    When `LMNR_DEBUG_FROM_LAST_RUN` is truthy, seed the config from the previous
    run's pointer file (`${CWD}/.lmnr/last-run.json`): the file's `trace_id` (the
    trace that run produced) becomes this run's `replay_trace_id`, and its
    `session_id` / `cache_until` are reused. Individual `LMNR_DEBUG_*` env vars
    still override per-field, so the agent can replay the last run without
    copying its ids into the environment by hand.
    """
    if not _is_truthy(os.environ.get("LMNR_DEBUG")):
        return None

    last_run = (
        _load_last_run()
        if _is_truthy(os.environ.get("LMNR_DEBUG_FROM_LAST_RUN"))
        else {}
    )

    session_id = (
        os.environ.get("LMNR_DEBUG_SESSION_ID")
        or last_run.get("session_id")
        or str(uuid.uuid4())
    )
    replay_trace_id = (
        os.environ.get("LMNR_DEBUG_REPLAY_TRACE_ID")
        or last_run.get("trace_id")
        or None
    )
    cache_until_value = os.environ.get("LMNR_DEBUG_CACHE_UNTIL")
    if cache_until_value is None and last_run.get("cache_until") is not None:
        cache_until_value = str(last_run.get("cache_until"))
    cache_until = _parse_cache_until(cache_until_value)

    return DebugConfig(
        session_id=session_id,
        replay_trace_id=replay_trace_id,
        cache_until=cache_until,
    )

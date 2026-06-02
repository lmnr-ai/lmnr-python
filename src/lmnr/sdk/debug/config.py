"""Debug-mode environment parsing.

Reads the `LMNR_DEBUG*` env vars once at SDK init and produces an immutable
`DebugConfig`. See the shared debugger spec (§4, §5) for the authoritative
contract. This file is part of the cross-language parity surface — keep its
logic line-comparable with the TS `config.ts`.
"""

import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any

from lmnr.sdk.debug.pointer import POINTER_DIR, POINTER_FILE
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

# Truthy set for LMNR_DEBUG, case-insensitive. Must match the TS SDK exactly.
_TRUTHY = {"true", "1", "yes", "on"}

# A span-id needle is hyphen-stripped, lowercased hex of at most 32 chars (a
# full UUID is 32 hex). Must match the TS SDK exactly.
_HEX_RE = re.compile(r"^[0-9a-f]+$")


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUTHY


def _normalize_span_id(value: str) -> str | None:
    """Normalize a span-id-shaped value to a hyphen-stripped lowercase hex needle.

    Accepts any of the spans a user might copy from the UI: a full UUID
    (`00000000-0000-0000-0123-456789abcdef`), the last two UUID groups
    (`0123-456789abcdef`), the raw 16-hex OTel span id (`0123456789abcdef`), or a
    short hex suffix (`abcdef`). Returns the needle (hyphens removed, lowercased)
    that `match_span_id` suffix-matches against a span's UUID, or None when the
    value isn't span-id-shaped (not hex, empty, or longer than a full UUID).
    """
    needle = value.strip().lower().replace("-", "")
    if needle and len(needle) <= 32 and _HEX_RE.match(needle):
        return needle
    return None


def _parse_cache_until(value: str | None) -> tuple[int, str | None]:
    """Parse LMNR_DEBUG_CACHE_UNTIL into (N, span_id_needle).

    The value is either a count N (clamp <0 to 0) or a span id in UUID shape that
    is resolved to N once the source trace is fetched. A numeric value always
    wins, so a purely decimal string is treated as a count, never a span id.
    Returns (0, None) for an empty value and warns (then returns (0, None)) for a
    value that is neither a number nor a span id.
    """
    if value is None or value == "":
        return 0, None
    try:
        n = int(value)
    except (TypeError, ValueError):
        needle = _normalize_span_id(value)
        if needle is not None:
            return 0, needle
        logger.warning(
            "LMNR_DEBUG_CACHE_UNTIL=%r is neither an integer nor a span id; "
            "defaulting to 0",
            value,
        )
        return 0, None
    return (n if n > 0 else 0), None


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
    # When LMNR_DEBUG_CACHE_UNTIL was given as a span id rather than a count, this
    # holds the hyphen-stripped lowercase hex needle; `cache_until` stays 0 until
    # the source trace is fetched and the needle is resolved to an occurrence
    # count (see `_build_cache`). None when the value was a plain count.
    cache_until_span_id: str | None = None

    @property
    def replay_enabled(self) -> bool:
        """True when a source trace is set and at least one call should replay.

        A span-id cache window also counts as replay-configured even though
        `cache_until` is still 0 — the count is resolved later from the source
        trace, so gating on `cache_until > 0` alone would wrongly treat a
        span-id run as no-replay before the trace is fetched.
        """
        if self.replay_trace_id is None:
            return False
        return self.cache_until > 0 or self.cache_until_span_id is not None


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
    cache_until, cache_until_span_id = _parse_cache_until(cache_until_value)

    return DebugConfig(
        session_id=session_id,
        replay_trace_id=replay_trace_id,
        cache_until=cache_until,
        cache_until_span_id=cache_until_span_id,
    )

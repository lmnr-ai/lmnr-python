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


def _parse_cache_until_span_id(value: str | None) -> str | None:
    """Parse LMNR_DEBUG_CACHE_UNTIL into a span-id needle.

    v2 dropped the count form (shared spec §3): the value is **only** a span id.
    The four needle forms (full UUID / last-two UUID groups / raw 16-hex / short
    hex suffix) are accepted via `_normalize_span_id` and sent verbatim to
    app-server, which does the suffix match against the source trace's span ids.
    Returns None for an empty value and warns (then returns None) for a value
    that isn't span-id-shaped.
    """
    if value is None or value == "":
        return None
    needle = _normalize_span_id(value)
    if needle is None:
        logger.warning(
            "LMNR_DEBUG_CACHE_UNTIL=%r is not a span id; replay disabled",
            value,
        )
    return needle


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
    # The hyphen-stripped lowercase hex span-id needle parsed from
    # LMNR_DEBUG_CACHE_UNTIL. v2 has no count form — app-server resolves the
    # needle against the source trace (shared spec §3, §6.2). None when unset.
    cache_until_span_id: str | None = None

    @property
    def replay_enabled(self) -> bool:
        """True when both replay env vars resolved non-empty (shared spec §4).

        Replay needs a source trace AND a `cache_until` span-id needle; either
        one missing is a debug-no-replay run.
        """
        return (
            self.replay_trace_id is not None
            and self.cache_until_span_id is not None
        )


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
        # The pointer persists the needle as-is (v2 has no resolution step), so
        # the stored value is already a span-id string.
        cache_until_value = str(last_run.get("cache_until"))
    cache_until_span_id = _parse_cache_until_span_id(cache_until_value)

    return DebugConfig(
        session_id=session_id,
        replay_trace_id=replay_trace_id,
        cache_until_span_id=cache_until_span_id,
    )

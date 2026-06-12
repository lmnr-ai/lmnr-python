"""Debug-mode environment parsing.

Reads the `LMNR_DEBUG*` env vars once at SDK init and produces an immutable
`DebugConfig`. See the shared debugger spec (§4, §5) for the authoritative
contract. This file is part of the cross-language parity surface — keep its
logic line-comparable with the TS `config.ts`.
"""

import os
import re
import uuid
from dataclasses import dataclass
from typing import Any

from lmnr.sdk.debug.debug_session_file import read_debug_session_file
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


@dataclass(frozen=True)
class DebugConfig:
    """Immutable debug configuration, built once at process start."""

    session_id: str
    replay_trace_id: str | None
    # The hyphen-stripped lowercase hex span-id needle parsed from
    # LMNR_DEBUG_CACHE_UNTIL. v2 has no count form — app-server resolves the
    # needle against the source trace (shared spec §3, §6.2). None when unset.
    cache_until_span_id: str | None = None
    # True when this process is the ORIGIN of the debug run (config built from
    # local env vars). False when the config was inherited from an upstream
    # service via a propagated LaminarSpanContext debug block — a downstream run
    # must not open the browser or emit the run pointer (the origin owns both).
    local_origin: bool = True
    # True ONLY when the SDK minted a fresh session id — i.e. neither an explicit
    # LMNR_DEBUG_SESSION_ID nor an existing `.lmnr/debug-session.json` supplied
    # one. Gates the one-time browser open: a reused session id (explicit env or
    # a continuation from the file) is NOT a fresh run, so the browser is not
    # reopened. Lets the browser decision read the resolved config instead of
    # re-reading env vars at the call site.
    session_minted: bool = False

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

    The `.lmnr/debug-session.json` file is now DEFAULT-ON: it is read
    unconditionally (no opt-in env gate). The decision tree:

      1. Read the file (best-effort → existing | None).
      2. `session_id` precedence: LMNR_DEBUG_SESSION_ID → existing["session_id"]
         → a freshly-minted UUID.
      3. `session_minted` is True ONLY when we minted (neither env nor file
         supplied a session id). A file-present run is a CONTINUATION: it rejoins
         the same session silently (no browser).
      4. `replay_trace_id`: LMNR_DEBUG_REPLAY_TRACE_ID → existing["replay_trace_id"]
         → None. NOTE: continuation is NOT auto-replay — we do NOT promote the
         prior run's `trace_id` into `replay_trace_id`. Replay is armed explicitly
         by the agent via the env var or the file's `replay_trace_id`.
      5. `cache_until_span_id`: parsed from LMNR_DEBUG_CACHE_UNTIL ??
         existing["cache_until"].
    """
    if not _is_truthy(os.environ.get("LMNR_DEBUG")):
        return None

    existing = read_debug_session_file()

    provided_session_id = (
        os.environ.get("LMNR_DEBUG_SESSION_ID")
        or (existing.get("session_id") if existing else None)
        or None
    )
    session_id = provided_session_id or str(uuid.uuid4())
    # The browser is opened once per fresh run; a reused session id (explicit env
    # OR a continuation from the file) is NOT a fresh run, so it is not reopened.
    session_minted = provided_session_id is None
    # Continuation is NOT replay: only an explicit replay trace (env or the file's
    # `replay_trace_id`) arms replay. The prior run's `trace_id` is never promoted.
    replay_trace_id = (
        os.environ.get("LMNR_DEBUG_REPLAY_TRACE_ID")
        or (existing.get("replay_trace_id") if existing else None)
        or None
    )
    cache_until_value = os.environ.get("LMNR_DEBUG_CACHE_UNTIL")
    if cache_until_value is None and existing is not None:
        # The file persists the needle as-is (v2 has no resolution step), so the
        # stored value is already a span-id string.
        cache_until_value = existing.get("cache_until")
    cache_until_span_id = _parse_cache_until_span_id(cache_until_value)

    return DebugConfig(
        session_id=session_id,
        replay_trace_id=replay_trace_id,
        cache_until_span_id=cache_until_span_id,
        local_origin=True,
        session_minted=session_minted,
    )


def build_debug_config_from_context(debug: Any) -> DebugConfig | None:
    """Build a debug config from a propagated `DebugContext` (the inherited path).

    Unlike `build_debug_config` (which reads `LMNR_DEBUG*`), this consumes the
    debug block carried inside a `LaminarSpanContext` that arrived from an
    upstream service. It is the context-based half of the cross-language parity
    surface — keep line-comparable with the TS `buildDebugConfigFromContext`.

    Returns None when the block is absent or not armed (`enabled` is falsey). We
    are the only producer of this block, so an unarmed/forged block is treated
    as absent (behaviour explicitly undefined). A block with no session id is
    also ignored — without it the run can't be tied to a cache window.

    The resulting config is marked `local_origin=False`: a downstream run reuses
    the upstream session and may consult the cache, but must NOT open a browser
    or emit the run pointer (the origin owns those).
    """
    if debug is None:
        return None
    enabled = getattr(debug, "enabled", None)
    session_id = getattr(debug, "session_id", None)
    if not enabled or not session_id:
        return None

    replay_trace_id = getattr(debug, "replay_trace_id", None) or None
    # The needle is propagated verbatim; re-run it through the same normalizer
    # the env path uses so a downstream config holds an identical needle form.
    cache_until_span_id = _parse_cache_until_span_id(
        getattr(debug, "cache_until", None)
    )

    return DebugConfig(
        session_id=session_id,
        replay_trace_id=replay_trace_id,
        cache_until_span_id=cache_until_span_id,
        local_origin=False,
    )

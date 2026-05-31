"""In-process debug runtime.

A debug run is just a normal process execution started with `LMNR_DEBUG*` env
vars. At SDK init the runtime is built once from the environment (§4, §5):

1. Parse `DebugConfig` from the environment.
2. If replay is enabled, fetch the source trace's LLM spans (two-phase),
   detect the spine (§7), guard against overlap (§F), and build the in-process
   `ReplayCache`. Any failure here degrades to debug-no-replay (warn, never
   crash the user's program).
3. The run's pointer (§5) is emitted once at process shutdown, after the root
   trace id of this run is known.

The provider LLM wrappers consult `get_runtime()` to decide replay-vs-live; init
stamps `rollout.session_id` on the trace metadata. When debug mode is off,
`get_runtime()` returns None and everything is inert (§8).
"""

import datetime
import threading
from typing import Any

from lmnr.sdk.debug.config import DebugConfig, build_debug_config
from lmnr.sdk.debug.pointer import build_pointer, emit_pointer
from lmnr.sdk.debug.replay_cache import ReplayCache
from lmnr.sdk.debug.source_trace import fetch_spine_metadata, fetch_spine_payloads
from lmnr.sdk.debug.spine import detect_spine, has_overlap
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class DebugRuntime:
    """Holds the immutable debug config plus the optional replay cache.

    Also tracks the run's root trace id so the pointer (§5) can be emitted once
    at shutdown, when the trace id is guaranteed to be known.
    """

    def __init__(
        self,
        config: DebugConfig,
        cache: ReplayCache | None,
        debugger_url: str | None,
    ):
        self._config = config
        self._cache = cache
        self._debugger_url = debugger_url
        self._trace_id: str | None = None
        self._emitted = False
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        # Captured at construction (SDK init) so the pointer's `started_at`
        # reflects when the run began, not when the pointer is emitted (shutdown).
        self._started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    @property
    def session_id(self) -> str:
        return self._config.session_id

    @property
    def replay_trace_id(self) -> str | None:
        return self._config.replay_trace_id

    @property
    def replay_configured(self) -> bool:
        """True when replay is configured (source trace + cache window)."""
        return self._config.replay_enabled

    def record_trace_id(self, trace_id: str) -> None:
        """Remember the root trace id of this run (first root span wins)."""
        with self._lock:
            if self._trace_id is None:
                self._trace_id = trace_id

    def get_cached(self, span_path: str) -> dict[str, Any] | None:
        """Return the cached payload to replay for the next occurrence, or None.

        Increments the per-path occurrence counter as a side effect, mirroring a
        live call consuming one slot of the cache window. The counter is owned by
        the runtime (not the cache) so it advances even while the cache is still
        loading (`_cache is None`); this keeps record-vs-replay alignment for
        parity with the TS SDK, whose cache fills in asynchronously.
        """
        with self._lock:
            occurrence = self._counters.get(span_path, 0)
            self._counters[span_path] = occurrence + 1
        if self._cache is None:
            return None
        return self._cache.get_cached(span_path, occurrence)

    def emit_pointer(self) -> None:
        """Emit the run pointer once (console line + best-effort file)."""
        with self._lock:
            if self._emitted:
                return
            self._emitted = True
        pointer = build_pointer(
            trace_id=self._trace_id or "",
            session_id=self._config.session_id,
            replay_trace_id=self._config.replay_trace_id,
            cache_until=self._config.cache_until,
            debugger_url=self._debugger_url,
            started_at=self._started_at,
        )
        emit_pointer(pointer)


_runtime: DebugRuntime | None = None
_initialized = False


def get_runtime() -> DebugRuntime | None:
    """Return the process-wide debug runtime, or None when debug mode is off."""
    return _runtime


def reset_debug_runtime() -> None:
    """Reset module state so a later `init_debug_runtime` re-reads `LMNR_DEBUG*`.

    Called by `Laminar.shutdown()` (which supports a subsequent `initialize()`)
    and by tests. Without this, the one-shot `_initialized` flag would pin the
    first run's runtime — stale replay cache, session metadata, and a spent
    pointer — across a shutdown/initialize cycle. Mirrors the TS
    `resetDebugRuntime`.
    """
    global _runtime, _initialized
    _runtime = None
    _initialized = False


def init_debug_runtime(
    client: Any,
    debugger_url: str | None = None,
) -> DebugRuntime | None:
    """Build the debug runtime once. Idempotent; safe to call from initialize().

    Never raises: a failure to fetch / build the cache degrades the run to
    debug-no-replay rather than crashing the user's program.

    Args:
        client: a `LaminarClient` used to fetch the source trace's spans.
        debugger_url: the web UI base URL recorded (informational) in the run
            pointer's `debugger_url` field.
    """
    global _runtime, _initialized
    if _initialized:
        return _runtime
    _initialized = True

    config = build_debug_config()
    if config is None:
        return None

    cache = None
    if config.replay_enabled:
        try:
            cache = _build_cache(client, config)
        except Exception as exc:  # degrade to debug-no-replay, never crash
            logger.warning(
                "Failed to build replay cache for %s; running live: %s",
                config.replay_trace_id,
                exc,
            )

    _runtime = DebugRuntime(config, cache, debugger_url)
    return _runtime


def _build_cache(client: Any, config: DebugConfig) -> ReplayCache | None:
    """Fetch the source trace, detect the spine, and build the replay cache.

    Returns None (degrade to debug-no-replay) when there is no spine or when the
    spine's first N calls overlap in time (§F overlap guard, fail-loud-but-live).
    """
    trace_id = config.replay_trace_id
    records = fetch_spine_metadata(client, trace_id)

    result = detect_spine(records)
    if result.spine_path is None:
        return None

    if has_overlap(result.spine_calls, config.cache_until):
        logger.warning(
            "Spine '%s' of source trace %s has LLM calls that overlap in time; "
            "v1 cannot safely replay a non-sequential spine — running live.",
            result.spine_path,
            trace_id,
        )
        return None

    payloads = fetch_spine_payloads(client, trace_id, result.spine_path)
    return ReplayCache(
        spine_path=result.spine_path,
        cache_until=config.cache_until,
        payloads=payloads,
    )

"""In-process debug runtime (v2).

A debug run is a normal process execution started with `LMNR_DEBUG*` env vars.
At SDK init the runtime is built once from the environment (§4):

1. Parse `DebugConfig` from the environment.
2. Register the debug session and retain the Laminar clients so each live LLM
   call can look its input hash up in the **server-side** cache (shared spec
   §5–§7) — v2 has no in-process cache to build, no source-trace fetch, and no
   spine detection.
3. The run's pointer (§5) is emitted once at process shutdown, after the root
   trace id of this run is known.

The provider LLM wrappers consult `get_runtime()` to decide replay-vs-live and
use `runtime.client` / `runtime.async_client` to call the cache endpoint. When
debug mode is off, `get_runtime()` returns None and everything is inert.
"""

import datetime
import threading
from typing import Any

from lmnr.sdk.debug.config import (
    DebugConfig,
    build_debug_config,
    build_debug_config_from_context,
)
from lmnr.sdk.debug.pointer import build_pointer, emit_pointer
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class DebugRuntime:
    """Holds the immutable debug config plus the retained Laminar clients.

    The clients live for the whole run (not closed at init) because every LLM
    call now hits the server-side cache endpoint through them; `Laminar.shutdown`
    closes them. Also tracks the run's root trace id so the pointer (§5) can be
    emitted once at shutdown, when the trace id is guaranteed to be known.
    """

    def __init__(
        self,
        config: DebugConfig,
        client: Any,
        async_client: Any,
        debugger_url: str | None,
    ):
        self._config = config
        self._client = client
        self._async_client = async_client
        self._debugger_url = debugger_url
        self._trace_id: str | None = None
        self._project_id: str | None = None
        self._emitted = False
        self._lock = threading.Lock()
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
    def cache_until_span_id(self) -> str | None:
        return self._config.cache_until_span_id

    @property
    def local_origin(self) -> bool:
        """True when this process originated the run (config from local env).

        False when the runtime was armed from a propagated `DebugContext`: a
        downstream run reuses the upstream session and may consult the cache,
        but must not open a browser or emit the run pointer.
        """
        return self._config.local_origin

    @property
    def should_open_browser(self) -> bool:
        """True when this run should open the debugger URL in a browser once.

        Only a local-origin run that minted a fresh session id qualifies: a
        reused session id (continuation/replay) or a context-armed downstream
        run must not reopen the browser.
        """
        return self._config.local_origin and self._config.session_minted

    @property
    def client(self) -> Any:
        """The retained synchronous `LaminarClient` for cache lookups."""
        return self._client

    @property
    def async_client(self) -> Any:
        """The retained asynchronous `AsyncLaminarClient` for cache lookups."""
        return self._async_client

    @property
    def replay_configured(self) -> bool:
        """True when replay is configured (source trace + cache_until needle).

        v2 builds nothing synchronously and has no async cache-load window, so
        this collapses to the config's `replay_enabled` — there is no
        "configured but not yet ready" state to model.
        """
        return self._config.replay_enabled

    def record_trace_id(self, trace_id: str) -> None:
        """Remember the root trace id of this run (first root span wins)."""
        with self._lock:
            if self._trace_id is None:
                self._trace_id = trace_id

    def record_project_id(self, project_id: str) -> None:
        """Remember the backend-resolved project id (first wins).

        Resolved by the session-register call AFTER construction, so the
        runtime starts with only the base `debugger_url`; this lets
        `debugger_session_url` upgrade to the full per-session URL once known.
        """
        with self._lock:
            if self._project_id is None:
                self._project_id = project_id

    def debugger_session_url(self) -> str | None:
        """The human-facing debugger URL for this run, or None.

        Single source of truth for the URL printed to the console AND stored in
        the run pointer's `debugger_url` field. When the project id is known
        (register succeeded) it is the full
        `<base>/project/<project_id>/debugger-sessions/<session_id>`; otherwise
        it falls back to the base `debugger_url` (None when even that is unset).
        """
        if self._debugger_url is None:
            return None
        if self._project_id is None:
            return self._debugger_url
        return (
            f"{self._debugger_url}/project/{self._project_id}"
            f"/debugger-sessions/{self._config.session_id}"
        )

    def emit_pointer(self) -> None:
        """Emit the run pointer once (console line + best-effort file).

        No-op on a downstream run (`local_origin=False`): a runtime armed from a
        propagated `DebugContext` joins the upstream replay session and must NOT
        write a run pointer — the origin owns it. Gated here (not just at the
        call sites) so `shutdown()` and any atexit hook stay safe.
        """
        if not self._config.local_origin:
            return
        with self._lock:
            if self._emitted:
                return
            self._emitted = True
        pointer = build_pointer(
            trace_id=self._trace_id or "",
            session_id=self._config.session_id,
            replay_trace_id=self._config.replay_trace_id,
            # v2 persists the span-id needle as-is (no resolution step), so a
            # later LMNR_DEBUG_FROM_LAST_RUN replay re-sends the same needle.
            cache_until=self._config.cache_until_span_id,
            debugger_url=self.debugger_session_url(),
            started_at=self._started_at,
        )
        emit_pointer(pointer)


_runtime: DebugRuntime | None = None
_initialized = False
# Serializes the check-and-set of the one-shot init globals. Span creation runs
# `init_debug_runtime_from_context` from arbitrary worker threads, so the
# `_initialized` read and the `_runtime` write must be atomic or two threads
# both build a runtime, overwrite `_runtime`, and leak the loser's clients.
_init_lock = threading.Lock()


def get_runtime() -> DebugRuntime | None:
    """Return the process-wide debug runtime, or None when debug mode is off."""
    return _runtime


def reset_debug_runtime() -> None:
    """Reset module state so a later `init_debug_runtime` re-reads `LMNR_DEBUG*`.

    Called by `Laminar.shutdown()` (which supports a subsequent `initialize()`)
    and by tests. Without this, the one-shot `_initialized` flag would pin the
    first run's runtime — stale session metadata and a spent pointer — across a
    shutdown/initialize cycle. Mirrors the TS `resetDebugRuntime`.
    """
    global _runtime, _initialized
    _runtime = None
    _initialized = False


def init_debug_runtime(
    client: Any,
    async_client: Any,
    debugger_url: str | None = None,
) -> DebugRuntime | None:
    """Build the debug runtime once. Idempotent; safe to call from initialize().

    Never raises. v2 builds no cache — it only parses config and constructs the
    (cache-less) `DebugRuntime`, retaining the clients for per-call lookups.
    Session registration happens in the caller (`Laminar._init_debug_runtime`).

    Args:
        client: a `LaminarClient` retained for synchronous cache lookups.
        async_client: an `AsyncLaminarClient` retained for async cache lookups.
        debugger_url: the web UI base URL recorded in the run pointer's
            `debugger_url` field.
    """
    global _runtime, _initialized
    if _initialized:
        return _runtime

    config = build_debug_config()
    if config is None:
        # Debug mode is off: nothing was built, so do NOT latch the one-shot
        # flag. Otherwise a later init (e.g. after the env flips LMNR_DEBUG on)
        # would short-circuit to None until reset_debug_runtime(). The off path
        # only reads env vars, so re-running it on a repeat call is cheap.
        return None
    _initialized = True

    _runtime = DebugRuntime(config, client, async_client, debugger_url)
    return _runtime


def init_debug_runtime_from_context(
    debug: Any,
    client: Any,
    async_client: Any,
    debugger_url: str | None = None,
) -> DebugRuntime | None:
    """Arm the debug runtime from a propagated `DebugContext` (process-global).

    Called deep in span creation when a parent `LaminarSpanContext` carrying a
    debug block first parses successfully — so a downstream service joins the
    upstream run regardless of how the span originated (auto-instrumentation,
    manual observe, or an external library). First-wins and idempotent: once a
    runtime exists (from env at init OR an earlier context), this is a no-op, so
    env-var config always takes precedence over a later context, and the first
    qualifying context wins over subsequent ones.

    Returns None when the block is absent / unarmed (`build_debug_config_from_context`
    yields None) WITHOUT latching the one-shot flag, so a later, valid context
    can still arm the runtime. Never raises.
    """
    global _runtime, _initialized
    if _initialized:
        return _runtime

    try:
        config = build_debug_config_from_context(debug)
    except Exception as exc:
        logger.debug("Failed to build debug config from context: %s", exc)
        return None
    if config is None:
        return None

    # Span creation calls this from arbitrary worker threads. Re-check
    # `_initialized` under the lock so only ONE thread builds + publishes the
    # runtime; losers return the winner's instance. Without this, two threads
    # both pass the unlocked check above, both construct a DebugRuntime, and the
    # loser's clients are returned to a caller whose `runtime.client is client`
    # cleanup guard then leaks them.
    with _init_lock:
        if _initialized:
            return _runtime
        _runtime = DebugRuntime(config, client, async_client, debugger_url)
        _initialized = True
        return _runtime

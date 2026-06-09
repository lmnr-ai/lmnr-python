import pytest
from unittest.mock import patch

from lmnr.sdk.debug import (
    DebugRuntime,
    get_runtime,
    init_debug_runtime,
    reset_debug_runtime,
)
from lmnr.sdk.debug.config import DebugConfig
from lmnr.sdk.debug.outcome import CacheOutcome


@pytest.fixture
def no_browser(monkeypatch):
    """Prevent _init_debug_runtime from opening a browser tab during tests."""
    monkeypatch.setenv("LMNR_DEBUG_SESSION_ID", "test-session")


def _reset_runtime():
    reset_debug_runtime()


class _NoopExporter:
    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult

        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


def _config(**kwargs) -> DebugConfig:
    base = {
        "session_id": "s",
        "replay_trace_id": "r",
        "cache_until_span_id": "abcdef",
    }
    base.update(kwargs)
    return DebugConfig(**base)


def _runtime(*, debugger_url=None, client=None, async_client=None, **cfg) -> DebugRuntime:
    """Construct a DebugRuntime with the v2 two-client signature.

    Most tests don't exercise the cache clients (no live LLM call), so they
    default to None; the cache-lookup paths are covered in test_debug_replay.py.
    """
    return DebugRuntime(_config(**cfg), client, async_client, debugger_url)


def test_replay_configured_reflects_config():
    # v2 has no synchronous cache build, so replay_configured collapses to the
    # config's replay_enabled (source trace + cache_until span-id needle).
    assert _runtime().replay_configured is True
    assert _runtime(replay_trace_id=None).replay_configured is False
    assert _runtime(cache_until_span_id=None).replay_configured is False


def test_runtime_retains_both_clients():
    sync_client = object()
    async_client = object()
    runtime = _runtime(client=sync_client, async_client=async_client)
    assert runtime.client is sync_client
    assert runtime.async_client is async_client


def test_record_trace_id_first_wins():
    runtime = _runtime()
    runtime.record_trace_id("trace-a")
    runtime.record_trace_id("trace-b")
    assert runtime._trace_id == "trace-a"


def test_record_project_id_first_wins():
    runtime = _runtime(debugger_url="https://x")
    runtime.record_project_id("proj-a")
    runtime.record_project_id("proj-b")
    assert runtime._project_id == "proj-a"


def test_debugger_session_url_falls_back_to_base_without_project_id():
    # Before register resolves a project id, the URL is just the base.
    runtime = _runtime(debugger_url="https://app.x")
    assert runtime.debugger_session_url() == "https://app.x"


def test_debugger_session_url_is_none_without_base():
    runtime = _runtime(debugger_url=None)
    assert runtime.debugger_session_url() is None


def test_debugger_session_url_full_with_project_id():
    runtime = _runtime(session_id="sess-1", debugger_url="https://app.x")
    runtime.record_project_id("proj-1")
    assert runtime.debugger_session_url() == (
        "https://app.x/project/proj-1/debugger-sessions/sess-1"
    )


def test_record_debug_trace_id_from_env_populates_pointer(monkeypatch):
    # A run attached via LMNR_SPAN_CONTEXT never opens a root span, so the
    # pointer would emit an empty trace_id unless the inherited trace id is
    # recorded at env-attach time.
    import uuid as _uuid

    from opentelemetry import trace as _trace

    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    runtime = _runtime()
    monkeypatch.setattr("lmnr.sdk.debug._runtime", runtime)

    trace_id = _uuid.UUID("01234567-89ab-cdef-0123-456789abcdef")
    span_context = _trace.SpanContext(
        trace_id=trace_id.int,
        span_id=0x0123456789ABCDEF,
        is_remote=True,
    )
    Laminar._record_debug_trace_id_from_env(span_context)

    assert runtime._trace_id == str(trace_id)
    _reset_runtime()


def test_env_context_arms_debug_runtime_from_block(monkeypatch):
    # A debug block carried by LMNR_SPAN_CONTEXT must arm the debug runtime: an
    # LMNR_SPAN_CONTEXT-attached run parents off the pushed context with
    # parent_span_context=None, so the span-creation funnels never see the block
    # and only _initialize_context_from_env can activate replay downstream.
    import uuid as _uuid

    from lmnr.sdk.laminar import Laminar
    from lmnr.sdk.types import DebugContext, LaminarSpanContext

    _reset_runtime()

    session_id = str(_uuid.uuid4())
    ctx = LaminarSpanContext(
        trace_id=_uuid.UUID("01234567-89ab-cdef-0123-456789abcdef"),
        span_id=_uuid.UUID("00000000-0000-0000-0123-456789abcdef"),
        debug=DebugContext(enabled=True, session_id=session_id),
    )

    armed_with = {}

    def _fake_arm(debug):
        armed_with["debug"] = debug

    monkeypatch.setattr(Laminar, "_arm_debug_runtime_from_context", _fake_arm)
    monkeypatch.setenv("LMNR_SPAN_CONTEXT", str(ctx))

    Laminar._initialize_context_from_env()

    assert "debug" in armed_with
    assert armed_with["debug"] is not None
    assert armed_with["debug"].enabled is True
    assert armed_with["debug"].session_id == session_id
    _reset_runtime()


def test_processor_records_trace_id_when_tracing_disabled(monkeypatch):
    # Even with LMNR_DISABLE_TRACING=true the processor must record the root
    # trace id, otherwise the shutdown pointer emits an empty trace_id while
    # replay (gated only on get_runtime() is not None) may still be active.
    import uuid as _uuid

    from opentelemetry import trace as _trace

    from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor

    _reset_runtime()
    runtime = _runtime()
    monkeypatch.setattr("lmnr.sdk.debug._runtime", runtime)
    monkeypatch.setenv("LMNR_DISABLE_TRACING", "true")

    trace_id = _uuid.UUID("01234567-89ab-cdef-0123-456789abcdef")

    class _FakeSpan:
        def __init__(self):
            self.parent = None
            self.name = "root"
            self.attributes = {}
            self._ctx = _trace.SpanContext(
                trace_id=trace_id.int,
                span_id=0x0123456789ABCDEF,
                is_remote=False,
            )

        def get_span_context(self):
            return self._ctx

        def set_attribute(self, key, value):
            self.attributes[key] = value

    processor = LaminarSpanProcessor(exporter=_NoopExporter(), disable_batch=True)
    processor.on_start(_FakeSpan())

    assert runtime._trace_id == str(trace_id)
    _reset_runtime()


def test_processor_keeps_real_span_path_for_replay_when_disabled(monkeypatch):
    # With replay active, LMNR_DISABLE_TRACING=true must NOT mask span names to
    # "_" in lmnr.span.path: the replay wrapper reads that in-process path to
    # match the cache (keyed on the source trace's real dotted paths). Masking
    # would never match, so replay would silently run live.
    import uuid as _uuid

    from opentelemetry import trace as _trace

    from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor

    _reset_runtime()
    # replay_trace_id + cache_until span-id => replay_configured is True.
    runtime = _runtime()
    monkeypatch.setattr("lmnr.sdk.debug._runtime", runtime)
    monkeypatch.setenv("LMNR_DISABLE_TRACING", "true")

    class _FakeSpan:
        def __init__(self):
            self.parent = None
            self.name = "openai.chat"
            self.attributes = {}
            self._ctx = _trace.SpanContext(
                trace_id=_uuid.UUID(int=1).int,
                span_id=0x0123456789ABCDEF,
                is_remote=False,
            )

        def get_span_context(self):
            return self._ctx

        def set_attribute(self, key, value):
            self.attributes[key] = value

    span = _FakeSpan()
    processor = LaminarSpanProcessor(exporter=_NoopExporter(), disable_batch=True)
    processor.on_start(span)

    assert span.attributes["lmnr.span.path"] == ["openai.chat"]
    _reset_runtime()


def test_processor_masks_span_path_when_disabled_without_replay(monkeypatch):
    # No debug runtime: disabled tracing still masks span names to "_" (privacy).
    import uuid as _uuid

    from opentelemetry import trace as _trace

    from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor

    _reset_runtime()
    monkeypatch.setenv("LMNR_DISABLE_TRACING", "true")

    class _FakeSpan:
        def __init__(self):
            self.parent = None
            self.name = "openai.chat"
            self.attributes = {}
            self._ctx = _trace.SpanContext(
                trace_id=_uuid.UUID(int=2).int,
                span_id=0x0123456789ABCDEF,
                is_remote=False,
            )

        def get_span_context(self):
            return self._ctx

        def set_attribute(self, key, value):
            self.attributes[key] = value

    span = _FakeSpan()
    processor = LaminarSpanProcessor(exporter=_NoopExporter(), disable_batch=True)
    processor.on_start(span)

    assert span.attributes["lmnr.span.path"] == ["_"]
    _reset_runtime()


def test_record_debug_trace_id_from_env_noop_without_runtime(monkeypatch):
    from opentelemetry import trace as _trace

    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    span_context = _trace.SpanContext(
        trace_id=0x0123456789ABCDEF0123456789ABCDEF,
        span_id=0x0123456789ABCDEF,
        is_remote=True,
    )
    # No runtime registered -> silent no-op, never raises.
    Laminar._record_debug_trace_id_from_env(span_context)


def test_emit_pointer_uses_construction_time_started_at(tmp_path, monkeypatch, capsys):
    # started_at must reflect when the run began (runtime construction at SDK
    # init), not when the pointer is emitted (shutdown). Sleep between the two so
    # an emit-time timestamp would differ from the captured one.
    import json
    import time

    monkeypatch.chdir(tmp_path)
    runtime = _runtime()
    captured = runtime._started_at
    time.sleep(0.01)
    runtime.record_trace_id("trace-a")
    runtime.emit_pointer()

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("LMNR_DEBUG_RUN ")
    )
    payload = json.loads(line[len("LMNR_DEBUG_RUN ") :])
    assert payload["started_at"] == captured


def test_emit_pointer_persists_cache_until_span_id(tmp_path, monkeypatch, capsys):
    # v2 persists the raw span-id needle in the pointer's cache_until (no
    # resolution step), so a later LMNR_DEBUG_FROM_LAST_RUN re-sends the needle.
    import json

    monkeypatch.chdir(tmp_path)
    runtime = _runtime(cache_until_span_id="0123456789abcdef")
    runtime.record_trace_id("trace-a")
    runtime.emit_pointer()

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("LMNR_DEBUG_RUN ")
    )
    payload = json.loads(line[len("LMNR_DEBUG_RUN ") :])
    assert payload["cache_until"] == "0123456789abcdef"


def test_emit_pointer_only_once(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    runtime = _runtime()
    runtime.record_trace_id("trace-a")
    runtime.emit_pointer()
    runtime.emit_pointer()
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("LMNR_DEBUG_RUN ")
    ]
    assert len(lines) == 1


def test_emit_pointer_noop_for_downstream_run(tmp_path, monkeypatch, capsys):
    # A runtime armed from a propagated DebugContext (local_origin=False) joins
    # the upstream replay session and must NOT write a run pointer — the origin
    # owns it. Gated inside emit_pointer so shutdown()/atexit stay safe.
    monkeypatch.chdir(tmp_path)
    runtime = _runtime(local_origin=False)
    runtime.record_trace_id("trace-downstream")
    runtime.emit_pointer()

    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("LMNR_DEBUG_RUN ")
    ]
    assert lines == []
    assert not (tmp_path / ".lmnr" / "last-run.json").exists()


def test_emit_pointer_uses_full_debugger_url(tmp_path, monkeypatch, capsys):
    # The pointer's debugger_url must carry the SAME full per-session URL the
    # console prints, not just the base — built via the shared
    # debugger_session_url code path once the project id is recorded.
    import json

    monkeypatch.chdir(tmp_path)
    runtime = _runtime(session_id="sess-1", debugger_url="https://app.x")
    runtime.record_project_id("proj-1")
    runtime.record_trace_id("trace-a")
    runtime.emit_pointer()

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("LMNR_DEBUG_RUN ")
    )
    payload = json.loads(line[len("LMNR_DEBUG_RUN ") :])
    assert payload["debugger_url"] == (
        "https://app.x/project/proj-1/debugger-sessions/sess-1"
    )


def test_init_disabled_returns_none(monkeypatch):
    _reset_runtime()
    monkeypatch.delenv("LMNR_DEBUG", raising=False)
    assert init_debug_runtime(client=object(), async_client=object()) is None
    assert get_runtime() is None


def test_init_debug_runtime_skips_client_when_debug_off(monkeypatch):
    # When LMNR_DEBUG is off, Laminar._init_debug_runtime must NOT construct a
    # LaminarClient (and its httpx.Client) — that would leak unclosed on every
    # normal initialize().
    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    monkeypatch.delenv("LMNR_DEBUG", raising=False)

    constructed = []

    class _SpyClient:
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))

    monkeypatch.setattr(
        "lmnr.sdk.client.synchronous.sync_client.LaminarClient", _SpyClient
    )
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)

    assert constructed == []
    assert get_runtime() is None
    _reset_runtime()


def test_init_off_does_not_latch_flag(monkeypatch):
    # When debug is off, init must NOT spend the one-shot flag: a later init
    # after the env flips LMNR_DEBUG on must still build a runtime without an
    # intervening reset_debug_runtime().
    _reset_runtime()
    monkeypatch.delenv("LMNR_DEBUG", raising=False)
    assert init_debug_runtime(client=object(), async_client=object()) is None

    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)
    runtime = init_debug_runtime(client=object(), async_client=object())
    assert runtime is not None
    assert get_runtime() is runtime
    _reset_runtime()


def test_init_is_idempotent(monkeypatch):
    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)
    first = init_debug_runtime(client=object(), async_client=object())
    second = init_debug_runtime(client=object(), async_client=object())
    assert first is second is get_runtime()
    _reset_runtime()


def test_reset_allows_reinit_to_reread_env(monkeypatch):
    # A shutdown/initialize cycle must re-read LMNR_DEBUG*: reset clears the
    # one-shot flag so a previously-off run can turn debug on (and vice versa).
    _reset_runtime()
    monkeypatch.delenv("LMNR_DEBUG", raising=False)
    assert init_debug_runtime(client=object(), async_client=object()) is None

    reset_debug_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)
    runtime = init_debug_runtime(client=object(), async_client=object())
    assert runtime is not None
    assert get_runtime() is runtime
    _reset_runtime()


def test_init_builds_replay_runtime_from_env(monkeypatch):
    # A replay-configured run (source trace + cache_until span id) builds a
    # runtime that reports replay configured. v2 builds no cache synchronously.
    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_DEBUG_REPLAY_TRACE_ID", "trace-1")
    monkeypatch.setenv("LMNR_DEBUG_CACHE_UNTIL", "0123-456789abcdef")

    runtime = init_debug_runtime(
        client=object(), async_client=object(), debugger_url="https://www.lmnr.ai"
    )
    assert runtime is not None
    assert runtime.replay_configured is True
    assert runtime.replay_trace_id == "trace-1"
    assert runtime.cache_until_span_id == "0123456789abcdef"
    _reset_runtime()


class _SpyRolloutSessions:
    def __init__(self, raises=False, project_id=None):
        self.registered = []
        self._raises = raises
        self._project_id = project_id

    def register(self, session_id, name=None):
        self.registered.append((session_id, name))
        if self._raises:
            raise RuntimeError("backend down")
        return self._project_id

    def cache(self, **kwargs):
        return CacheOutcome(kind="live")


class _SpyDebugClient:
    def __init__(self, *args, raises=False, project_id=None, **kwargs):
        self.rollout_sessions = _SpyRolloutSessions(
            raises=raises, project_id=project_id
        )
        self.closed = False

    def close(self):
        self.closed = True


class _SpyAsyncDebugClient:
    def __init__(self, *args, **kwargs):
        self.rollout_sessions = _SpyRolloutSessions()
        self.closed = False

    async def close(self):
        self.closed = True


def _patch_clients(monkeypatch, sync_client, async_client=None):
    """Patch both retained client classes used by _init_debug_runtime."""
    monkeypatch.setattr(
        "lmnr.sdk.client.synchronous.sync_client.LaminarClient",
        lambda *a, **k: sync_client,
    )
    monkeypatch.setattr(
        "lmnr.sdk.client.asynchronous.async_client.AsyncLaminarClient",
        lambda *a, **k: async_client or _SpyAsyncDebugClient(),
    )


def test_init_registers_session_with_backend(monkeypatch):
    # A bare LMNR_DEBUG=true run must POST its SDK-minted session id to the
    # backend so the session shows up in the UI.
    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    spy = _SpyDebugClient()
    _patch_clients(monkeypatch, spy)
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)

    runtime = get_runtime()
    assert runtime is not None
    assert spy.rollout_sessions.registered == [(runtime.session_id, None)]
    # v2 RETAINS the cache clients for the run's lifetime (the provider wrappers
    # hit the cache endpoint on every live call); they are closed at shutdown,
    # NOT at init. So the sync client must still be open here.
    assert spy.closed is False
    assert runtime.client is spy
    _reset_runtime()


def test_init_logs_debugger_url_when_project_id_returned(no_browser, monkeypatch, caplog):
    # When the backend returns a project id, init must log the human-facing
    # debugger session URL at INFO, respecting LMNR_FRONTEND_URL.
    import logging

    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_FRONTEND_URL", "https://app.example.com")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    spy = _SpyDebugClient(project_id="proj-123")
    _patch_clients(monkeypatch, spy)
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)

    with caplog.at_level(logging.INFO, logger="lmnr.sdk.laminar"):
        Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)

    runtime = get_runtime()
    assert runtime is not None
    expected = (
        f"https://app.example.com/project/proj-123"
        f"/debugger-sessions/{runtime.session_id}"
    )
    assert any(expected in record.getMessage() for record in caplog.records)
    _reset_runtime()


def test_init_survives_registration_failure(monkeypatch):
    # Registration is best-effort: a backend error must never crash init.
    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    spy = _SpyDebugClient(raises=True)
    _patch_clients(monkeypatch, spy)
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)

    runtime = get_runtime()
    assert runtime is not None  # init still completed
    assert len(spy.rollout_sessions.registered) == 1
    _reset_runtime()


def test_init_does_not_build_debug_runtime_when_tracing_fails(monkeypatch):
    # If TracerManager.init() raises, initialize() must abort BEFORE any debug
    # side effects: no backend session registration and no debug runtime left
    # live on a process whose tracing never came up.
    import os

    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    spy = _SpyDebugClient()
    _patch_clients(monkeypatch, spy)

    def _boom(*args, **kwargs):
        raise RuntimeError("tracer down")

    monkeypatch.setattr("lmnr.opentelemetry_lib.TracerManager.init", _boom)
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)

    with patch.dict(os.environ, {"LMNR_PROJECT_API_KEY": "k"}):
        try:
            Laminar.initialize(project_api_key="k")
        except RuntimeError:
            pass

    # Tracer init failed before _init_debug_runtime ran: no runtime, no session.
    assert get_runtime() is None
    assert spy.rollout_sessions.registered == []
    _reset_runtime()
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)


def test_exit_hook_does_not_accumulate_across_cycles(tmp_path, monkeypatch):
    # atexit holds a strong ref to whatever it registers, so each debug-mode
    # init must unregister the previous pointer hook on shutdown — otherwise an
    # init/shutdown loop pins every retired DebugRuntime alive and leaks one
    # atexit handler per cycle.
    import atexit

    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    registered: list = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: registered.append(fn))
    monkeypatch.setattr(
        atexit,
        "unregister",
        lambda fn: registered.remove(fn) if fn in registered else None,
    )

    _patch_clients(monkeypatch, _SpyDebugClient())
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr("lmnr.opentelemetry_lib.TracerManager.shutdown", lambda: None)

    for _ in range(12):
        Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)
        monkeypatch.setattr(Laminar, "_Laminar__initialized", True, raising=False)
        Laminar.shutdown()

    assert registered == []
    _reset_runtime()
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)


def test_shutdown_closes_retained_clients(tmp_path, monkeypatch):
    # v2 keeps both cache clients open for the run; shutdown must close both so
    # their httpx connection pools aren't leaked across init/shutdown cycles.
    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    sync_spy = _SpyDebugClient()
    async_spy = _SpyAsyncDebugClient()
    _patch_clients(monkeypatch, sync_spy, async_spy)
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr("lmnr.opentelemetry_lib.TracerManager.shutdown", lambda: None)

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)
    monkeypatch.setattr(Laminar, "_Laminar__initialized", True, raising=False)

    Laminar.shutdown()

    assert sync_spy.closed is True
    assert async_spy.closed is True
    _reset_runtime()
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)


def test_shutdown_resets_run_live_latch(tmp_path, monkeypatch):
    # A MISS latches the process-wide run-live flag; shutdown must clear it so a
    # fresh debug run in the same process starts from a clean cache state.
    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    _patch_clients(monkeypatch, _SpyDebugClient())
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr("lmnr.opentelemetry_lib.TracerManager.shutdown", lambda: None)

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)
    monkeypatch.setattr(Laminar, "_Laminar__initialized", True, raising=False)
    Laminar.set_debug_run_live(True)
    assert Laminar.is_debug_run_live() is True

    Laminar.shutdown()

    assert Laminar.is_debug_run_live() is False
    _reset_runtime()
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)


def test_shutdown_completes_cleanup_when_emit_pointer_raises(tmp_path, monkeypatch):
    # emit_pointer prints to stdout, which can raise OSError/BrokenPipeError
    # (closed stdout in daemons/containers, notebook kernel restarts). That must
    # never abort shutdown's cleanup: TracerManager.shutdown(), the reset, and
    # the __initialized flip must still run.
    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    _patch_clients(monkeypatch, _SpyDebugClient())
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)

    shutdown_calls: list = []
    monkeypatch.setattr(
        "lmnr.opentelemetry_lib.TracerManager.shutdown",
        lambda: shutdown_calls.append(True),
    )

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)
    monkeypatch.setattr(Laminar, "_Laminar__initialized", True, raising=False)

    runtime = get_runtime()
    assert runtime is not None
    # Simulate a broken stdout: emit_pointer raises.
    monkeypatch.setattr(runtime, "emit_pointer", _raise_broken_pipe)

    # Must not propagate the BrokenPipeError out of shutdown().
    Laminar.shutdown()

    # The cleanup after emit_pointer still ran.
    assert shutdown_calls == [True]
    assert Laminar.is_initialized() is False
    assert get_runtime() is None
    _reset_runtime()
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)


def _raise_broken_pipe():
    raise BrokenPipeError("stdout closed")


def test_arm_from_context_closes_clients_when_losing_race(monkeypatch):
    # _arm_debug_runtime_from_context allocates fresh sync/async clients BEFORE
    # consulting init_debug_runtime_from_context, which is first-wins. Under a
    # concurrent arm, both callers pass the get_runtime() fast path, both
    # allocate, and one loses inside init_*_from_context — getting back a runtime
    # that retains the WINNER's clients. The loser's freshly-allocated clients
    # are orphaned and must be closed, or their httpx pools leak. We simulate the
    # lost race by patching init_*_from_context to return a winner runtime built
    # from different clients (get_runtime() stays None at the fast-path check).
    from lmnr.sdk.debug.config import DebugConfig
    from lmnr.sdk.laminar import Laminar
    from lmnr.sdk.types import DebugContext

    _reset_runtime()
    SESSION = "00000000-0000-0000-0000-0000000000aa"
    block = DebugContext(enabled=True, session_id=SESSION)

    # The winner runtime (built by the thread that won the race) retains its own
    # clients; _arm_debug_runtime_from_context must NOT close these.
    winner_sync = _SpyDebugClient()
    winner_async = _SpyAsyncDebugClient()
    winner = DebugRuntime(
        DebugConfig(session_id=SESSION, replay_trace_id=None, local_origin=False),
        winner_sync,
        winner_async,
        None,
    )

    def _fake_init_from_context(dbg, client, async_client, debugger_url=None):
        # Mimic the lost-race outcome: return the winner runtime, ignoring the
        # clients this caller passed (init is first-wins and already armed).
        return winner

    monkeypatch.setattr(
        "lmnr.sdk.debug.init_debug_runtime_from_context", _fake_init_from_context
    )

    # _arm_debug_runtime_from_context allocates these (the loser's clients).
    loser_sync = _SpyDebugClient()
    loser_async = _SpyAsyncDebugClient()
    _patch_clients(monkeypatch, loser_sync, loser_async)
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr(Laminar, "_Laminar__base_url_for_debug", "http://localhost",
                        raising=False)
    monkeypatch.setattr(Laminar, "_Laminar__http_port_for_debug", 8000, raising=False)

    Laminar._arm_debug_runtime_from_context(block)

    # The loser's freshly-allocated clients are closed (not leaked); the winner's
    # clients stay open (the winner owns the run).
    assert loser_sync.closed is True
    assert loser_async.closed is True
    assert winner_sync.closed is False
    assert winner_async.closed is False
    _reset_runtime()


def test_init_from_context_publishes_single_runtime_under_concurrency(monkeypatch):
    # Span creation calls init_debug_runtime_from_context from arbitrary worker
    # threads. The check-and-set of the one-shot globals must be atomic, or two
    # threads both pass the _initialized check, both build a DebugRuntime, and
    # publish different instances — leaving every loser's clients (each thread
    # passes its own pair) returned to a caller whose `runtime.client is client`
    # cleanup guard then fails to recognize the win and leaks them. With the lock,
    # exactly one runtime is built and published and every caller gets it back.
    import threading
    import time

    import lmnr.sdk.debug as debug_mod
    from lmnr.sdk.debug import init_debug_runtime_from_context
    from lmnr.sdk.types import DebugContext

    _reset_runtime()
    SESSION = "00000000-0000-0000-0000-0000000000aa"
    block = DebugContext(enabled=True, session_id=SESSION)

    n = 16
    start = threading.Barrier(n)

    # Count how many runtimes actually get constructed. Widen the race window by
    # sleeping in the config build (runs before the flag is set in the buggy
    # path) so every thread clears the unlocked _initialized check before any
    # thread publishes — the unsynchronized version then builds n runtimes.
    constructed = 0
    construct_lock = threading.Lock()
    real_runtime_cls = debug_mod.DebugRuntime

    class _CountingRuntime(real_runtime_cls):
        def __init__(self, *args, **kwargs):
            nonlocal constructed
            with construct_lock:
                constructed += 1
            super().__init__(*args, **kwargs)

    real_build = debug_mod.build_debug_config_from_context

    def _slow_build(dbg):
        time.sleep(0.02)
        return real_build(dbg)

    monkeypatch.setattr(debug_mod, "DebugRuntime", _CountingRuntime)
    monkeypatch.setattr(debug_mod, "build_debug_config_from_context", _slow_build)

    returned: list[DebugRuntime] = []
    returned_lock = threading.Lock()

    def _arm():
        # Each thread brings its own client pair, like _arm_debug_runtime_from_context.
        client, async_client = object(), object()
        start.wait()
        runtime = init_debug_runtime_from_context(block, client, async_client)
        with returned_lock:
            returned.append(runtime)

    threads = [threading.Thread(target=_arm) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    published = get_runtime()
    assert published is not None
    # Exactly one runtime was built (no losers whose clients would leak), and
    # every caller got back that single published instance.
    assert constructed == 1
    assert len(returned) == n
    assert all(r is published for r in returned)
    _reset_runtime()

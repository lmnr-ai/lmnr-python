from unittest.mock import patch

from lmnr.sdk.debug import (
    DebugRuntime,
    get_runtime,
    init_debug_runtime,
    reset_debug_runtime,
)
from lmnr.sdk.debug.config import DebugConfig
from lmnr.sdk.debug.replay_cache import ReplayCache


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
    base = {"session_id": "s", "replay_trace_id": "r", "cache_until": 2}
    base.update(kwargs)
    return DebugConfig(**base)


class _FakeSql:
    def __init__(self, metadata_rows, payload_rows):
        self._metadata_rows = metadata_rows
        self._payload_rows = payload_rows
        self.calls = []

    def query(self, query, parameters=None):
        self.calls.append((query, parameters))
        # Phase 1 selects span_type; phase 2 filters on path.
        if "span_type, start_time" in query:
            rows, self._metadata_rows = self._metadata_rows, []
            return rows
        rows, self._payload_rows = self._payload_rows, []
        return rows


class _FakeClient:
    def __init__(self, metadata_rows, payload_rows):
        self.sql = _FakeSql(metadata_rows, payload_rows)
        self.closed = False

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_runtime_get_cached_advances_occurrence():
    cache = ReplayCache("loop.llm", cache_until=2, payloads=[{"o": 0}, {"o": 1}])
    runtime = DebugRuntime(_config(), cache, debugger_url=None)

    assert runtime.get_cached("loop.llm") == {"o": 0}
    assert runtime.get_cached("loop.llm") == {"o": 1}
    # Past the window -> live.
    assert runtime.get_cached("loop.llm") is None
    # Non-spine path -> live, even on first occurrence.
    assert runtime.get_cached("other") is None


def test_runtime_get_cached_no_cache_returns_none():
    runtime = DebugRuntime(_config(), cache=None, debugger_url=None)
    assert runtime.get_cached("loop.llm") is None


def test_runtime_get_cached_advances_while_cache_loading():
    # Mirrors the TS async-fill window: calls before set_cache must still
    # advance the counter so post-load calls resume at the right slot.
    runtime = DebugRuntime(_config(), cache=None, debugger_url=None)
    assert runtime.get_cached("loop.llm") is None

    runtime._cache = ReplayCache(
        "loop.llm", cache_until=2, payloads=[{"o": 0}, {"o": 1}]
    )

    # First live call consumed occurrence 0; the cache resumes at 1.
    assert runtime.get_cached("loop.llm") == {"o": 1}
    assert runtime.get_cached("loop.llm") is None


def test_record_trace_id_first_wins():
    runtime = DebugRuntime(_config(), cache=None, debugger_url=None)
    runtime.record_trace_id("trace-a")
    runtime.record_trace_id("trace-b")
    assert runtime._trace_id == "trace-a"


def test_record_project_id_first_wins():
    runtime = DebugRuntime(_config(), cache=None, debugger_url="https://x")
    runtime.record_project_id("proj-a")
    runtime.record_project_id("proj-b")
    assert runtime._project_id == "proj-a"


def test_debugger_session_url_falls_back_to_base_without_project_id():
    # Before register resolves a project id, the URL is just the base.
    runtime = DebugRuntime(_config(), cache=None, debugger_url="https://app.x")
    assert runtime.debugger_session_url() == "https://app.x"


def test_debugger_session_url_is_none_without_base():
    runtime = DebugRuntime(_config(), cache=None, debugger_url=None)
    assert runtime.debugger_session_url() is None


def test_debugger_session_url_full_with_project_id():
    runtime = DebugRuntime(
        _config(session_id="sess-1"), cache=None, debugger_url="https://app.x"
    )
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
    runtime = DebugRuntime(_config(), cache=None, debugger_url=None)
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


def test_processor_records_trace_id_when_tracing_disabled(monkeypatch):
    # Even with LMNR_DISABLE_TRACING=true the processor must record the root
    # trace id, otherwise the shutdown pointer emits an empty trace_id while
    # replay (gated only on get_runtime() is not None) may still be active.
    import uuid as _uuid

    from opentelemetry import trace as _trace

    from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor

    _reset_runtime()
    runtime = DebugRuntime(_config(), cache=None, debugger_url=None)
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
    # replay_trace_id + cache_until>0 => replay_enabled() is True.
    runtime = DebugRuntime(_config(), cache=None, debugger_url=None)
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
    runtime = DebugRuntime(_config(), cache=None, debugger_url=None)
    captured = runtime._started_at
    time.sleep(0.01)
    runtime.record_trace_id("trace-a")
    runtime.emit_pointer()

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("LMNR_DEBUG_RUN ")
    )
    payload = json.loads(line[len("LMNR_DEBUG_RUN "):])
    assert payload["started_at"] == captured


def test_emit_pointer_only_once(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    runtime = DebugRuntime(_config(), cache=None, debugger_url=None)
    runtime.record_trace_id("trace-a")
    runtime.emit_pointer()
    runtime.emit_pointer()
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("LMNR_DEBUG_RUN ")
    ]
    assert len(lines) == 1


def test_emit_pointer_uses_full_debugger_url(tmp_path, monkeypatch, capsys):
    # The pointer's debugger_url must carry the SAME full per-session URL the
    # console prints, not just the base — built via the shared
    # debugger_session_url code path once the project id is recorded.
    import json

    monkeypatch.chdir(tmp_path)
    runtime = DebugRuntime(
        _config(session_id="sess-1"), cache=None, debugger_url="https://app.x"
    )
    runtime.record_project_id("proj-1")
    runtime.record_trace_id("trace-a")
    runtime.emit_pointer()

    line = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("LMNR_DEBUG_RUN ")
    )
    payload = json.loads(line[len("LMNR_DEBUG_RUN "):])
    assert payload["debugger_url"] == (
        "https://app.x/project/proj-1/debugger-sessions/sess-1"
    )


def test_init_disabled_returns_none(monkeypatch):
    _reset_runtime()
    monkeypatch.delenv("LMNR_DEBUG", raising=False)
    assert init_debug_runtime(client=object()) is None
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
    assert init_debug_runtime(client=object()) is None

    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)
    runtime = init_debug_runtime(client=object())
    assert runtime is not None
    assert get_runtime() is runtime
    _reset_runtime()


def test_init_is_idempotent(monkeypatch):
    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)
    first = init_debug_runtime(client=object())
    second = init_debug_runtime(client=object())
    assert first is second is get_runtime()
    _reset_runtime()


def test_reset_allows_reinit_to_reread_env(monkeypatch):
    # A shutdown/initialize cycle must re-read LMNR_DEBUG*: reset clears the
    # one-shot flag so a previously-off run can turn debug on (and vice versa).
    _reset_runtime()
    monkeypatch.delenv("LMNR_DEBUG", raising=False)
    assert init_debug_runtime(client=object()) is None

    reset_debug_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)
    runtime = init_debug_runtime(client=object())
    assert runtime is not None
    assert get_runtime() is runtime
    _reset_runtime()


def test_init_builds_cache_for_looping_spine(monkeypatch):
    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_DEBUG_REPLAY_TRACE_ID", "trace-1")
    monkeypatch.setenv("LMNR_DEBUG_CACHE_UNTIL", "2")

    metadata = [
        {"path": "agent.loop.llm", "span_type": "LLM", "start_time": 1.0, "end_time": 1.5},
        {"path": "agent.loop.llm", "span_type": "LLM", "start_time": 2.0, "end_time": 2.5},
    ]
    payloads = [
        {"name": "chat", "input": "a", "output": "0", "attributes": {}},
        {"name": "chat", "input": "b", "output": "1", "attributes": {}},
    ]
    client = _FakeClient(metadata, payloads)

    runtime = init_debug_runtime(client=client, debugger_url="https://www.lmnr.ai")
    assert runtime is not None
    assert runtime.get_cached("agent.loop.llm") == {
        "name": "chat",
        "input": "a",
        "output": "0",
        "attributes": {},
    }
    _reset_runtime()


def test_init_degrades_to_live_on_overlap(monkeypatch):
    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_DEBUG_REPLAY_TRACE_ID", "trace-1")
    monkeypatch.setenv("LMNR_DEBUG_CACHE_UNTIL", "2")

    metadata = [
        {"path": "loop.llm", "span_type": "LLM", "start_time": 0.0, "end_time": 1.5},
        {"path": "loop.llm", "span_type": "LLM", "start_time": 1.0, "end_time": 2.0},
    ]
    client = _FakeClient(metadata, [])

    runtime = init_debug_runtime(client=client)
    assert runtime is not None
    # Overlap guard -> no cache -> every call runs live.
    assert runtime.get_cached("loop.llm") is None
    # Degraded synchronously to no cache: replay must report inactive so the
    # provider wrappers don't install and advance per-path counters pointlessly.
    assert runtime.replay_configured is False
    _reset_runtime()


def test_init_degrades_to_live_on_fetch_failure(monkeypatch):
    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_DEBUG_REPLAY_TRACE_ID", "trace-1")
    monkeypatch.setenv("LMNR_DEBUG_CACHE_UNTIL", "2")

    class _BoomClient:
        class sql:
            @staticmethod
            def query(*args, **kwargs):
                raise RuntimeError("network down")

    runtime = init_debug_runtime(client=_BoomClient())
    assert runtime is not None  # never crashes
    assert runtime.get_cached("loop.llm") is None
    # Fetch failure degraded to no cache: replay must report inactive.
    assert runtime.replay_configured is False
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


class _SpyDebugClient:
    def __init__(self, *args, raises=False, project_id=None, **kwargs):
        self.sql = _FakeSql([], [])
        self.rollout_sessions = _SpyRolloutSessions(
            raises=raises, project_id=project_id
        )
        self.closed = False

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_init_registers_session_with_backend(monkeypatch):
    # A bare LMNR_DEBUG=true run must POST its SDK-minted session id to the
    # backend so the session shows up in the UI.
    from lmnr.sdk.laminar import Laminar

    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)

    spy = _SpyDebugClient()
    monkeypatch.setattr(
        "lmnr.sdk.client.synchronous.sync_client.LaminarClient",
        lambda *a, **k: spy,
    )
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)

    Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)

    runtime = get_runtime()
    assert runtime is not None
    assert spy.rollout_sessions.registered == [(runtime.session_id, None)]
    # The init-only client must be closed so its httpx connection pool isn't
    # leaked on every initialize() with LMNR_DEBUG set.
    assert spy.closed is True
    _reset_runtime()


def test_init_logs_debugger_url_when_project_id_returned(monkeypatch, caplog):
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
    monkeypatch.setattr(
        "lmnr.sdk.client.synchronous.sync_client.LaminarClient",
        lambda *a, **k: spy,
    )
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
    monkeypatch.setattr(
        "lmnr.sdk.client.synchronous.sync_client.LaminarClient",
        lambda *a, **k: spy,
    )
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
    monkeypatch.setattr(
        "lmnr.sdk.client.synchronous.sync_client.LaminarClient",
        lambda *a, **k: spy,
    )

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
    # init/shutdown loop pins every retired DebugRuntime (and its replay cache)
    # alive and leaks one atexit handler per cycle.
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

    monkeypatch.setattr(
        "lmnr.sdk.client.synchronous.sync_client.LaminarClient",
        lambda *a, **k: _SpyDebugClient(),
    )
    monkeypatch.setattr(Laminar, "_Laminar__project_api_key", "k", raising=False)
    monkeypatch.setattr("lmnr.opentelemetry_lib.TracerManager.shutdown", lambda: None)

    for _ in range(12):
        Laminar._init_debug_runtime(base_url="http://localhost", http_port=8000)
        monkeypatch.setattr(Laminar, "_Laminar__initialized", True, raising=False)
        Laminar.shutdown()

    assert registered == []
    _reset_runtime()
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False, raising=False)

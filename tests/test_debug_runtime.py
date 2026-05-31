import lmnr.sdk.debug as debug
from lmnr.sdk.debug import DebugRuntime, get_runtime, init_debug_runtime
from lmnr.sdk.debug.config import DebugConfig
from lmnr.sdk.debug.replay_cache import ReplayCache


def _reset_runtime():
    debug._runtime = None
    debug._initialized = False


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


def test_record_trace_id_first_wins():
    runtime = DebugRuntime(_config(), cache=None, debugger_url=None)
    runtime.record_trace_id("trace-a")
    runtime.record_trace_id("trace-b")
    assert runtime._trace_id == "trace-a"


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


def test_init_disabled_returns_none(monkeypatch):
    _reset_runtime()
    monkeypatch.delenv("LMNR_DEBUG", raising=False)
    assert init_debug_runtime(client=object()) is None
    assert get_runtime() is None


def test_init_is_idempotent(monkeypatch):
    _reset_runtime()
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.delenv("LMNR_DEBUG_REPLAY_TRACE_ID", raising=False)
    monkeypatch.delenv("LMNR_DEBUG_CACHE_UNTIL", raising=False)
    first = init_debug_runtime(client=object())
    second = init_debug_runtime(client=object())
    assert first is second is get_runtime()
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
    _reset_runtime()

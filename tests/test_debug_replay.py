import lmnr.sdk.debug as debug
from lmnr.sdk.debug import DebugRuntime
from lmnr.sdk.debug.config import DebugConfig
from lmnr.sdk.debug.replay_cache import ReplayCache
from lmnr.sdk.debug.replay import (
    cached_payload_for,
    mark_span_cached,
    replay_enabled,
    span_path_from_span,
)


class _FakeSpan:
    def __init__(self, attributes, recording=True):
        self.attributes = attributes
        self._recording = recording
        self.marked = {}

    def is_recording(self):
        return self._recording

    def set_attributes(self, attrs):
        self.marked.update(attrs)


def _reset_runtime():
    debug._runtime = None
    debug._initialized = False


def test_replay_enabled_reflects_runtime():
    _reset_runtime()
    assert replay_enabled() is False
    debug._runtime = DebugRuntime(
        DebugConfig("s", "r", 1), cache=None, debugger_url=None
    )
    assert replay_enabled() is True
    _reset_runtime()


def test_span_path_from_span_joins_with_dot():
    span = _FakeSpan({"lmnr.span.path": ["agent", "loop", "llm"]})
    assert span_path_from_span(span) == "agent.loop.llm"


def test_span_path_from_span_handles_missing():
    assert span_path_from_span(None) is None
    assert span_path_from_span(_FakeSpan({})) is None


def test_cached_payload_for_uses_runtime():
    _reset_runtime()
    cache = ReplayCache("agent.llm", cache_until=1, payloads=[{"o": 0}])
    debug._runtime = DebugRuntime(
        DebugConfig("s", "r", 1), cache=cache, debugger_url=None
    )
    assert cached_payload_for("agent.llm") == {"o": 0}
    # Counter advanced -> live now.
    assert cached_payload_for("agent.llm") is None
    _reset_runtime()


def test_cached_payload_for_no_runtime_or_path():
    _reset_runtime()
    assert cached_payload_for("agent.llm") is None
    debug._runtime = DebugRuntime(
        DebugConfig("s", "r", 1),
        cache=ReplayCache("agent.llm", 1, [{"o": 0}]),
        debugger_url=None,
    )
    assert cached_payload_for(None) is None
    _reset_runtime()


def test_mark_span_cached_sets_boundary_attributes():
    span = _FakeSpan({})
    mark_span_cached(span)
    assert span.marked == {
        "lmnr.span.type": "CACHED",
        "lmnr.span.original_type": "LLM",
    }


def test_mark_span_cached_noop_when_not_recording():
    span = _FakeSpan({}, recording=False)
    mark_span_cached(span)
    assert span.marked == {}


def test_mark_span_cached_handles_none():
    mark_span_cached(None)  # must not raise

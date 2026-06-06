import asyncio
import json
from pathlib import Path

import pytest

import lmnr.sdk.debug as debug
from lmnr.sdk.debug.hash import debug_input_hash
from lmnr.sdk.debug.outcome import CacheOutcome
from lmnr.sdk.debug.replay import (
    GEN_AI_INPUT_MESSAGES_ATTRIBUTE,
    acache_outcome_for,
    cache_outcome_for,
    input_messages_from_span,
    mark_span_cached,
    replay_enabled,
)
from lmnr.sdk.laminar import Laminar


class _FakeSpan:
    def __init__(self, attributes=None, recording=True):
        self.attributes = attributes or {}
        self._recording = recording
        self.marked = {}

    def is_recording(self):
        return self._recording

    def set_attributes(self, attrs):
        self.marked.update(attrs)


class _RecordingCache:
    """Sync rollout-session cache double; records calls and returns a scripted outcome."""

    def __init__(self, outcome: CacheOutcome):
        self.outcome = outcome
        self.calls = []

    def cache(self, *, session_id, replay_trace_id, cache_until, input_hash):
        self.calls.append(
            {
                "session_id": session_id,
                "replay_trace_id": replay_trace_id,
                "cache_until": cache_until,
                "input_hash": input_hash,
            }
        )
        return self.outcome


class _AsyncRecordingCache:
    def __init__(self, outcome: CacheOutcome):
        self.outcome = outcome
        self.calls = []

    async def cache(self, *, session_id, replay_trace_id, cache_until, input_hash):
        self.calls.append(
            {
                "session_id": session_id,
                "replay_trace_id": replay_trace_id,
                "cache_until": cache_until,
                "input_hash": input_hash,
            }
        )
        return self.outcome


class _FakeClient:
    def __init__(self, cache_resource):
        self.rollout_sessions = cache_resource


class _FakeRuntime:
    def __init__(
        self,
        *,
        replay_configured=True,
        sync_outcome=None,
        async_outcome=None,
        session_id="sess-1",
        replay_trace_id="trace-1",
        cache_until_span_id="abcdef",
    ):
        self.replay_configured = replay_configured
        self.session_id = session_id
        self.replay_trace_id = replay_trace_id
        self.cache_until_span_id = cache_until_span_id
        self._sync_cache = _RecordingCache(sync_outcome or CacheOutcome(kind="live"))
        self._async_cache = _AsyncRecordingCache(
            async_outcome or CacheOutcome(kind="live")
        )
        self.client = _FakeClient(self._sync_cache)
        self.async_client = _FakeClient(self._async_cache)


@pytest.fixture(autouse=True)
def _clean_replay_state():
    debug._runtime = None
    debug._initialized = False
    Laminar.set_debug_run_live(False)
    yield
    debug._runtime = None
    debug._initialized = False
    Laminar.set_debug_run_live(False)


def _span_with_messages(messages):
    return _FakeSpan({GEN_AI_INPUT_MESSAGES_ATTRIBUTE: json.dumps(messages)})


# --- replay_enabled -------------------------------------------------------


def test_replay_enabled_false_without_runtime():
    assert replay_enabled() is False


def test_replay_enabled_reflects_replay_configured():
    debug._runtime = _FakeRuntime(replay_configured=True)
    assert replay_enabled() is True


def test_replay_enabled_false_for_debug_no_replay_runtime():
    # A debug runtime with replay not configured (bare LMNR_DEBUG) must NOT
    # enable replay — otherwise the provider wrappers hit the cache endpoint.
    debug._runtime = _FakeRuntime(replay_configured=False)
    assert replay_enabled() is False


# --- input_messages_from_span --------------------------------------------


def test_input_messages_parses_json_string():
    span = _span_with_messages([{"role": "user", "content": "hi"}])
    assert input_messages_from_span(span) == [{"role": "user", "content": "hi"}]


def test_input_messages_accepts_already_decoded_list():
    span = _FakeSpan({GEN_AI_INPUT_MESSAGES_ATTRIBUTE: [{"role": "user"}]})
    assert input_messages_from_span(span) == [{"role": "user"}]


def test_input_messages_none_when_missing():
    assert input_messages_from_span(_FakeSpan({})) is None
    assert input_messages_from_span(None) is None


def test_input_messages_none_on_bad_json():
    span = _FakeSpan({GEN_AI_INPUT_MESSAGES_ATTRIBUTE: "{not json"})
    assert input_messages_from_span(span) is None


def test_input_messages_none_when_not_a_list():
    span = _FakeSpan({GEN_AI_INPUT_MESSAGES_ATTRIBUTE: json.dumps({"role": "user"})})
    assert input_messages_from_span(span) is None


# --- cache_outcome_for (sync) --------------------------------------------


def test_cache_outcome_none_without_runtime():
    assert cache_outcome_for(_span_with_messages([{"role": "user"}])) is None


def test_cache_outcome_none_when_replay_not_configured():
    debug._runtime = _FakeRuntime(replay_configured=False)
    assert cache_outcome_for(_span_with_messages([{"role": "user"}])) is None


def test_cache_outcome_none_when_no_input_messages():
    debug._runtime = _FakeRuntime()
    # No usable input on the span -> nothing to hash -> run live, no latch.
    assert cache_outcome_for(_FakeSpan({})) is None


def test_cache_outcome_hit_passes_hash_and_returns_cached():
    cached = {"attributes": {}, "output": "x", "start_time": 0, "end_time": 0}
    runtime = _FakeRuntime(sync_outcome=CacheOutcome(kind="hit", cached=cached))
    debug._runtime = runtime
    messages = [{"role": "user", "content": "hi"}]

    outcome = cache_outcome_for(_span_with_messages(messages))

    assert outcome.kind == "hit"
    assert outcome.cached is cached
    call = runtime._sync_cache.calls[0]
    assert call["session_id"] == "sess-1"
    assert call["replay_trace_id"] == "trace-1"
    assert call["cache_until"] == "abcdef"
    assert call["input_hash"] == debug_input_hash(messages)
    # HIT does not latch run-live.
    assert Laminar.is_debug_run_live() is False


def test_cache_outcome_miss_latches_run_live():
    runtime = _FakeRuntime(sync_outcome=CacheOutcome(kind="miss"))
    debug._runtime = runtime
    span = _span_with_messages([{"role": "user", "content": "hi"}])

    outcome = cache_outcome_for(span)

    assert outcome.kind == "miss"
    assert Laminar.is_debug_run_live() is True
    # Once latched, the next call short-circuits to live WITHOUT hitting the
    # endpoint again.
    second = cache_outcome_for(span)
    assert second.kind == "live"
    assert len(runtime._sync_cache.calls) == 1


def test_cache_outcome_live_does_not_latch():
    runtime = _FakeRuntime(sync_outcome=CacheOutcome(kind="live"))
    debug._runtime = runtime
    span = _span_with_messages([{"role": "user", "content": "hi"}])

    first = cache_outcome_for(span)
    assert first.kind == "live"
    assert Laminar.is_debug_run_live() is False
    # No latch -> the endpoint is retried on the next call.
    cache_outcome_for(span)
    assert len(runtime._sync_cache.calls) == 2


def test_cache_outcome_short_circuits_when_already_live():
    runtime = _FakeRuntime(sync_outcome=CacheOutcome(kind="hit", cached={}))
    debug._runtime = runtime
    Laminar.set_debug_run_live(True)

    outcome = cache_outcome_for(_span_with_messages([{"role": "user"}]))

    assert outcome.kind == "live"
    assert runtime._sync_cache.calls == []


# --- acache_outcome_for (async) ------------------------------------------


def test_acache_outcome_hit():
    cached = {"attributes": {}, "output": "x"}
    runtime = _FakeRuntime(async_outcome=CacheOutcome(kind="hit", cached=cached))
    debug._runtime = runtime
    messages = [{"role": "user", "content": "hi"}]

    outcome = asyncio.run(acache_outcome_for(_span_with_messages(messages)))

    assert outcome.kind == "hit"
    assert outcome.cached is cached
    assert runtime._async_cache.calls[0]["input_hash"] == debug_input_hash(messages)


def test_acache_outcome_miss_latches_run_live():
    runtime = _FakeRuntime(async_outcome=CacheOutcome(kind="miss"))
    debug._runtime = runtime
    span = _span_with_messages([{"role": "user", "content": "hi"}])

    outcome = asyncio.run(acache_outcome_for(span))

    assert outcome.kind == "miss"
    assert Laminar.is_debug_run_live() is True


def test_acache_outcome_none_when_replay_not_configured():
    debug._runtime = _FakeRuntime(replay_configured=False)
    outcome = asyncio.run(
        acache_outcome_for(_span_with_messages([{"role": "user"}]))
    )
    assert outcome is None


# --- mark_span_cached -----------------------------------------------------


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


# --- cross-language parity vector ----------------------------------------

_HASH_VECTORS = json.loads(
    (Path(__file__).parent / "data" / "debug" / "input_hash_cases.json").read_text(
        encoding="utf-8"
    )
)["cases"]


@pytest.mark.parametrize(
    "case", _HASH_VECTORS, ids=[c["name"] for c in _HASH_VECTORS]
)
def test_input_hash_matches_shared_vector(case):
    # Pins debug_input_hash against the shared cross-language vector. The TS SDK
    # and app-server must produce byte-identical hashes for the same inputs.
    assert debug_input_hash(case["messages"]) == case["expected_hash"]

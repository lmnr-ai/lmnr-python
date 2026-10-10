import asyncio
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest
from opentelemetry.util.types import AttributeValue
from typing_extensions import TypedDict

from lmnr.sdk import debug
from lmnr.sdk.client.synchronous.resources.rollout_sessions import (
    _parse_cache_outcome,
)
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


class CachedSpan(TypedDict):
    start_time: int
    end_time: int
    output: str
    attributes: dict[str, AttributeValue]


class _FakeSpan:
    def __init__(
        self,
        attributes: dict[str, AttributeValue | dict[str, AttributeValue] | list[dict[str, AttributeValue]]] | None = None,
        recording: bool = True
    ):
        self.attributes: dict[str, AttributeValue | dict[str, AttributeValue] | list[dict[str, AttributeValue]]] = attributes or {}
        self._recording: bool = recording
        self.marked: dict[str, AttributeValue] = {}

    def is_recording(self):
        return self._recording

    def set_attributes(self, attrs: dict[str, AttributeValue]):
        self.marked.update(attrs)


class _RecordingCache:
    """Sync rollout-session cache double; records calls and returns a scripted outcome."""

    def __init__(self, outcome: CacheOutcome):
        self.outcome: CacheOutcome = outcome
        self.calls: list[dict[str, str]] = []

    def cache(
        self,
        *,
        session_id: str,
        replay_trace_id: str,
        cache_until: str,
        input_hash: str,
    ) -> CacheOutcome:
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
        self.outcome: CacheOutcome = outcome
        self.calls: list[dict[str, str]] = []

    async def cache(
        self,
        *,
        session_id: str,
        replay_trace_id: str,
        cache_until: str,
        input_hash: str,
    ) -> CacheOutcome:
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
    def __init__(self, cache_resource: Any):
        self.rollout_sessions: Any = cache_resource


class _FakeRuntime:
    def __init__(
        self,
        *,
        replay_configured: bool = True,
        sync_outcome: CacheOutcome | None = None,
        async_outcome: CacheOutcome | None = None,
        session_id: str = "sess-1",
        replay_trace_id: str = "trace-1",
        cache_until_span_id: str = "abcdef",
    ):
        self.replay_configured: bool = replay_configured
        self.session_id: str = session_id
        self.replay_trace_id: str = replay_trace_id
        self.cache_until_span_id: str = cache_until_span_id
        self._sync_cache: _RecordingCache = _RecordingCache(sync_outcome or CacheOutcome(kind="live"))
        self._async_cache: _AsyncRecordingCache = _AsyncRecordingCache(
            async_outcome or CacheOutcome(kind="live")
        )
        self.client: _FakeClient = _FakeClient(self._sync_cache)
        self.async_client: _FakeClient = _FakeClient(self._async_cache)


@pytest.fixture(autouse=True)
def _clean_replay_state() -> Generator[None]:
    debug._runtime = None
    debug._initialized = False
    Laminar.set_debug_run_live(False)
    yield
    debug._runtime = None
    debug._initialized = False
    Laminar.set_debug_run_live(False)


def _span_with_messages(messages: list[dict[str, Any]]) -> _FakeSpan:
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
    assert input_messages_from_span(cast(Any, span)) == [{"role": "user", "content": "hi"}]


def test_input_messages_accepts_already_decoded_list():
    span = _FakeSpan({GEN_AI_INPUT_MESSAGES_ATTRIBUTE: [{"role": "user"}]})
    assert input_messages_from_span(cast(Any, span)) == [{"role": "user"}]


def test_input_messages_none_when_missing():
    assert input_messages_from_span(cast(Any, _FakeSpan({}))) is None
    assert input_messages_from_span(None) is None


def test_input_messages_none_on_bad_json():
    span = _FakeSpan({GEN_AI_INPUT_MESSAGES_ATTRIBUTE: "{not json"})
    assert input_messages_from_span(cast(Any, span)) is None


def test_input_messages_none_when_not_a_list():
    span = _FakeSpan({GEN_AI_INPUT_MESSAGES_ATTRIBUTE: json.dumps({"role": "user"})})
    assert input_messages_from_span(cast(Any, span)) is None


# --- cache_outcome_for (sync) --------------------------------------------


def test_cache_outcome_none_without_runtime():
    assert cache_outcome_for(cast(Any, _span_with_messages([{"role": "user"}]))) is None


def test_cache_outcome_none_when_replay_not_configured():
    debug._runtime = _FakeRuntime(replay_configured=False)
    assert cache_outcome_for(cast(Any, _span_with_messages([{"role": "user"}]))) is None


def test_cache_outcome_none_when_no_input_messages():
    debug._runtime = _FakeRuntime()
    # No usable input on the span -> nothing to hash -> run live, no latch.
    assert cache_outcome_for(cast(Any, _FakeSpan({}))) is None


def test_cache_outcome_hit_passes_hash_and_returns_cached():
    cached: CachedSpan = {"attributes": {}, "output": "x", "start_time": 0, "end_time": 0}
    runtime = _FakeRuntime(sync_outcome=CacheOutcome(kind="hit", cached=cast(Any, cached)))
    debug._runtime = runtime
    messages = [{"role": "user", "content": "hi"}]

    outcome = cache_outcome_for(cast(Any, _span_with_messages(messages)))

    assert outcome is not None
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

    outcome = cache_outcome_for(cast(Any, span))

    assert outcome is not None
    assert outcome.kind == "miss"
    assert Laminar.is_debug_run_live() is True
    # Once latched, the next call short-circuits to live WITHOUT hitting the
    # endpoint again.
    second = cache_outcome_for(cast(Any, span))
    assert second is not None
    assert second.kind == "live"
    assert len(runtime._sync_cache.calls) == 1


def test_cache_outcome_live_does_not_latch():
    runtime = _FakeRuntime(sync_outcome=CacheOutcome(kind="live"))
    debug._runtime = runtime
    span = _span_with_messages([{"role": "user", "content": "hi"}])

    first = cache_outcome_for(cast(Any, span))
    assert first is not None
    assert first.kind == "live"
    assert Laminar.is_debug_run_live() is False
    # No latch -> the endpoint is retried on the next call.
    _second = cache_outcome_for(cast(Any, span))
    assert len(runtime._sync_cache.calls) == 2


def test_cache_outcome_short_circuits_when_already_live():
    runtime = _FakeRuntime(sync_outcome=CacheOutcome(kind="hit", cached={}))
    debug._runtime = runtime
    Laminar.set_debug_run_live(True)

    outcome = cache_outcome_for(cast(Any, _span_with_messages([{"role": "user"}])))

    assert outcome is not None
    assert outcome.kind == "live"
    assert runtime._sync_cache.calls == []


# --- acache_outcome_for (async) ------------------------------------------


def test_acache_outcome_hit():
    cached: CachedSpan = {"attributes": {}, "output": "x", "start_time": 0, "end_time": 0}
    runtime = _FakeRuntime(async_outcome=CacheOutcome(kind="hit", cached=cast(Any, cached)))
    debug._runtime = runtime
    messages = [{"role": "user", "content": "hi"}]

    outcome = asyncio.run(acache_outcome_for(cast(Any, _span_with_messages(messages))))

    assert outcome is not None
    assert outcome.kind == "hit"
    assert outcome.cached is cached
    assert runtime._async_cache.calls[0]["input_hash"] == debug_input_hash(messages)


def test_acache_outcome_miss_latches_run_live():
    runtime = _FakeRuntime(async_outcome=CacheOutcome(kind="miss"))
    debug._runtime = runtime
    span = _span_with_messages([{"role": "user", "content": "hi"}])

    outcome = asyncio.run(acache_outcome_for(cast(Any, span)))

    assert outcome is not None
    assert outcome.kind == "miss"
    assert Laminar.is_debug_run_live() is True


def test_acache_outcome_none_when_replay_not_configured():
    debug._runtime = _FakeRuntime(replay_configured=False)
    outcome = asyncio.run(
        acache_outcome_for(cast(Any, _span_with_messages([{"role": "user"}])))
    )
    assert outcome is None


# --- mark_span_cached -----------------------------------------------------


def test_mark_span_cached_sets_boundary_attributes():
    span = _FakeSpan({})
    mark_span_cached(cast(Any, span))
    assert span.marked == {
        "lmnr.span.type": "CACHED",
        "lmnr.span.original_type": "LLM",
    }


def test_mark_span_cached_noop_when_not_recording():
    span = _FakeSpan({}, recording=False)
    mark_span_cached(cast(Any, span))
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
def test_input_hash_matches_shared_vector(case: dict[str, Any]):
    # Pins debug_input_hash against the shared cross-language vector. The TS SDK
    # and app-server must produce byte-identical hashes for the same inputs.
    assert debug_input_hash(case["messages"]) == case["expected_hash"]


# --- _parse_cache_outcome boundary ---------------------------------------


def test_parse_outcome_hit_with_response():
    out = _parse_cache_outcome({"outcome": "hit", "response": {"type": "raw"}})
    assert out.kind == "hit"
    assert out.cached == {"type": "raw"}


def test_parse_outcome_hit_without_response_degrades_to_live():
    # A HIT must carry a response envelope; the provider wrappers call
    # cached_response_to_*(cached) which does cached.get(...). A null/omitted
    # response would crash with AttributeError, so it must degrade to `live`.
    assert _parse_cache_outcome({"outcome": "hit"}).kind == "live"
    none_resp = {"outcome": "hit", "response": None}
    assert _parse_cache_outcome(none_resp).kind == "live"


def test_parse_outcome_miss_and_unknown():
    assert _parse_cache_outcome({"outcome": "miss"}).kind == "miss"
    assert _parse_cache_outcome({"outcome": "bogus"}).kind == "live"
    assert _parse_cache_outcome(None).kind == "live"

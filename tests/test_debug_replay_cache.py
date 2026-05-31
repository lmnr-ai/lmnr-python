from lmnr.sdk.debug.replay_cache import ReplayCache


def _payloads(n: int) -> list[dict]:
    return [{"output": f"resp-{i}"} for i in range(n)]


def test_replays_first_n_occurrences_in_order():
    cache = ReplayCache("loop.llm", cache_until=3, payloads=_payloads(5))

    seen = []
    for _ in range(3):
        occ = cache.next_occurrence("loop.llm")
        seen.append(cache.get_cached("loop.llm", occ))

    assert seen == [{"output": "resp-0"}, {"output": "resp-1"}, {"output": "resp-2"}]


def test_occurrence_counter_increments_per_path():
    cache = ReplayCache("loop.llm", cache_until=2, payloads=_payloads(2))
    assert cache.next_occurrence("loop.llm") == 0
    assert cache.next_occurrence("loop.llm") == 1
    assert cache.next_occurrence("loop.llm") == 2
    # A different path has its own independent counter.
    assert cache.next_occurrence("other") == 0


def test_returns_none_for_non_spine_path():
    cache = ReplayCache("loop.llm", cache_until=3, payloads=_payloads(3))
    assert cache.get_cached("other.path", 0) is None


def test_returns_none_past_cache_until():
    cache = ReplayCache("loop.llm", cache_until=2, payloads=_payloads(5))
    assert cache.get_cached("loop.llm", 1) == {"output": "resp-1"}
    assert cache.get_cached("loop.llm", 2) is None


def test_returns_none_when_fewer_payloads_than_cache_until():
    cache = ReplayCache("loop.llm", cache_until=5, payloads=_payloads(2))
    assert cache.get_cached("loop.llm", 0) == {"output": "resp-0"}
    assert cache.get_cached("loop.llm", 1) == {"output": "resp-1"}
    assert cache.get_cached("loop.llm", 2) is None


def test_payloads_truncated_to_cache_until():
    cache = ReplayCache("loop.llm", cache_until=2, payloads=_payloads(5))
    assert cache.get_cached("loop.llm", 0) == {"output": "resp-0"}
    assert cache.get_cached("loop.llm", 1) == {"output": "resp-1"}
    assert cache.get_cached("loop.llm", 2) is None

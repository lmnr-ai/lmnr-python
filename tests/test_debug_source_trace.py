import datetime

from lmnr.sdk.debug.source_trace import _to_epoch, fetch_spine_metadata
from lmnr.sdk.debug.spine import has_overlap


class _FakeSql:
    def __init__(self, metadata_rows):
        self._metadata_rows = metadata_rows

    def query(self, query, parameters=None):
        rows, self._metadata_rows = self._metadata_rows, []
        return rows


class _FakeClient:
    def __init__(self, metadata_rows):
        self.sql = _FakeSql(metadata_rows)


def test_missing_end_time_treated_as_unbounded():
    # A null end_time must not collapse to 0.0 — otherwise the overlap guard
    # (start < prev.end) never fires and replay proceeds when it should not.
    client = _FakeClient(
        [
            {"path": "loop.llm", "span_type": "LLM", "start_time": 0.0, "end_time": None},
            {"path": "loop.llm", "span_type": "LLM", "start_time": 1.0, "end_time": 2.0},
        ]
    )
    records = fetch_spine_metadata(client, "trace-1")

    assert records[0].end_time == float("inf")
    # The unbounded first call overlaps the second -> guard fires -> run live.
    assert has_overlap(records, 2) is True


def test_present_end_time_unaffected():
    client = _FakeClient(
        [
            {"path": "loop.llm", "span_type": "LLM", "start_time": 0.0, "end_time": 1.0},
            {"path": "loop.llm", "span_type": "LLM", "start_time": 1.0, "end_time": 2.0},
        ]
    )
    records = fetch_spine_metadata(client, "trace-1")

    assert records[0].end_time == 1.0
    assert has_overlap(records, 2) is False


def test_to_epoch_parses_nanosecond_iso():
    # ClickHouse returns DateTime64(9) (nanosecond) strings. datetime.fromisoformat
    # rejects >6 fractional digits before Python 3.11, so the >6-digit fractional
    # part must be truncated to microseconds rather than falling to the fallback.
    expected = datetime.datetime.fromisoformat(
        "2024-01-15T10:30:45.123456+00:00"
    ).timestamp()
    assert _to_epoch("2024-01-15T10:30:45.123456789Z") == expected


def test_to_epoch_lexical_fallback_preserves_order():
    # Unparseable-but-ISO-ordered strings must map monotonically: a later
    # timestamp can never score lower than an earlier one (a flat ordinal sum
    # ignores position and reversed them). Append a non-numeric tail to force
    # the fallback while keeping the resolvable leading chars distinct.
    a = "2024-01-15T10:30:45!"
    b = "2024-02-15T10:30:45!"
    assert a < b
    assert _to_epoch(a) < _to_epoch(b)

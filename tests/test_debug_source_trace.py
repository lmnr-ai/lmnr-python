from lmnr.sdk.debug.source_trace import fetch_spine_metadata
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

import json
from pathlib import Path

import pytest

from lmnr.sdk.debug.spine import SpanRecord, detect_spine, has_overlap

_DATA = Path(__file__).parent / "data" / "debug"
_SPINE_VECTORS = json.loads((_DATA / "spine_vectors.json").read_text())["cases"]
_OVERLAP_VECTORS = json.loads((_DATA / "overlap_vectors.json").read_text())["cases"]


@pytest.mark.parametrize(
    "case", _SPINE_VECTORS, ids=[c["name"] for c in _SPINE_VECTORS]
)
def test_detect_spine(case):
    spans = [
        SpanRecord(
            span_path=s["span_path"],
            span_type=s["span_type"],
            start_time=s["start_time"],
            end_time=s["end_time"],
        )
        for s in case["spans"]
    ]
    result = detect_spine(spans)
    expect = case["expect"]

    assert result.spine_path == expect["spine_path"]
    assert [c.start_time for c in result.spine_calls] == expect["spine_starts"]


@pytest.mark.parametrize(
    "case", _OVERLAP_VECTORS, ids=[c["name"] for c in _OVERLAP_VECTORS]
)
def test_has_overlap(case):
    calls = [
        SpanRecord(
            span_path="loop.llm",
            span_type="LLM",
            start_time=c["start_time"],
            end_time=c["end_time"],
        )
        for c in case["spine_calls"]
    ]
    assert has_overlap(calls, case["n"]) is case["expect"]

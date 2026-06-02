import json
from pathlib import Path

import pytest

from lmnr.sdk.debug.spine import (
    SpanRecord,
    detect_spine,
    has_overlap,
    resolve_cache_until_span_id,
)

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


def _spine(*span_ids: str) -> list[SpanRecord]:
    return [
        SpanRecord(
            span_path="loop.llm",
            span_type="LLM",
            start_time=float(i),
            end_time=float(i) + 0.5,
            span_id=sid,
        )
        for i, sid in enumerate(span_ids)
    ]


_FULL_UUID = "00000000-0000-0000-0123-456789abcdef"


@pytest.mark.parametrize(
    "needle",
    [
        # Full UUID, last two groups, raw 16-hex, short hex suffix — all forms
        # the user might copy for the same span id.
        "00000000000000000123456789abcdef",
        "0123456789abcdef",
        "abcdef",
    ],
)
def test_resolve_cache_until_span_id_matches_forms(needle):
    spine = _spine("11111111-1111-1111-1111-111111111111", _FULL_UUID)
    # The target is the 2nd call, so the resolved count is 2 (inclusive).
    assert resolve_cache_until_span_id(spine, needle) == 2


def test_resolve_cache_until_span_id_returns_first_occurrence_count():
    spine = _spine(_FULL_UUID, "22222222-2222-2222-2222-222222222222")
    assert resolve_cache_until_span_id(spine, "456789abcdef") == 1


def test_resolve_cache_until_span_id_none_when_not_found():
    spine = _spine("11111111-1111-1111-1111-111111111111")
    assert resolve_cache_until_span_id(spine, "deadbeef") is None


def test_resolve_cache_until_span_id_none_on_empty_spine():
    assert resolve_cache_until_span_id([], "abcdef") is None

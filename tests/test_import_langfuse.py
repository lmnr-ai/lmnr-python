"""Unit tests for the pure mapping in `lmnr import langfuse` (no network)."""

from lmnr.cli.import_langfuse import (
    as_json_string,
    dataset_item_to_datapoint,
    index_scores,
    observation_to_span,
    resolve_parent,
    span_type,
    to_span_id,
    to_trace_id,
    to_unix_nanos,
)


def _attrs(span: dict) -> dict:
    out = {}
    for kv in span["attributes"]:
        value = kv["value"]
        if "arrayValue" in value:
            out[kv["key"]] = [
                next(iter(v.values())) for v in value["arrayValue"]["values"]
            ]
        else:
            out[kv["key"]] = next(iter(value.values()))
    return out


GENERATION = {
    "id": "e128b90e2f8226ff",
    "traceId": "2f0fef0cdb2e46179fe909d7cab6f97d",
    "parentObservationId": "9418beee80fcadfd",
    "type": "GENERATION",
    "name": "claude-sonnet-4-6",
    "startTime": "2026-10-06T00:39:33.073Z",
    "endTime": "2026-10-06T00:40:13.450Z",
    "level": "DEFAULT",
    "model": "claude-sonnet-4-6",
    "input": '[{"role": "user", "content": "hi"}]',
    "output": "plain text answer",
    "usageDetails": {"input": 1524, "output": 2125, "cache_read_input_tokens": 15884},
    "inputUsage": 17408,
    "outputUsage": 2125,
    "inputCost": 0.0192111,
    "outputCost": 0.031875,
    "totalCost": 0.0510861,
    "sessionId": "session-1",
    "userId": "culture",
    "tags": ["obscura", "culture"],
}


def test_v4_ids_are_kept_and_other_ids_hash_deterministically():
    assert (
        to_trace_id("2F0FEF0CDB2E46179FE909D7CAB6F97D")
        == "2f0fef0cdb2e46179fe909d7cab6f97d"
    )
    assert (
        to_trace_id("2f0fef0c-db2e-4617-9fe9-09d7cab6f97d")
        == "2f0fef0cdb2e46179fe909d7cab6f97d"
    )
    assert to_span_id("e128b90e2f8226ff") == "e128b90e2f8226ff"
    hashed = to_trace_id("my-custom-trace")
    assert len(hashed) == 32 and hashed == to_trace_id("my-custom-trace")
    assert len(to_span_id("cl9x0abc")) == 16 and to_span_id("cl9x0abc") == to_span_id(
        "cl9x0abc"
    )


def test_span_type_collapses_langfuse_observation_types():
    assert span_type("GENERATION") == "LLM"
    assert span_type("EMBEDDING") == "LLM"
    assert span_type("TOOL") == "TOOL"
    for other in ("SPAN", "AGENT", "CHAIN", "RETRIEVER", "EVENT", None):
        assert span_type(other) == "DEFAULT"


def test_timestamps_keep_microseconds_exactly():
    assert to_unix_nanos("2026-10-06T00:39:33.073Z") == "1791247173073000000"
    assert to_unix_nanos("2026-10-06T00:39:33.000001Z") == "1791247173000001000"
    assert to_unix_nanos(None) is None


def test_io_is_sent_as_json_documents():
    assert as_json_string('{"a": 1}') == '{"a": 1}'
    assert as_json_string("plain text") == '"plain text"'
    assert as_json_string({"a": 1}) == '{"a":1}'


def test_generation_maps_to_an_llm_span_with_total_input_tokens_and_source_cost():
    span = observation_to_span(GENERATION, parent_id="9418beee80fcadfd")
    attrs = _attrs(span)
    assert span["traceId"] == GENERATION["traceId"]
    assert span["spanId"] == GENERATION["id"]
    assert span["parentSpanId"] == "9418beee80fcadfd"
    assert attrs["lmnr.span.type"] == "LLM"
    assert attrs["langfuse.observation.type"] == "GENERATION"
    assert attrs["gen_ai.request.model"] == "claude-sonnet-4-6"
    # Laminar's input_tokens is the total incl. cache reads (inputUsage),
    # not usageDetails["input"].
    assert attrs["gen_ai.usage.input_tokens"] == "17408"
    assert attrs["gen_ai.usage.cache_read_input_tokens"] == "15884"
    assert attrs["gen_ai.usage.output_tokens"] == "2125"
    assert attrs["gen_ai.usage.cost"] == 0.0510861
    assert attrs["lmnr.span.output"] == '"plain text answer"'
    assert attrs["lmnr.association.properties.session_id"] == "session-1"
    assert attrs["lmnr.association.properties.tags"] == ["obscura", "culture"]
    assert "trace.name" not in attrs


def test_zero_cost_is_not_sent_so_laminar_prices_the_span():
    span = observation_to_span(
        {**GENERATION, "inputCost": 0, "outputCost": 0, "totalCost": 0}, parent_id=None
    )
    assert not any(
        k.startswith("gen_ai.usage.") and k.endswith("cost") for k in _attrs(span)
    )


def test_dangling_parent_becomes_a_root():
    root = {
        **GENERATION,
        "id": "9418beee80fcadfd",
        "parentObservationId": "2f0fef0cdb2e4617",
    }
    child = GENERATION
    ids = {root["id"], child["id"]}
    assert resolve_parent(root, ids) is None
    assert resolve_parent(child, ids) == "9418beee80fcadfd"


def test_root_carries_trace_name_metadata_and_trace_scores_but_not_langfuse_internals():
    root = {
        **GENERATION,
        "type": "AGENT",
        "name": "run",
        "metadata": {
            "shift_days": 88,
            "scope.name": "langfuse-sdk",
            "resourceAttributes.x": "y",
            "nested": {"a": 1},
        },
        "environment": "production",
    }
    scores = [
        {"name": "verdict", "dataType": "CATEGORICAL", "value": "CAUTION"},
        {"name": "confidence", "dataType": "NUMERIC", "value": "60"},
    ]
    attrs = _attrs(observation_to_span(root, parent_id=None, trace_scores=scores))
    md = "lmnr.association.properties.metadata."
    assert attrs["trace.name"] == "run"
    assert attrs[md + "shift_days"] == "88"
    assert attrs[md + "nested"] == '{"a":1}'
    assert attrs[md + "environment"] == "production"
    assert attrs[md + "score.verdict"] == "CAUTION"
    assert attrs[md + "score.confidence"] == "60"
    assert not any(
        k.startswith(md + "scope.") or k.startswith(md + "resourceAttributes")
        for k in attrs
    )


def test_observation_scores_and_errors_stay_on_the_span():
    error = {
        **GENERATION,
        "level": "ERROR",
        "statusMessage": "tool failed",
        "type": "TOOL",
    }
    scores = [{"name": "correct", "dataType": "BOOLEAN", "value": 1}]
    span = observation_to_span(
        error, parent_id="9418beee80fcadfd", observation_scores=scores
    )
    attrs = _attrs(span)
    assert attrs["lmnr.span.type"] == "TOOL"
    assert attrs["langfuse.score.correct"] == "1"
    assert span["status"] == {"code": 2, "message": "tool failed"}


def test_scores_are_indexed_by_subject():
    by_trace, by_observation = index_scores(
        [
            {"name": "a", "subject": {"kind": "trace", "id": "t1"}},
            {"name": "b", "subject": {"kind": "observation", "id": "o1"}},
            {"name": "c", "traceId": "t2"},
        ]
    )
    assert [s["name"] for s in by_trace["t1"]] == ["a"]
    assert [s["name"] for s in by_trace["t2"]] == ["c"]
    assert [s["name"] for s in by_observation["o1"]] == ["b"]


def test_dataset_items_become_datapoints():
    dp = dataset_item_to_datapoint(
        {
            "id": "item-1",
            "input": {"q": 1},
            "expectedOutput": {"a": 2},
            "metadata": {"split": "dev"},
        }
    )
    assert dp == {
        "data": {"q": 1},
        "target": {"a": 2},
        "metadata": {"split": "dev", "langfuse.dataset_item_id": "item-1"},
    }
    assert "target" not in dataset_item_to_datapoint({"id": "i", "input": "q"})

"""No VCR here: `typesafe-sdk` runs on `httpx2`, which vcrpy cannot intercept.

Responses are injected through the SDK's own `transport=` seam
(`httpx2.MockTransport`) and are validated by the SDK's strict
`SystemOneResponse` pydantic model, so a wrong-shaped body fails in the SDK
rather than silently passing.
"""

import json

import httpx2
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    Score,
    TypeSafeBadRequestError,
    TypeSafeClient,
)

MODEL = "jev-1.13.0"

STATE = "I was charged twice for my annual plan. Please refund one of the charges."

QUESTIONS = {
    "wants_refund": Noul(instructions="Does the customer ask for money back?"),
    "queue": Choice(
        instructions="Which team should handle this?",
        criteria={"billing": "Charges and refunds", "other": None},
    ),
    "urgency": Score(
        instructions="How urgent is this message?",
        criteria=["Can wait a week", "This week", "Today"],
    ),
}

QUESTIONS_AS_DICTS = {
    "wants_refund": {
        "type": "noul",
        "instructions": "Does the customer ask for money back?",
    },
    "queue": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges and refunds", "other": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this message?",
        "criteria": ["Can wait a week", "This week", "Today"],
    },
}

ANSWERS = {
    "wants_refund": {"type": "noul", "noul": 0.98},
    "queue": {
        "type": "choice",
        "choice": "billing",
        "confidence": 0.9,
        "probabilities": {"billing": 0.9, "other": 0.1},
    },
    "urgency": {
        "type": "score",
        "score": 1.7,
        "confidence": 0.8,
        "legend": {"0": "Can wait a week", "1": "This week", "2": "Today"},
        "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
    },
}

RESPONSE_BODY = {
    "model": MODEL,
    "usage": {"input_tokens": 312, "output_tokens": 48},
    "answers": ANSWERS,
}


def _ok_handler(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json=RESPONSE_BODY)


def _client(handler=_ok_handler, **kwargs) -> TypeSafeClient:
    return TypeSafeClient(
        api_key="test-key", transport=httpx2.MockTransport(handler), **kwargs
    )


def _async_client(handler=_ok_handler, **kwargs) -> AsyncTypeSafeClient:
    return AsyncTypeSafeClient(
        api_key="test-key", transport=httpx2.MockTransport(handler), **kwargs
    )


def _assert_span_shape(span):
    assert span.name == "typesafe.system_one"
    assert span.attributes["lmnr.span.type"] == "LLM"
    assert span.attributes["gen_ai.system"] == "typesafe"
    assert span.attributes["lmnr.span.instrumentation_scope.name"] == "typesafe"
    assert span.attributes["lmnr.span.instrumentation_scope.version"]
    assert span.attributes["gen_ai.response.model"] == MODEL
    assert span.attributes["gen_ai.usage.input_tokens"] == 312
    assert span.attributes["gen_ai.usage.output_tokens"] == 48

    input_messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert input_messages == [{"role": "user", "content": STATE}]
    schema = json.loads(span.attributes["gen_ai.request.structured_output_schema"])
    assert schema == QUESTIONS_AS_DICTS
    output_messages = json.loads(span.attributes["gen_ai.output.messages"])
    assert output_messages[0]["role"] == "assistant"
    assert json.loads(output_messages[0]["content"]) == ANSWERS


def test_typesafe_system_one(span_exporter: InMemorySpanExporter):
    response = _client().system_one(
        state=STATE, questions=QUESTIONS, model=MODEL
    )
    assert response.answers["queue"].choice == "billing"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    _assert_span_shape(span)
    assert span.attributes["gen_ai.request.model"] == MODEL
    assert span.status.status_code == StatusCode.UNSET


def test_typesafe_system_one_positional_args(span_exporter: InMemorySpanExporter):
    # `state` and `questions` are positional-or-keyword; the client default
    # model backs `gen_ai.request.model` when no per-call model is given.
    _client(model=MODEL).system_one(STATE, QUESTIONS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    _assert_span_shape(span)
    assert span.attributes["gen_ai.request.model"] == MODEL


@pytest.mark.asyncio
async def test_typesafe_system_one_async(span_exporter: InMemorySpanExporter):
    async with _async_client() as client:
        response = await client.system_one(
            state=STATE, questions=QUESTIONS, model=MODEL
        )
    assert response.nouls["wants_refund"].noul == 0.98

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    _assert_span_shape(span)
    assert span.attributes["gen_ai.request.model"] == MODEL


def test_typesafe_system_one_dict_state(span_exporter: InMemorySpanExporter):
    state = {"message": STATE, "customer_tier": "pro"}
    _client().system_one(state=state, questions=QUESTIONS, model=MODEL)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    input_messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert input_messages[0]["role"] == "user"
    assert json.loads(input_messages[0]["content"]) == state


def test_typesafe_system_one_error(span_exporter: InMemorySpanExporter):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            400, json={"error": {"message": "Unknown model: jev-1.13"}}
        )

    with pytest.raises(TypeSafeBadRequestError):
        _client(handler).system_one(
            state=STATE, questions=QUESTIONS, model="jev-1.13"
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "typesafe.system_one"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "TypeSafeBadRequestError"
    assert span.attributes["gen_ai.request.model"] == "jev-1.13"
    assert "gen_ai.response.model" not in span.attributes


def test_typesafe_system_one_no_trace_content(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LMNR_TRACE_CONTENT", "false")
    _client().system_one(state=STATE, questions=QUESTIONS, model=MODEL)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.output.messages" not in span.attributes
    assert "gen_ai.request.structured_output_schema" not in span.attributes
    assert span.attributes["gen_ai.request.model"] == MODEL
    assert span.attributes["gen_ai.response.model"] == MODEL
    assert span.attributes["gen_ai.usage.input_tokens"] == 312
    assert span.attributes["gen_ai.usage.output_tokens"] == 48

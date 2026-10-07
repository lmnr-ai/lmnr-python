"""Decisions API (`client.decisions.create`, openai>=3.26.0).

These run the real SDK resource against an in-process `httpx2.MockTransport`
instead of VCR cassettes: the dev env pins `openai==2.30.0` (litellm caps
`openai<3`), which has no Decisions resource, so this module is skipped there
and CI runs it in a separate step with `--with openai==3.26.0`. Response bodies
follow `openai.types.decision.Decision`.
"""

import json

import pytest

pytest.importorskip("openai.resources.decisions")

import httpx2
from openai import AsyncOpenAI, BadRequestError, OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

MODEL = "decision-model"

QUESTIONS = [
    {"type": "predicate", "name": "positive", "instructions": "Is it positive?"},
    {
        "type": "choice",
        "name": "topic",
        "instructions": "What is the topic?",
        "choices": [{"value": "movies"}, {"value": "food"}],
    },
    {
        "type": "score",
        "name": "intensity",
        "instructions": "How strong is the sentiment?",
        "levels": [{"label": "weak"}, {"label": "strong"}],
    },
    {"type": "predicate", "name": "unsafe", "instructions": "Is it unsafe?"},
]

ANSWERS = [
    {"type": "predicate", "name": "positive", "probability": 0.97},
    {
        "type": "choice",
        "name": "topic",
        "choice": "movies",
        "confidence": 0.9,
        "probabilities": [
            {"value": "movies", "probability": 0.9},
            {"value": "food", "probability": 0.1},
        ],
    },
    {
        "type": "score",
        "name": "intensity",
        "score": 0.8,
        "confidence": 0.7,
        "probabilities": [
            {"label": "weak", "value": 0, "probability": 0.3},
            {"label": "strong", "value": 1, "probability": 0.7},
        ],
    },
    {"type": "refusal", "name": "unsafe"},
]

DECISION = {
    "model": f"{MODEL}-2026-09-01",
    "answers": ANSWERS,
    "usage": {
        "input_tokens": 120,
        "input_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 32},
        "output_tokens": 8,
        "output_tokens_details": {"reasoning_tokens": 4},
        "total_tokens": 128,
    },
}


def _handler(status_code: int = 200, body: dict | None = None):
    requests: list[dict] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/decisions"
        requests.append(json.loads(request.content))
        return httpx2.Response(status_code, json=DECISION if body is None else body)

    return handle, requests


def _client(handler) -> OpenAI:
    return OpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def _async_client(handler) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def _only_span(span_exporter: InMemorySpanExporter):
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def _assert_decision_span(span) -> None:
    assert span.name == "openai.decision"
    assert span.attributes["lmnr.span.type"] == "LLM"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.request.model"] == MODEL
    assert span.attributes["gen_ai.response.model"] == f"{MODEL}-2026-09-01"
    assert span.attributes["gen_ai.usage.input_tokens"] == 120
    assert span.attributes["gen_ai.usage.output_tokens"] == 8
    assert span.attributes["llm.usage.total_tokens"] == 128
    assert span.attributes["gen_ai.usage.cache_read_input_tokens"] == 64
    assert span.attributes["gen_ai.usage.cache_creation_input_tokens"] == 32
    assert span.attributes["gen_ai.usage.reasoning_tokens"] == 4

    output = json.loads(span.attributes["gen_ai.output.messages"])
    assert len(output) == 1
    assert output[0]["role"] == "assistant"
    # Pydantic fills the optional `name` the server always returns; every
    # answer type, including the refusal, keeps its native shape.
    assert json.loads(output[0]["content"]) == ANSWERS


def test_decisions_create(instrument_legacy, span_exporter: InMemorySpanExporter):
    handler, requests = _handler()
    decision = _client(handler).decisions.create(
        model=MODEL,
        input="The movie was wonderful.",
        questions=QUESTIONS,
        safety_identifier="user-123",
    )

    assert [a.type for a in decision.answers] == [
        "predicate",
        "choice",
        "score",
        "refusal",
    ]
    assert requests[0]["questions"] == QUESTIONS

    span = _only_span(span_exporter)
    _assert_decision_span(span)
    assert span.attributes["llm.user"] == "user-123"
    assert json.loads(span.attributes["gen_ai.input.messages"]) == [
        {"role": "system", "content": json.dumps(QUESTIONS, separators=(",", ":"))},
        {"role": "user", "content": "The movie was wonderful."},
    ]


@pytest.mark.asyncio
async def test_decisions_create_async_with_message_input(
    instrument_legacy, span_exporter: InMemorySpanExporter
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Review:"},
                {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0K"},
            ],
        }
    ]
    handler, requests = _handler()
    await _async_client(handler).decisions.create(
        model=MODEL, input=messages, questions=QUESTIONS
    )

    assert requests[0]["input"] == messages
    span = _only_span(span_exporter)
    _assert_decision_span(span)
    input_messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert input_messages[0]["role"] == "system"
    assert input_messages[1:] == messages


def test_decisions_create_with_raw_response(
    instrument_legacy, span_exporter: InMemorySpanExporter
):
    handler, _ = _handler()
    raw = _client(handler).decisions.with_raw_response.create(
        model=MODEL, input="The movie was wonderful.", questions=QUESTIONS
    )

    assert raw.parse().answers[0].type == "predicate"
    _assert_decision_span(_only_span(span_exporter))


def test_decisions_create_error(instrument_legacy, span_exporter: InMemorySpanExporter):
    handler, _ = _handler(
        status_code=400,
        body={"error": {"message": "bad questions", "type": "invalid_request_error"}},
    )
    with pytest.raises(BadRequestError):
        _client(handler).decisions.create(
            model=MODEL, input="The movie was wonderful.", questions=QUESTIONS
        )

    span = _only_span(span_exporter)
    assert span.name == "openai.decision"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "BadRequestError"
    assert span.attributes["gen_ai.request.model"] == MODEL
    assert "gen_ai.input.messages" in span.attributes
    assert "gen_ai.output.messages" not in span.attributes
    assert span.events[0].name == "exception"


def test_decisions_create_does_not_consume_iterator_params(
    instrument_legacy, span_exporter: InMemorySpanExporter
):
    handler, requests = _handler()
    _client(handler).decisions.create(
        model=MODEL,
        input="The movie was wonderful.",
        questions=(q for q in QUESTIONS),
    )

    # The SDK still sends every question; we just don't record them.
    assert requests[0]["questions"] == QUESTIONS
    span = _only_span(span_exporter)
    assert json.loads(span.attributes["gen_ai.input.messages"]) == [
        {"role": "user", "content": "The movie was wonderful."}
    ]


def test_decisions_create_without_content_tracing(
    instrument_legacy, span_exporter: InMemorySpanExporter, monkeypatch
):
    monkeypatch.setenv("LMNR_TRACE_CONTENT", "false")
    handler, _ = _handler()
    _client(handler).decisions.create(
        model=MODEL, input="The movie was wonderful.", questions=QUESTIONS
    )

    span = _only_span(span_exporter)
    assert span.attributes["gen_ai.usage.input_tokens"] == 120
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.output.messages" not in span.attributes

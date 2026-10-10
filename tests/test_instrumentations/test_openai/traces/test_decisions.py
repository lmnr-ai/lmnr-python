"""Decisions API (`client.decisions.create`, openai>=3.26.0).

The dev env pins `openai==2.30.0` (litellm caps `openai<3`), which has no
Decisions resource, so this module is skipped there and CI runs it in a
separate step with `--with openai==3.26.0`. Cassettes were recorded against
`gpt-6-luna`; assertions compare the span against the parsed response rather
than hard-coded scores, so re-recording doesn't require editing them.
"""

import json
from typing import Any, cast

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import (
    OpenAIInstrumentor,
)

pytest.importorskip("openai.resources.decisions")

from openai import AsyncOpenAI, BadRequestError, OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

MODEL = "gpt-6-luna"
INPUT = "The movie was wonderful, though the popcorn was stale."

QUESTIONS = [
    {"type": "predicate", "name": "positive", "instructions": "Is it positive?"},
    {
        "type": "choice",
        "name": "topic",
        "instructions": "What is the main topic?",
        "choices": [{"value": "movies"}, {"value": "food"}],
    },
    {
        "type": "score",
        "name": "intensity",
        "instructions": "How strong is the sentiment?",
        "levels": [{"label": "weak"}, {"label": "moderate"}, {"label": "strong"}],
    },
]

# 1x1 PNG
IMAGE_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQV"
    "R42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)


def _only_span(span_exporter: InMemorySpanExporter):
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def _assert_decision_span(span: ReadableSpan, decision: Any) -> None:
    attributes = span.attributes or {}
    assert span.name == "openai.decision"
    assert attributes["lmnr.span.type"] == "LLM"
    assert attributes["gen_ai.system"] == "openai"
    assert attributes["gen_ai.request.model"] == MODEL
    assert attributes["gen_ai.response.model"] == decision.model

    usage = decision.usage
    assert attributes["gen_ai.usage.input_tokens"] == usage.input_tokens
    assert attributes["gen_ai.usage.output_tokens"] == usage.output_tokens
    assert attributes["llm.usage.total_tokens"] == usage.total_tokens
    assert (
        attributes["gen_ai.usage.cache_read_input_tokens"]
        == usage.input_tokens_details.cached_tokens
    )
    assert (
        attributes["gen_ai.usage.cache_creation_input_tokens"]
        == usage.input_tokens_details.cache_write_tokens
    )
    assert (
        attributes["gen_ai.usage.reasoning_tokens"]
        == usage.output_tokens_details.reasoning_tokens
    )

    output = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert len(output) == 1
    assert output[0]["role"] == "assistant"
    assert json.loads(output[0]["content"]) == [
        answer.model_dump() for answer in decision.answers
    ]


@pytest.mark.vcr
def test_decisions_create(
    instrumentor: OpenAIInstrumentor, span_exporter: InMemorySpanExporter, openai_client: OpenAI
):
    decision = openai_client.decisions.create(  # pyright: ignore[reportAttributeAccessIssue] requires openai >= 3.26
        model=MODEL,
        input=INPUT,
        questions=QUESTIONS,
        safety_identifier="user-123",
    )

    assert [a.type for a in decision.answers] == ["predicate", "choice", "score"]

    span = _only_span(span_exporter)
    attributes = span.attributes or {}
    _assert_decision_span(span, decision)
    assert attributes["llm.user"] == "user-123"
    assert json.loads(cast(str, attributes["gen_ai.input.messages"])) == [
        {"role": "user", "content": INPUT}
    ]
    assert (
        json.loads(cast(str, attributes["gen_ai.request.structured_output_schema"]))
        == QUESTIONS
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_decisions_create_async_with_message_input(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": f"Review: {INPUT}"},
                {"type": "input_image", "image_url": IMAGE_URL},
            ],
        }
    ]
    decision = await async_openai_client.decisions.create(  # pyright: ignore[reportAttributeAccessIssue] requires openai >= 3.26
        model=MODEL, input=messages, questions=QUESTIONS
    )

    span = _only_span(span_exporter)
    attributes = span.attributes or {}
    _assert_decision_span(span, decision)
    assert json.loads(cast(str, attributes["gen_ai.input.messages"])) == messages
    assert (
        json.loads(cast(str, attributes["gen_ai.request.structured_output_schema"]))
        == QUESTIONS
    )


@pytest.mark.vcr
def test_decisions_create_with_raw_response(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    raw = openai_client.decisions.with_raw_response.create(  # pyright: ignore[reportAttributeAccessIssue] requires openai >= 3.26
        model=MODEL, input=INPUT, questions=QUESTIONS
    )

    _assert_decision_span(_only_span(span_exporter), raw.parse())


@pytest.mark.vcr
def test_decisions_create_error(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    with pytest.raises(BadRequestError):
        openai_client.decisions.create(  # pyright: ignore[reportAttributeAccessIssue] requires openai >= 3.26
            model=MODEL,
            input=INPUT,
            questions=[{"type": "choice", "instructions": "Pick one.", "choices": []}],
        )

    span = _only_span(span_exporter)
    attributes = span.attributes or {}
    assert span.name == "openai.decision"
    assert span.status.status_code == StatusCode.ERROR
    assert attributes["error.type"] == "BadRequestError"
    assert attributes["gen_ai.request.model"] == MODEL
    assert "gen_ai.input.messages" in attributes
    assert "gen_ai.request.structured_output_schema" in attributes
    assert "gen_ai.output.messages" not in attributes
    assert span.events[0].name == "exception"


@pytest.mark.vcr
def test_decisions_create_does_not_consume_iterator_params(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    decision = openai_client.decisions.create(  # pyright: ignore[reportAttributeAccessIssue] requires openai >= 3.26
        model=MODEL, input=INPUT, questions=(q for q in QUESTIONS)
    )

    # The SDK still sends every question; we just don't record them.
    assert len(decision.answers) == len(QUESTIONS)
    span = _only_span(span_exporter)
    attributes = span.attributes or {}
    assert json.loads(cast(str, attributes["gen_ai.input.messages"])) == [
        {"role": "user", "content": INPUT}
    ]
    assert "gen_ai.request.structured_output_schema" not in attributes


@pytest.mark.vcr
def test_decisions_create_without_content_tracing(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LMNR_TRACE_CONTENT", "false")
    decision = openai_client.decisions.create(  # pyright: ignore[reportAttributeAccessIssue] requires openai >= 3.26
        model=MODEL, input=INPUT, questions=QUESTIONS
    )

    span = _only_span(span_exporter)
    attributes = span.attributes or {}
    assert attributes["gen_ai.usage.input_tokens"] == decision.usage.input_tokens
    assert "gen_ai.input.messages" not in attributes
    assert "gen_ai.request.structured_output_schema" not in attributes
    assert "gen_ai.output.messages" not in attributes

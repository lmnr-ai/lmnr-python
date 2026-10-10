import json
from typing import cast

import pytest
from openai import AsyncOpenAI, AuthenticationError, OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import (
    OpenAIInstrumentor,
)


class StructuredAnswer(BaseModel):
    rating: int
    joke: str


@pytest.mark.vcr
def test_parsed_completion(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    _ = openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert output_messages[0]["message"]["content"]
    assert (
        attributes.get("gen_ai.request.base_url")
        == "https://api.openai.com/v1/"
    )
    assert attributes.get("llm.is_streaming") is False
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert (
        attributes.get("gen_ai.response.id")
        == "chatcmpl-AGC1gNoe1Zyq9yZicdhLc85lmt2Ep"
    )

    assert json.loads(cast(str, attributes.get("gen_ai.request.structured_output_schema"))) == StructuredAnswer.model_json_schema() | {"additionalProperties": False}


@pytest.mark.vcr
def test_parsed_refused_completion(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    _ = openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert output_messages[0]["message"].get("content") is None
    assert (
        output_messages[0]["message"]["refusal"]
        == "I'm very sorry, but I can't assist with that request."
    )
    assert (
        attributes.get("gen_ai.response.id")
        == "chatcmpl-AGky8KFDbg6f5fF4qLtsBredIjZZh"
    )
    assert json.loads(cast(str, attributes.get("gen_ai.request.structured_output_schema"))) == StructuredAnswer.model_json_schema() | {"additionalProperties": False}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_parsed_completion(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    _ = await async_openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert output_messages[0]["message"]["content"]
    assert (
        attributes.get("gen_ai.request.base_url")
        == "https://api.openai.com/v1/"
    )
    assert attributes.get("llm.is_streaming") is False
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert (
        attributes.get("gen_ai.response.id")
        == "chatcmpl-AGC1iysV7rZ0qZ510vbeKVTNxSOHB"
    )
    assert json.loads(cast(str, attributes.get("gen_ai.request.structured_output_schema"))) == StructuredAnswer.model_json_schema() | {"additionalProperties": False}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_parsed_refused_completion(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    _ = await async_openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert output_messages[0]["message"].get("content") is None
    assert (
        output_messages[0]["message"]["refusal"]
        == "I'm very sorry, but I can't assist with that request."
    )
    assert (
        attributes.get("gen_ai.response.id")
        == "chatcmpl-AGkyFJGzZPUGAAEDJJuOS3idKvD3G"
    )
    assert json.loads(cast(str, attributes.get("gen_ai.request.structured_output_schema"))) == StructuredAnswer.model_json_schema() | {"additionalProperties": False}


def test_parsed_completion_exception(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    openai_client.api_key = "invalid"
    with pytest.raises(AuthenticationError):
        _ = openai_client.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
            response_format=StructuredAnswer,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = span.attributes or {}
    assert span.name == "openai.chat"
    assert (
        attributes.get("gen_ai.request.base_url") == "https://api.openai.com/v1/"
    )
    assert attributes.get("llm.is_streaming") is False
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert input_messages[0]["role"] == "user"

    assert span.status.status_code == StatusCode.ERROR
    assert (span.status.description or "").startswith("Error code: 401")
    events = span.events
    assert len(events) == 1
    event = events[0]
    event_attributes = event.attributes or {}
    assert event.name == "exception"
    assert event_attributes["exception.type"] == "openai.AuthenticationError"
    assert cast(str, event_attributes["exception.message"]).startswith("Error code: 401")
    assert (
        "Traceback (most recent call last):" in cast(str, event_attributes["exception.stacktrace"])
    )
    assert "openai.AuthenticationError" in cast(str, event_attributes["exception.stacktrace"])
    assert "invalid_api_key" in cast(str, event_attributes["exception.stacktrace"])
    assert attributes.get("error.type") == "AuthenticationError"


@pytest.mark.asyncio
async def test_async_parsed_completion_exception(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    async_openai_client.api_key = "invalid"
    with pytest.raises(AuthenticationError):
        _ = await async_openai_client.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
            response_format=StructuredAnswer,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = span.attributes or {}
    assert span.name == "openai.chat"
    assert (
        attributes.get("gen_ai.request.base_url") == "https://api.openai.com/v1/"
    )
    assert attributes.get("llm.is_streaming") is False
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert input_messages[0]["role"] == "user"

    assert span.status.status_code == StatusCode.ERROR
    assert (span.status.description or "").startswith("Error code: 401")
    events = span.events
    assert len(events) == 1
    event = events[0]
    event_attributes = event.attributes or {}
    assert event.name == "exception"
    assert event_attributes["exception.type"] == "openai.AuthenticationError"
    assert cast(str, event_attributes["exception.message"]).startswith("Error code: 401")
    assert (
        "Traceback (most recent call last):" in cast(str, event_attributes["exception.stacktrace"])
    )
    assert "openai.AuthenticationError" in cast(str, event_attributes["exception.stacktrace"])
    assert "invalid_api_key" in cast(str, event_attributes["exception.stacktrace"])
    assert attributes.get("error.type") == "AuthenticationError"

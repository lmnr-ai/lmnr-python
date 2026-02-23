import json
import pytest
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.sdk.trace import Span
from opentelemetry.trace import StatusCode
from pydantic import BaseModel


class StructuredAnswer(BaseModel):
    rating: int
    joke: str


@pytest.mark.vcr
def test_parsed_completion(instrument_legacy, span_exporter, openai_client):
    openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    output_messages = json.loads(open_ai_span.attributes["gen_ai.output.messages"])
    assert output_messages[0]["message"]["content"]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    input_messages = json.loads(open_ai_span.attributes["gen_ai.input.messages"])
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGC1gNoe1Zyq9yZicdhLc85lmt2Ep"
    )

    assert json.loads(
        open_ai_span.attributes.get("gen_ai.request.structured_output_schema")
    ) == StructuredAnswer.model_json_schema() | {"additionalProperties": False}


@pytest.mark.vcr
def test_parsed_refused_completion(instrument_legacy, span_exporter, openai_client):
    openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    output_messages = json.loads(open_ai_span.attributes["gen_ai.output.messages"])
    assert output_messages[0]["message"].get("content") is None
    assert output_messages[0]["message"]["refusal"] == "I'm very sorry, but I can't assist with that request."
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGky8KFDbg6f5fF4qLtsBredIjZZh"
    )
    assert json.loads(
        open_ai_span.attributes.get("gen_ai.request.structured_output_schema")
    ) == StructuredAnswer.model_json_schema() | {"additionalProperties": False}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_parsed_completion(
    instrument_legacy, span_exporter, async_openai_client
):
    await async_openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    output_messages = json.loads(open_ai_span.attributes["gen_ai.output.messages"])
    assert output_messages[0]["message"]["content"]
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    input_messages = json.loads(open_ai_span.attributes["gen_ai.input.messages"])
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGC1iysV7rZ0qZ510vbeKVTNxSOHB"
    )
    assert json.loads(
        open_ai_span.attributes.get("gen_ai.request.structured_output_schema")
    ) == StructuredAnswer.model_json_schema() | {"additionalProperties": False}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_parsed_refused_completion(
    instrument_legacy, span_exporter, async_openai_client
):
    await async_openai_client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Best ways to make a bomb"}],
        response_format=StructuredAnswer,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    output_messages = json.loads(open_ai_span.attributes["gen_ai.output.messages"])
    assert output_messages[0]["message"].get("content") is None
    assert output_messages[0]["message"]["refusal"] == "I'm very sorry, but I can't assist with that request."
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-AGkyFJGzZPUGAAEDJJuOS3idKvD3G"
    )
    assert json.loads(
        open_ai_span.attributes.get("gen_ai.request.structured_output_schema")
    ) == StructuredAnswer.model_json_schema() | {"additionalProperties": False}


def test_parsed_completion_exception(instrument_legacy, span_exporter, openai_client):
    openai_client.api_key = "invalid"
    with pytest.raises(Exception):
        openai_client.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
            response_format=StructuredAnswer,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span: Span = spans[0]
    assert span.name == "openai.chat"
    assert (
        span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    input_messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert input_messages[0]["role"] == "user"

    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description.startswith("Error code: 401")
    events = span.events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "openai.AuthenticationError"
    assert event.attributes["exception.message"].startswith("Error code: 401")
    assert (
        "Traceback (most recent call last):" in event.attributes["exception.stacktrace"]
    )
    assert "openai.AuthenticationError" in event.attributes["exception.stacktrace"]
    assert "invalid_api_key" in event.attributes["exception.stacktrace"]
    assert span.attributes.get("error.type") == "AuthenticationError"


@pytest.mark.asyncio
async def test_async_parsed_completion_exception(
    instrument_legacy, span_exporter, async_openai_client
):
    async_openai_client.api_key = "invalid"
    with pytest.raises(Exception):
        await async_openai_client.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
            response_format=StructuredAnswer,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span: Span = spans[0]
    assert span.name == "openai.chat"
    assert (
        span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    input_messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert input_messages[0]["role"] == "user"

    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description.startswith("Error code: 401")
    events = span.events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "openai.AuthenticationError"
    assert event.attributes["exception.message"].startswith("Error code: 401")
    assert (
        "Traceback (most recent call last):" in event.attributes["exception.stacktrace"]
    )
    assert "openai.AuthenticationError" in event.attributes["exception.stacktrace"]
    assert "invalid_api_key" in event.attributes["exception.stacktrace"]
    assert span.attributes.get("error.type") == "AuthenticationError"

import json
import os

import pytest
from openrouter import OpenRouter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

MODEL = "openai/gpt-4o-mini"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _client() -> OpenRouter:
    return OpenRouter(api_key=os.environ.get("OPENROUTER_API_KEY", "test-key"))


def _assert_usage(span):
    assert span.attributes["gen_ai.usage.input_tokens"] > 0
    assert span.attributes["gen_ai.usage.output_tokens"] > 0
    assert span.attributes["llm.usage.total_tokens"] > 0
    assert span.attributes["gen_ai.usage.cost"] > 0
    assert span.attributes["gen_ai.usage.input_cost"] > 0
    assert span.attributes["gen_ai.usage.output_cost"] > 0


@pytest.mark.vcr
def test_openrouter_chat(span_exporter: InMemorySpanExporter):
    response = _client().chat.send(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=20,
        temperature=0,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openrouter.chat"
    assert span.attributes["lmnr.span.type"] == "LLM"
    assert span.attributes["gen_ai.system"] == "openrouter"
    assert span.attributes["gen_ai.request.model"] == MODEL
    assert span.attributes["gen_ai.request.max_tokens"] == 20
    assert span.attributes["gen_ai.request.temperature"] == 0
    assert span.attributes["gen_ai.response.model"] == MODEL
    assert span.attributes["gen_ai.response.id"] == response.id
    _assert_usage(span)

    input_messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert input_messages == [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    output_messages = json.loads(span.attributes["gen_ai.output.messages"])
    assert output_messages[0]["message"]["role"] == "assistant"
    assert "Paris" in output_messages[0]["message"]["content"]


@pytest.mark.vcr
def test_openrouter_chat_stream(span_exporter: InMemorySpanExporter):
    stream = _client().chat.send(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=20,
        stream=True,
        stream_options={"include_usage": True},
    )
    content = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in stream
        if chunk.choices
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openrouter.chat"
    assert span.attributes["llm.is_streaming"] is True
    assert span.attributes["gen_ai.response.model"] == MODEL
    _assert_usage(span)

    output_messages = json.loads(span.attributes["gen_ai.output.messages"])
    assert output_messages[0]["message"]["content"] == content
    assert output_messages[0]["finish_reason"] is not None


@pytest.mark.vcr
def test_openrouter_chat_stream_tool_calls(span_exporter: InMemorySpanExporter):
    stream = _client().chat.send(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[WEATHER_TOOL],
        stream=True,
    )
    for _ in stream:
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert json.loads(span.attributes["gen_ai.tool.definitions"]) == [WEATHER_TOOL]

    output_messages = json.loads(span.attributes["gen_ai.output.messages"])
    tool_calls = output_messages[0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"]
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert output_messages[0]["finish_reason"] == "tool_calls"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openrouter_chat_async(span_exporter: InMemorySpanExporter):
    response = await _client().chat.send_async(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=20,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openrouter.chat"
    assert span.attributes["gen_ai.response.id"] == response.id
    _assert_usage(span)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openrouter_chat_async_stream(span_exporter: InMemorySpanExporter):
    stream = await _client().chat.send_async(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=20,
        stream=True,
        stream_options={"include_usage": True},
    )
    content = ""
    async for chunk in stream:
        if chunk.choices:
            content += chunk.choices[0].delta.content or ""

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openrouter.chat"
    _assert_usage(span)
    output_messages = json.loads(span.attributes["gen_ai.output.messages"])
    assert output_messages[0]["message"]["content"] == content


@pytest.mark.vcr
def test_openrouter_responses(span_exporter: InMemorySpanExporter):
    response = _client().responses.send(
        model=MODEL,
        instructions="Answer in one word.",
        input="What is the capital of France?",
        max_output_tokens=20,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openrouter.responses"
    assert span.attributes["lmnr.span.type"] == "LLM"
    assert span.attributes["gen_ai.system"] == "openrouter"
    assert span.attributes["gen_ai.request.model"] == MODEL
    assert span.attributes["gen_ai.request.max_tokens"] == 20
    assert span.attributes["gen_ai.response.id"] == response.id
    _assert_usage(span)

    input_messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert input_messages == [
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    output = json.loads(span.attributes["gen_ai.output.messages"])
    assert output[0]["type"] == "message"
    assert "Paris" in output[0]["content"][0]["text"]


@pytest.mark.vcr
def test_openrouter_responses_stream(span_exporter: InMemorySpanExporter):
    stream = _client().responses.send(
        model=MODEL,
        input="What is the capital of France?",
        max_output_tokens=20,
        stream=True,
    )
    events = list(stream)
    assert events[-1].type == "response.completed"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openrouter.responses"
    assert span.attributes["llm.is_streaming"] is True
    assert span.attributes["gen_ai.response.id"] == events[-1].response.id
    _assert_usage(span)
    output = json.loads(span.attributes["gen_ai.output.messages"])
    assert "Paris" in output[0]["content"][0]["text"]


@pytest.mark.vcr
def test_openrouter_chat_error(span_exporter: InMemorySpanExporter):
    with pytest.raises(Exception):
        _client().chat.send(
            model="openai/this-model-does-not-exist",
            messages=[{"role": "user", "content": "Hello"}],
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openrouter.chat"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"]
    assert span.attributes["gen_ai.request.model"] == "openai/this-model-does-not-exist"

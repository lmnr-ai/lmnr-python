import json
import os

import pytest
from openrouter import OpenRouter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

MODEL = "openai/gpt-4o-mini"
EMBEDDINGS_MODEL = "openai/text-embedding-3-small"

CAPITAL_SCHEMA = {
    "type": "object",
    "properties": {"capital": {"type": "string"}},
    "required": ["capital"],
    "additionalProperties": False,
}

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
    assert span.attributes["lmnr.span.instrumentation_scope.name"] == "openrouter"
    assert span.attributes["lmnr.span.instrumentation_scope.version"]
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
def test_openrouter_chat_request_attributes(span_exporter: InMemorySpanExporter):
    _client().chat.send(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=30,
        frequency_penalty=0.5,
        presence_penalty=0.3,
        reasoning_effort="low",
        user="user-123",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "capital",
                "strict": True,
                "schema": CAPITAL_SCHEMA,
            },
        },
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.frequency_penalty"] == 0.5
    assert span.attributes["gen_ai.request.presence_penalty"] == 0.3
    assert span.attributes["gen_ai.request.reasoning_effort"] == "low"
    assert span.attributes["llm.user"] == "user-123"
    schema = json.loads(span.attributes["gen_ai.request.structured_output_schema"])
    assert schema == CAPITAL_SCHEMA


@pytest.mark.vcr
def test_openrouter_responses_request_attributes(span_exporter: InMemorySpanExporter):
    _client().responses.send(
        model=MODEL,
        input="What is the capital of France?",
        max_output_tokens=50,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        user="user-123",
        reasoning={"effort": "low"},
        # Speakeasy's typed dicts spell the aliased fields with a trailing
        # underscore; the models dump them as `format` / `schema`.
        text={
            "format_": {
                "type": "json_schema",
                "name": "capital",
                "schema_": CAPITAL_SCHEMA,
            }
        },
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.frequency_penalty"] == 0.2
    assert span.attributes["gen_ai.request.presence_penalty"] == 0.1
    assert span.attributes["gen_ai.request.reasoning_effort"] == "low"
    assert span.attributes["llm.user"] == "user-123"
    schema = json.loads(span.attributes["gen_ai.request.structured_output_schema"])
    assert schema == CAPITAL_SCHEMA


@pytest.mark.vcr
def test_openrouter_chat_no_trace_content(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LMNR_TRACE_CONTENT", "false")
    _client().chat.send(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        tools=[WEATHER_TOOL],
        max_tokens=20,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.output.messages" not in span.attributes
    assert "gen_ai.tool.definitions" not in span.attributes
    assert span.attributes["gen_ai.request.model"] == MODEL
    _assert_usage(span)


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
def test_openrouter_chat_stream_closed_unread(span_exporter: InMemorySpanExporter):
    with _client().chat.send(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=20,
        stream=True,
    ):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "openrouter.chat"
    assert spans[0].attributes["gen_ai.request.model"] == MODEL
    assert "gen_ai.output.messages" not in spans[0].attributes


@pytest.mark.vcr
def test_openrouter_chat_stream_close_error(span_exporter: InMemorySpanExporter):
    stream = _client().chat.send(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=20,
        stream=True,
    )

    def failing_close():
        raise RuntimeError("close failed")

    stream.response.close = failing_close
    with pytest.raises(RuntimeError):
        stream.close()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "openrouter.chat"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openrouter_chat_async_stream_closed_unread(
    span_exporter: InMemorySpanExporter,
):
    stream = await _client().chat.send_async(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=20,
        stream=True,
    )
    async with stream:
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "openrouter.chat"
    assert "gen_ai.output.messages" not in spans[0].attributes


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
def test_openrouter_responses_incomplete(span_exporter: InMemorySpanExporter):
    _client().responses.send(
        model=MODEL,
        input="Write a 500 word essay about the history of France.",
        max_output_tokens=16,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openrouter.responses"
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "max_output_tokens"
    assert span.attributes["error.type"] == "incomplete"


@pytest.mark.vcr
def test_openrouter_embeddings(span_exporter: InMemorySpanExporter):
    response = _client().embeddings.generate(
        model=EMBEDDINGS_MODEL,
        input=["hello world", "bonjour"],
        user="user-123",
        # Keeps the recorded cassette small.
        dimensions=8,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openrouter.embeddings"
    assert span.attributes["lmnr.span.type"] == "LLM"
    assert span.attributes["gen_ai.system"] == "openrouter"
    assert span.attributes["gen_ai.request.model"] == EMBEDDINGS_MODEL
    assert span.attributes["llm.user"] == "user-123"
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.response.model"]
    assert span.attributes["gen_ai.usage.input_tokens"] > 0
    assert span.attributes["llm.usage.total_tokens"] > 0
    assert span.attributes["gen_ai.usage.cost"] > 0
    assert span.attributes["gen_ai.usage.input_cost"] > 0

    input_messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert input_messages == [{"content": "hello world"}, {"content": "bonjour"}]


@pytest.mark.vcr
def test_openrouter_embeddings_token_ids(span_exporter: InMemorySpanExporter):
    # A flat list of token ids is one document, not a batch of them.
    token_ids = [15339, 1917]
    _client().embeddings.generate(
        model=EMBEDDINGS_MODEL,
        input=token_ids,
        dimensions=8,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    input_messages = json.loads(spans[0].attributes["gen_ai.input.messages"])
    assert input_messages == [{"content": token_ids}]


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

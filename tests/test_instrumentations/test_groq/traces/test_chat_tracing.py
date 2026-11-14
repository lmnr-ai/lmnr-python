from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import pytest
from groq import APIError, AsyncGroq, Groq, RateLimitError
from groq.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

with open(
    Path(__file__).parent.parent.joinpath("data/logo.jpg"),
    "rb",
) as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")


image_content_block: ChatCompletionContentPartImageParam = {
    "type": "image_url",
    "image_url": {
        "url": f"data:image/jpeg;base64,{image_base64}",
        "detail": "high",
    },
}

JOKE_MESSAGES: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "Tell me a joke about opentelemetry"}
]

LEGACY_KEYS = ("gen_ai.prompt.", "gen_ai.completion.")


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _attr_json(span: ReadableSpan, key: str) -> Any:
    return json.loads(cast(str, _attrs(span)[key]))


def _only_span(span_exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["groq.chat"]
    return spans[0]


def _assert_no_legacy_attributes(span: ReadableSpan):
    assert not [key for key in _attrs(span) if key.startswith(LEGACY_KEYS)]


@pytest.mark.vcr
def test_chat_legacy(
    instrument_legacy: MagicMock, groq_client: Groq, span_exporter: InMemorySpanExporter
):
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=JOKE_MESSAGES,
    )

    groq_span = _only_span(span_exporter)
    attrs = _attrs(groq_span)
    assert _attr_json(groq_span, "gen_ai.input.messages") == JOKE_MESSAGES
    output = _attr_json(groq_span, "gen_ai.output.messages")
    assert len(output) == 1
    assert output[0]["finish_reason"] == response.choices[0].finish_reason
    assert output[0]["message"]["role"] == "assistant"
    assert output[0]["message"]["content"] == response.choices[0].message.content
    _assert_no_legacy_attributes(groq_span)
    assert "gen_ai.tool.definitions" not in attrs
    assert attrs.get("llm.is_streaming") is False
    assert cast(int, attrs.get("gen_ai.usage.input_tokens")) > 0
    assert cast(int, attrs.get("gen_ai.usage.output_tokens")) > 0
    assert cast(int, attrs.get("llm.usage.total_tokens")) > 0
    assert attrs.get("gen_ai.response.id") == "chatcmpl-645691ff-34af-4d0f-a1c1-fe888f8685cc"


@pytest.mark.vcr
def test_chat_legacy_image(
    instrument_legacy: MagicMock,
    groq_client: Groq,
    span_exporter: InMemorySpanExporter,
):
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is depicted in this image?"},
                image_content_block,
            ],
        }
    ]
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-guard-4-12b",
        messages=messages,
    )

    groq_span = _only_span(span_exporter)
    attrs = _attrs(groq_span)
    assert _attr_json(groq_span, "gen_ai.input.messages") == messages
    output = _attr_json(groq_span, "gen_ai.output.messages")
    assert output[0]["message"]["content"] == response.choices[0].message.content
    assert output[0]["message"]["role"] == "assistant"
    _assert_no_legacy_attributes(groq_span)
    assert attrs.get("llm.is_streaming") is False
    assert cast(int, attrs.get("gen_ai.usage.input_tokens")) > 0
    assert cast(int, attrs.get("gen_ai.usage.output_tokens")) > 0
    assert cast(int, attrs.get("llm.usage.total_tokens")) > 0
    assert attrs.get("gen_ai.response.id") == "chatcmpl-741dc0bb-6df1-4e88-9d35-920d0dddc3c0"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chat_legacy(
    instrument_legacy: MagicMock,
    async_groq_client: AsyncGroq,
    span_exporter: InMemorySpanExporter,
):
    response = await async_groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=JOKE_MESSAGES,
    )

    groq_span = _only_span(span_exporter)
    attrs = _attrs(groq_span)
    assert _attr_json(groq_span, "gen_ai.input.messages") == JOKE_MESSAGES
    output = _attr_json(groq_span, "gen_ai.output.messages")
    assert output[0]["message"]["content"] == response.choices[0].message.content
    _assert_no_legacy_attributes(groq_span)
    assert attrs.get("llm.is_streaming") is False
    assert cast(int, attrs.get("gen_ai.usage.input_tokens")) > 0
    assert cast(int, attrs.get("gen_ai.usage.output_tokens")) > 0
    assert cast(int, attrs.get("llm.usage.total_tokens")) > 0
    assert attrs.get("gen_ai.response.id") == "chatcmpl-ec0a74e9-df7f-4e91-aa09-e9618451f5c9"


@pytest.mark.vcr
def test_chat_streaming_legacy(
    instrument_legacy: MagicMock,
    groq_client: Groq,
    span_exporter: InMemorySpanExporter,
):
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=JOKE_MESSAGES,
        stream=True,
    )

    content = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content

    groq_span = _only_span(span_exporter)
    attrs = _attrs(groq_span)
    assert _attr_json(groq_span, "gen_ai.input.messages") == JOKE_MESSAGES
    assert _attr_json(groq_span, "gen_ai.output.messages") == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }
    ]
    _assert_no_legacy_attributes(groq_span)
    assert attrs.get("llm.is_streaming") is True
    assert attrs.get("gen_ai.usage.input_tokens") == 18
    assert attrs.get("gen_ai.usage.output_tokens") == 73
    assert attrs.get("llm.usage.total_tokens") == 91
    assert attrs.get("gen_ai.response.id") == "chatcmpl-532a6640-2e18-4a4c-8406-8437a857ce48"
    assert attrs.get("gen_ai.response.model") == "llama3-8b-8192"


# The remaining tests serve canned bodies (shaped like groq's typed
# `ChatCompletion` / `ChatCompletionChunk`) through an `httpx.MockTransport`.

MODEL = "openai/gpt-oss-20b"

WEATHER_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
}

USAGE = {
    "prompt_tokens": 120,
    "completion_tokens": 40,
    "total_tokens": 160,
    "prompt_tokens_details": {"cached_tokens": 64},
    "completion_tokens_details": {"reasoning_tokens": 25},
}

TOOL_CALL_COMPLETION = {
    "id": "chatcmpl-tool",
    "object": "chat.completion",
    "created": 1762400000,
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "logprobs": None,
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning": "The user wants the weather in Paris.",
                "tool_calls": [TOOL_CALL],
            },
        }
    ],
    "usage": USAGE,
    "x_groq": {"id": "req_1"},
}


def _chunk(delta: dict[str, Any], finish_reason: str | None = None, **extra: Any):
    return {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1762400000,
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        **extra,
    }


TOOL_CALL_STREAM = [
    _chunk({"role": "assistant", "content": None}),
    _chunk({"reasoning": "The user wants "}),
    _chunk({"reasoning": "the weather in Paris."}),
    _chunk({"tool_calls": [{"index": 0, **TOOL_CALL}]}),
    _chunk({}, "tool_calls", x_groq={"id": "req_2", "usage": USAGE}),
]

TOOL_MESSAGES: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "You are a weather bot."},
    {"role": "user", "content": "What's the weather in Paris?"},
]

Handler = Callable[[httpx.Request], httpx.Response]


def _json_handler(body: dict[str, Any]) -> Handler:
    return lambda _request: httpx.Response(200, json=body)


def _sse_handler(chunks: list[dict[str, Any]]) -> Handler:
    payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return lambda _request: httpx.Response(
        200, content=payload.encode(), headers={"content-type": "text/event-stream"}
    )


def _client(handler: Handler) -> Groq:
    return Groq(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _async_client(handler: Handler) -> AsyncGroq:
    return AsyncGroq(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _assert_tool_call_span(span: ReadableSpan, streaming: bool):
    attrs = _attrs(span)
    assert _attr_json(span, "gen_ai.input.messages") == TOOL_MESSAGES
    assert _attr_json(span, "gen_ai.tool.definitions") == [WEATHER_TOOL]
    output = _attr_json(span, "gen_ai.output.messages")
    assert len(output) == 1
    assert output[0]["finish_reason"] == "tool_calls"
    message = output[0]["message"]
    assert message["role"] == "assistant"
    assert message["content"] is None
    assert message["reasoning"] == "The user wants the weather in Paris."
    assert message["tool_calls"] == [TOOL_CALL]
    _assert_no_legacy_attributes(span)

    assert attrs["llm.is_streaming"] is streaming
    assert attrs["gen_ai.request.model"] == MODEL
    assert attrs["gen_ai.request.max_tokens"] == 512
    assert attrs["gen_ai.request.reasoning.level"] == "low"
    assert attrs["gen_ai.response.model"] == MODEL
    assert attrs["gen_ai.usage.input_tokens"] == 120
    assert attrs["gen_ai.usage.output_tokens"] == 40
    assert attrs["llm.usage.total_tokens"] == 160
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 64
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 25


def test_chat_tool_calls(instrument_legacy: MagicMock, span_exporter: InMemorySpanExporter):
    _ = _client(_json_handler(TOOL_CALL_COMPLETION)).chat.completions.create(
        model=MODEL,
        messages=TOOL_MESSAGES,
        tools=[WEATHER_TOOL],
        max_completion_tokens=512,
        reasoning_effort="low",
    )

    span = _only_span(span_exporter)
    _assert_tool_call_span(span, streaming=False)
    assert _attrs(span)["gen_ai.response.id"] == "chatcmpl-tool"


def test_chat_stream_tool_calls(instrument_legacy: MagicMock, span_exporter: InMemorySpanExporter):
    stream = _client(_sse_handler(TOOL_CALL_STREAM)).chat.completions.create(
        model=MODEL,
        messages=TOOL_MESSAGES,
        tools=[WEATHER_TOOL],
        max_completion_tokens=512,
        reasoning_effort="low",
        stream=True,
    )
    for _ in stream:
        pass

    span = _only_span(span_exporter)
    _assert_tool_call_span(span, streaming=True)
    assert _attrs(span)["gen_ai.response.id"] == "chatcmpl-stream"


@pytest.mark.asyncio
async def test_async_chat_stream_tool_calls(
    instrument_legacy: MagicMock, span_exporter: InMemorySpanExporter
):
    stream = await _async_client(_sse_handler(TOOL_CALL_STREAM)).chat.completions.create(
        model=MODEL,
        messages=TOOL_MESSAGES,
        tools=[WEATHER_TOOL],
        max_completion_tokens=512,
        reasoning_effort="low",
        stream=True,
    )
    async for _ in stream:
        pass

    _assert_tool_call_span(_only_span(span_exporter), streaming=True)


def test_chat_usage_without_details(
    instrument_legacy: MagicMock, span_exporter: InMemorySpanExporter
):
    body = {
        **TOOL_CALL_COMPLETION,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    _ = _client(_json_handler(body)).chat.completions.create(model=MODEL, messages=TOOL_MESSAGES)

    attrs = _attrs(_only_span(span_exporter))
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert "gen_ai.usage.cache_read.input_tokens" not in attrs
    assert "gen_ai.usage.reasoning.output_tokens" not in attrs


def test_chat_content_not_traced(
    instrument_legacy: MagicMock,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LMNR_TRACE_CONTENT", "false")
    _ = _client(_json_handler(TOOL_CALL_COMPLETION)).chat.completions.create(
        model=MODEL, messages=TOOL_MESSAGES, tools=[WEATHER_TOOL]
    )

    attrs = _attrs(_only_span(span_exporter))
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs
    assert "gen_ai.tool.definitions" not in attrs
    assert attrs["gen_ai.usage.input_tokens"] == 120


def _error_handler(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(429, json={"error": {"message": "rate limited"}})


def test_chat_api_error_ends_span_with_error(
    instrument_legacy: MagicMock, span_exporter: InMemorySpanExporter
):
    client = Groq(
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_error_handler)),
    )
    with pytest.raises(RateLimitError):
        client.chat.completions.create(model=MODEL, messages=TOOL_MESSAGES)

    span = _only_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert _attrs(span)["error.type"] == "RateLimitError"
    assert [event.name for event in span.events] == ["exception"]


@pytest.mark.asyncio
async def test_async_chat_api_error_ends_span_with_error(
    instrument_legacy: MagicMock, span_exporter: InMemorySpanExporter
):
    client = AsyncGroq(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_error_handler)),
    )
    with pytest.raises(RateLimitError):
        await client.chat.completions.create(model=MODEL, messages=TOOL_MESSAGES)

    span = _only_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert _attrs(span)["error.type"] == "RateLimitError"


def _stream_error_handler(_request: httpx.Request) -> httpx.Response:
    chunk = _chunk({"role": "assistant", "content": "Hel"})
    error = {"error": {"message": "stream failed", "type": "server_error"}}
    payload = f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(error)}\n\n"
    return httpx.Response(
        200, content=payload.encode(), headers={"content-type": "text/event-stream"}
    )


def test_chat_stream_error_ends_span(
    instrument_legacy: MagicMock, span_exporter: InMemorySpanExporter
):
    stream = _client(_stream_error_handler).chat.completions.create(
        model=MODEL, messages=TOOL_MESSAGES, stream=True
    )
    with pytest.raises(APIError):
        for _ in stream:
            pass

    span = _only_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert _attrs(span)["error.type"] == "APIError"
    assert [event.name for event in span.events] == ["exception"]


@pytest.mark.asyncio
async def test_async_chat_stream_error_ends_span(
    instrument_legacy: MagicMock, span_exporter: InMemorySpanExporter
):
    stream = await _async_client(_stream_error_handler).chat.completions.create(
        model=MODEL, messages=TOOL_MESSAGES, stream=True
    )
    with pytest.raises(APIError):
        async for _ in stream:
            pass

    span = _only_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert _attrs(span)["error.type"] == "APIError"
    assert [event.name for event in span.events] == ["exception"]


def _two_chunk_handler(_request: httpx.Request) -> httpx.Response:
    chunks = [
        _chunk({"role": "assistant", "content": "Hel"}),
        _chunk({"content": "lo"}, finish_reason="stop"),
    ]
    payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(
        200, content=payload.encode(), headers={"content-type": "text/event-stream"}
    )


def test_chat_stream_early_close_ends_span(
    instrument_legacy: MagicMock, span_exporter: InMemorySpanExporter
):
    stream = _client(_two_chunk_handler).chat.completions.create(
        model=MODEL, messages=TOOL_MESSAGES, stream=True
    )
    for _ in stream:
        break
    stream.close()  # type: ignore[attr-defined]

    span = _only_span(span_exporter)
    assert span.status.status_code == StatusCode.OK


@pytest.mark.asyncio
async def test_async_chat_stream_early_close_ends_span(
    instrument_legacy: MagicMock, span_exporter: InMemorySpanExporter
):
    stream = await _async_client(_two_chunk_handler).chat.completions.create(
        model=MODEL, messages=TOOL_MESSAGES, stream=True
    )
    async for _ in stream:
        break
    await stream.aclose()  # type: ignore[attr-defined]

    span = _only_span(span_exporter)
    assert span.status.status_code == StatusCode.OK

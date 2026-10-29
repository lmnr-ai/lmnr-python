import json
from unittest.mock import MagicMock

import httpx
import pytest
from groq import AsyncGroq, Groq
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def _chunk(delta: dict, finish_reason: str | None = None) -> str:
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "llama-3.3-70b-versatile",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(body)}\n\n"


def _tool(index: int, **fields) -> dict:
    return {"tool_calls": [{"index": index, **fields}]}


# Two parallel tool calls, the second one's arguments split over two chunks.
SSE_BODY = (
    _chunk({"role": "assistant", "content": ""})
    + _chunk(
        _tool(
            0,
            id="call_a",
            type="function",
            function={"name": "get_weather", "arguments": '{"city": "Paris"}'},
        )
    )
    + _chunk(
        _tool(
            1,
            id="call_b",
            type="function",
            function={"name": "get_time", "arguments": '{"tz":'},
        )
    )
    + _chunk(_tool(1, function={"arguments": ' "CET"}'}))
    + _chunk({}, finish_reason="tool_calls")
    + "data: [DONE]\n\n"
)

MESSAGES = [{"role": "user", "content": "Weather in Paris and the time in CET?"}]


def _handler(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=SSE_BODY.encode(),
    )


def _assert_tool_calls(span_exporter: InMemorySpanExporter) -> None:
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["groq.chat"]
    attributes = spans[0].attributes or {}
    assert attributes.get("gen_ai.completion.0.finish_reason") == "tool_calls"
    assert attributes.get("gen_ai.completion.0.tool_calls.0.id") == "call_a"
    assert attributes.get("gen_ai.completion.0.tool_calls.0.name") == "get_weather"
    assert attributes.get("gen_ai.completion.0.tool_calls.0.arguments") == '{"city": "Paris"}'
    assert attributes.get("gen_ai.completion.0.tool_calls.1.id") == "call_b"
    assert attributes.get("gen_ai.completion.0.tool_calls.1.name") == "get_time"
    assert attributes.get("gen_ai.completion.0.tool_calls.1.arguments") == '{"tz": "CET"}'


def test_chat_streaming_records_tool_calls(
    instrument_legacy: MagicMock,
    span_exporter: InMemorySpanExporter,
):
    client = Groq(
        api_key="api-key", http_client=httpx.Client(transport=httpx.MockTransport(_handler))
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=MESSAGES, stream=True
    )
    for _ in response:
        pass

    _assert_tool_calls(span_exporter)


@pytest.mark.asyncio
async def test_async_chat_streaming_records_tool_calls(
    instrument_legacy: MagicMock,
    span_exporter: InMemorySpanExporter,
):
    client = AsyncGroq(
        api_key="api-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )
    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=MESSAGES, stream=True
    )
    async for _ in response:
        pass

    _assert_tool_calls(span_exporter)

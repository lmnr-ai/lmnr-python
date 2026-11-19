import json
from typing import Any, cast

import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic import (
    AnthropicInstrumentor,
)


def _sse(*events: tuple[str, dict[str, Any]]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def _handler(_request: httpx2.Request) -> httpx2.Response:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hel"},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "lo"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return httpx2.Response(
        200, content=_sse(*events), headers={"content-type": "text/event-stream"}
    )


def _assert_finished_ok(span_exporter: InMemorySpanExporter):
    spans = span_exporter.get_finished_spans()
    assert [s.name for s in spans] == ["anthropic.chat"]
    assert spans[0].status.status_code == StatusCode.OK


def test_stream_closed_early_ends_span(
    instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    client = Anthropic(
        api_key="test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(_handler)),
    )
    stream = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    next(iter(stream))
    stream.close()
    _assert_finished_ok(span_exporter)


@pytest.mark.asyncio
async def test_async_stream_closed_early_ends_span(
    instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    client = AsyncAnthropic(
        api_key="test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(_handler)),
    )
    stream = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    await stream.__anext__()
    await cast("Any", stream).aclose()  # wrapped stream is an async generator at runtime
    _assert_finished_ok(span_exporter)

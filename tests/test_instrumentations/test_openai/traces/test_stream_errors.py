import json
from collections.abc import Mapping

import httpx2
import pytest
from openai import APIError, AsyncOpenAI, OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import (
    OpenAIInstrumentor,
)

ERROR = {"error": {"message": "stream failed", "type": "server_error"}}


def _sse(*events: Mapping[str, object]) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()


def _chat_handler(_request: httpx2.Request) -> httpx2.Response:
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}],
    }
    return httpx2.Response(
        200,
        content=_sse(chunk, ERROR),
        headers={"content-type": "text/event-stream"},
    )


def _completion_handler(_request: httpx2.Request) -> httpx2.Response:
    chunk = {
        "id": "cmpl-1",
        "object": "text_completion",
        "created": 1,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [{"index": 0, "text": "Hel", "finish_reason": None}],
    }
    return httpx2.Response(
        200,
        content=_sse(chunk, ERROR),
        headers={"content-type": "text/event-stream"},
    )


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


def _assert_errored(span_exporter: InMemorySpanExporter, name: str):
    spans = span_exporter.get_finished_spans()
    assert [s.name for s in spans] == [name]
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert (span.attributes or {})["error.type"] == "APIError"
    assert [e.name for e in span.events if e.name == "exception"] == ["exception"]


def test_chat_stream_error_records_error(
    instrumentor: OpenAIInstrumentor, span_exporter: InMemorySpanExporter
):
    stream = _client(_chat_handler).chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    with pytest.raises(APIError):
        for _ in stream:
            pass
    _assert_errored(span_exporter, "openai.chat")


@pytest.mark.asyncio
async def test_async_chat_stream_error_records_error(
    instrumentor: OpenAIInstrumentor, span_exporter: InMemorySpanExporter
):
    stream = await _async_client(_chat_handler).chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    with pytest.raises(APIError):
        async for _ in stream:
            pass
    _assert_errored(span_exporter, "openai.chat")


def test_completion_stream_error_records_error(
    instrumentor: OpenAIInstrumentor, span_exporter: InMemorySpanExporter
):
    stream = _client(_completion_handler).completions.create(
        model="gpt-3.5-turbo-instruct", prompt="hi", stream=True
    )
    with pytest.raises(APIError):
        for _ in stream:
            pass
    _assert_errored(span_exporter, "openai.completion")


@pytest.mark.asyncio
async def test_async_completion_stream_error_records_error(
    instrumentor: OpenAIInstrumentor, span_exporter: InMemorySpanExporter
):
    stream = await _async_client(_completion_handler).completions.create(
        model="gpt-3.5-turbo-instruct", prompt="hi", stream=True
    )
    with pytest.raises(APIError):
        async for _ in stream:
            pass
    _assert_errored(span_exporter, "openai.completion")

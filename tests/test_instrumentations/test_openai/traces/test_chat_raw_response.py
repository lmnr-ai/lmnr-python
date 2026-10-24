"""`chat.completions.with_raw_response.create`, the way LangChain's `ChatOpenAI` calls the SDK.

The caller reads the body with `.parse()`, so the instrumented stream (live) and a
debug-replay cache hit both have to come back behind it. Plain streaming cache hits
are covered here too. httpx `MockTransport`, no cassettes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import (
    OpenAIInstrumentor,
    rollout,
)
from lmnr.sdk.debug import replay
from lmnr.sdk.debug.outcome import CacheOutcome

MODEL = "gpt-4.1-nano"
MESSAGES: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Capital of France?"}]


def _completion(text: str, completion_id: str) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }


def _chunk(delta: dict[str, Any], finish_reason: str | None = None, **extra: Any) -> str:
    chunk = {
        "id": "chatcmpl_live",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        **extra,
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _sse(text: str) -> bytes:
    return (
        _chunk({"role": "assistant", "content": ""})
        + _chunk({"content": text})
        + _chunk(
            {},
            "stop",
            usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        )
        + "data: [DONE]\n\n"
    ).encode()


def _transport(requests: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        headers = {"x-request-id": "req_live"}
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200, content=_sse("Paris"), headers={**headers, "content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=_completion("Paris", "chatcmpl_live"), headers=headers)

    return httpx.MockTransport(handler)


def _client(requests: list[httpx.Request]) -> OpenAI:
    return OpenAI(
        api_key="test", max_retries=0, http_client=httpx.Client(transport=_transport(requests))
    )


def _async_client(requests: list[httpx.Request]) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=_transport(requests)),
    )


def _only_span(span_exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["openai.chat"]
    return spans[0]


def _attributes(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _output_text(span: ReadableSpan) -> str:
    output = json.loads(cast(str, _attributes(span)["gen_ai.output.messages"]))
    return output[0]["message"]["content"]


def _text(chunks: Any) -> str:
    return "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)


def test_raw_response_stream_is_instrumented(
    instrumentor: OpenAIInstrumentor, span_exporter: InMemorySpanExporter
):
    requests: list[httpx.Request] = []
    raw = _client(requests).chat.completions.with_raw_response.create(
        model=MODEL, messages=MESSAGES, stream=True
    )

    assert raw.headers["x-request-id"] == "req_live"
    assert _text(raw.parse()) == "Paris"
    span = _only_span(span_exporter)
    attributes = _attributes(span)
    assert attributes["lmnr.span.type"] == "LLM"
    assert attributes["gen_ai.response.id"] == "chatcmpl_live"
    assert attributes["gen_ai.usage.output_tokens"] == 3
    assert _output_text(span) == "Paris"


@pytest.mark.asyncio
async def test_raw_response_stream_is_instrumented_async(
    instrumentor: OpenAIInstrumentor, span_exporter: InMemorySpanExporter
):
    requests: list[httpx.Request] = []
    raw = await _async_client(requests).chat.completions.with_raw_response.create(
        model=MODEL, messages=MESSAGES, stream=True
    )

    assert raw.headers["x-request-id"] == "req_live"
    assert _text([chunk async for chunk in raw.parse()]) == "Paris"
    assert _output_text(_only_span(span_exporter)) == "Paris"


def test_raw_response_without_stream_is_unchanged(
    instrumentor: OpenAIInstrumentor, span_exporter: InMemorySpanExporter
):
    requests: list[httpx.Request] = []
    raw = _client(requests).chat.completions.with_raw_response.create(
        model=MODEL, messages=MESSAGES
    )

    assert type(raw).__name__ == "LegacyAPIResponse"
    assert raw.parse().choices[0].message.content == "Paris"
    assert _output_text(_only_span(span_exporter)) == "Paris"


# --- debug replay -------------------------------------------------------------


@pytest.fixture
def replay_outcome(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[CacheOutcome]]:
    """Pretend this is a debug run with replay configured; each call pops the
    next cache outcome."""
    outcomes: list[CacheOutcome] = []

    def outcome_for(_span: Any) -> CacheOutcome | None:
        return outcomes.pop(0) if outcomes else None

    async def aoutcome_for(span: Any) -> CacheOutcome | None:
        return outcome_for(span)

    monkeypatch.setattr(replay, "replay_enabled", lambda: True)
    monkeypatch.setattr(rollout, "replay_enabled", lambda: True)
    monkeypatch.setattr(rollout, "cache_outcome_for", outcome_for)
    monkeypatch.setattr(rollout, "acache_outcome_for", aoutcome_for)
    yield outcomes


def _hit() -> CacheOutcome:
    return CacheOutcome(
        kind="hit", cached={"type": "raw", "response": _completion("Rome", "chatcmpl_cached")}
    )


def test_replay_hit_with_raw_response(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    replay_outcome.append(_hit())
    requests: list[httpx.Request] = []
    raw = _client(requests).chat.completions.with_raw_response.create(
        model=MODEL, messages=MESSAGES
    )

    completion = raw.parse()
    assert isinstance(completion, ChatCompletion)
    assert completion.choices[0].message.content == "Rome"
    assert dict(raw.headers) == {}
    assert requests == []
    span = _only_span(span_exporter)
    assert _attributes(span)["lmnr.span.type"] == "CACHED"
    assert _output_text(span) == "Rome"


def test_replay_hit_with_raw_response_stream(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    replay_outcome.append(_hit())
    requests: list[httpx.Request] = []
    raw = _client(requests).chat.completions.with_raw_response.create(
        model=MODEL, messages=MESSAGES, stream=True
    )

    assert _text(raw.parse()) == "Rome"
    assert requests == []
    span = _only_span(span_exporter)
    assert _attributes(span)["lmnr.span.type"] == "CACHED"
    assert _output_text(span) == "Rome"


@pytest.mark.asyncio
async def test_replay_hit_with_raw_response_async(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    replay_outcome.append(_hit())
    requests: list[httpx.Request] = []
    raw = await _async_client(requests).chat.completions.with_raw_response.create(
        model=MODEL, messages=MESSAGES
    )

    assert raw.parse().choices[0].message.content == "Rome"
    assert requests == []
    assert _attributes(_only_span(span_exporter))["lmnr.span.type"] == "CACHED"


def test_replay_hit_stream(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    # Not raw-response specific: a cached stream used to be a generator, which
    # `ChatStream` (a wrapt proxy that sets `_span` on what it wraps) rejected.
    replay_outcome.append(_hit())
    requests: list[httpx.Request] = []
    stream = _client(requests).chat.completions.create(model=MODEL, messages=MESSAGES, stream=True)

    assert _text(stream) == "Rome"
    assert requests == []
    span = _only_span(span_exporter)
    assert _attributes(span)["lmnr.span.type"] == "CACHED"
    assert _output_text(span) == "Rome"


@pytest.mark.asyncio
async def test_replay_hit_stream_async(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    replay_outcome.append(_hit())
    requests: list[httpx.Request] = []
    stream = await _async_client(requests).chat.completions.create(
        model=MODEL, messages=MESSAGES, stream=True
    )

    assert _text([chunk async for chunk in stream]) == "Rome"
    assert requests == []
    assert _output_text(_only_span(span_exporter)) == "Rome"

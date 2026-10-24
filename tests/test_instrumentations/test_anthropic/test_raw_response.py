"""`messages.with_raw_response.create`, the way LangChain's `ChatAnthropic` calls the SDK.

The caller reads the body with `.parse()`, so the instrumented stream (live) and a
debug-replay cache hit both have to come back behind it. `MockTransport` (anthropic >= 2
uses `httpx2`), no cassettes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, cast

import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic import (
    AnthropicInstrumentor,
    rollout,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.raw_response import (
    RAW_RESPONSE_HEADER,
    aparse_raw_response,
    as_raw_response,
)
from lmnr.sdk.debug.outcome import CacheOutcome

MODEL = "claude-haiku-4-5"
MESSAGES: list[Any] = [{"role": "user", "content": "Capital of France?"}]


def _message(text: str, message_id: str) -> dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }


def _sse(text: str) -> bytes:
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    **_message("", "msg_live"),
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 12, "output_tokens": 1},
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
                "delta": {"type": "text_delta", "text": text},
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
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def _transport(requests: list[httpx2.Request]) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if json.loads(request.content).get("stream"):
            return httpx2.Response(
                200,
                content=_sse("Paris"),
                headers={"content-type": "text/event-stream", "request-id": "req_live"},
            )
        return httpx2.Response(
            200, json=_message("Paris", "msg_live"), headers={"request-id": "req_live"}
        )

    return httpx2.MockTransport(handler)


def _client(requests: list[httpx2.Request]) -> Anthropic:
    return Anthropic(
        api_key="test", max_retries=0, http_client=httpx2.Client(transport=_transport(requests))
    )


def _async_client(requests: list[httpx2.Request]) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key="test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=_transport(requests)),
    )


def _only_span(span_exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["anthropic.chat"]
    return spans[0]


def _attributes(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _output_text(span: ReadableSpan) -> str:
    output = json.loads(cast(str, _attributes(span)["gen_ai.output.messages"]))
    return output[0]["content"][0]["text"]


def _text(events: Any) -> str:
    return "".join(
        event.delta.text
        for event in events
        if event.type == "content_block_delta" and event.delta.type == "text_delta"
    )


def test_raw_response_stream_is_instrumented(
    instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    requests: list[httpx2.Request] = []
    raw = _client(requests).messages.with_raw_response.create(
        model=MODEL, max_tokens=10, messages=MESSAGES, stream=True
    )

    assert raw.headers["request-id"] == "req_live"
    assert _text(raw.parse()) == "Paris"
    span = _only_span(span_exporter)
    attributes = _attributes(span)
    assert attributes["lmnr.span.type"] == "LLM"
    assert attributes["gen_ai.response.id"] == "msg_live"
    assert attributes["gen_ai.usage.input_tokens"] == 12
    assert attributes["gen_ai.usage.output_tokens"] == 3
    assert _output_text(span) == "Paris"


@pytest.mark.asyncio
async def test_raw_response_stream_is_instrumented_async(
    instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    requests: list[httpx2.Request] = []
    raw = await _async_client(requests).messages.with_raw_response.create(
        model=MODEL, max_tokens=10, messages=MESSAGES, stream=True
    )

    assert raw.headers["request-id"] == "req_live"
    assert _text([event async for event in await raw.parse()]) == "Paris"
    span = _only_span(span_exporter)
    assert _attributes(span)["gen_ai.usage.output_tokens"] == 3
    assert _output_text(span) == "Paris"


def test_raw_response_without_stream_is_unchanged(
    instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    requests: list[httpx2.Request] = []
    raw = _client(requests).messages.with_raw_response.create(
        model=MODEL, max_tokens=10, messages=MESSAGES
    )

    assert type(raw).__name__ == "APIResponse"
    assert raw.parse().content[0].text == "Paris"  # pyright: ignore[reportAttributeAccessIssue]
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

    monkeypatch.setattr(rollout, "replay_enabled", lambda: True)
    monkeypatch.setattr(rollout, "cache_outcome_for", outcome_for)
    monkeypatch.setattr(rollout, "acache_outcome_for", aoutcome_for)
    yield outcomes


def _hit() -> CacheOutcome:
    return CacheOutcome(
        kind="hit", cached={"type": "raw", "response": _message("Rome", "msg_cached")}
    )


def test_replay_hit_with_raw_response(
    instrumentor: AnthropicInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    replay_outcome.append(_hit())
    requests: list[httpx2.Request] = []
    raw = _client(requests).messages.with_raw_response.create(
        model=MODEL, max_tokens=10, messages=MESSAGES
    )

    message = raw.parse()
    assert isinstance(message, Message)
    assert message.content[0].text == "Rome"  # pyright: ignore[reportAttributeAccessIssue]
    assert dict(raw.headers) == {}
    assert requests == []
    span = _only_span(span_exporter)
    assert _attributes(span)["lmnr.span.type"] == "CACHED"
    assert _output_text(span) == "Rome"


def test_replay_hit_with_raw_response_stream(
    instrumentor: AnthropicInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    replay_outcome.append(_hit())
    requests: list[httpx2.Request] = []
    raw = _client(requests).messages.with_raw_response.create(
        model=MODEL, max_tokens=10, messages=MESSAGES, stream=True
    )

    assert _text(raw.parse()) == "Rome"
    assert requests == []
    span = _only_span(span_exporter)
    assert _attributes(span)["lmnr.span.type"] == "CACHED"
    assert _output_text(span) == "Rome"


def test_replay_hit_with_raw_header_from_anthropic_v1(
    instrumentor: AnthropicInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    # anthropic >= 1 sends `raw` where 0.x sends `true`.
    replay_outcome.append(_hit())
    requests: list[httpx2.Request] = []
    raw = _client(requests).messages.create(
        model=MODEL,
        max_tokens=10,
        messages=MESSAGES,
        extra_headers={RAW_RESPONSE_HEADER: "raw"},
    )

    assert cast(Any, raw).parse().content[0].text == "Rome"
    assert requests == []


@pytest.mark.asyncio
async def test_replay_hit_with_raw_header_from_anthropic_v1_async(
    instrumentor: AnthropicInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    # anthropic >= 1's `AsyncAPIResponse.parse()` is a coroutine, so callers await it.
    replay_outcome.append(_hit())
    requests: list[httpx2.Request] = []
    raw = await _async_client(requests).messages.create(
        model=MODEL,
        max_tokens=10,
        messages=MESSAGES,
        extra_headers={RAW_RESPONSE_HEADER: "raw"},
    )

    message = await cast(Any, raw).parse()
    assert message.content[0].text == "Rome"
    assert requests == []


@pytest.mark.asyncio
async def test_replay_hit_with_raw_response_async(
    instrumentor: AnthropicInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    replay_outcome.append(_hit())
    requests: list[httpx2.Request] = []
    raw = await _async_client(requests).messages.with_raw_response.create(
        model=MODEL, max_tokens=10, messages=MESSAGES
    )

    message = await raw.parse()
    assert message.content[0].text == "Rome"  # pyright: ignore[reportAttributeAccessIssue]
    assert requests == []
    assert _attributes(_only_span(span_exporter))["lmnr.span.type"] == "CACHED"


# --- anthropic >= 1 -------------------------------------------------------------


class _AsyncAPIResponse:
    """anthropic >= 1's `AsyncAPIResponse` shape: `parse()` is a coroutine."""

    http_response: httpx2.Response = httpx2.Response(200, headers={"request-id": "req_live"})
    headers: httpx2.Headers = http_response.headers

    def __init__(self, body: object) -> None:
        self._body: object = body

    async def parse(self, *, to: type[Any] | None = None) -> object:
        return self._body


@pytest.mark.asyncio
async def test_async_api_response_is_parsed_up_front():
    # The async wrappers' attribute helpers parse synchronously, so a coroutine
    # `parse()` has to be awaited before they run, stream or not.
    kwargs = {"extra_headers": {RAW_RESPONSE_HEADER: "raw"}}
    response = _AsyncAPIResponse(Message.model_validate(_message("Paris", "msg_live")))

    parsed, raw = await aparse_raw_response(kwargs, response)
    assert raw is response
    assert isinstance(parsed, Message)

    wrapped = as_raw_response(kwargs, parsed, raw, is_async=True)
    assert wrapped.headers["request-id"] == "req_live"
    assert await wrapped.parse() is parsed

"""Responses-API streaming: `responses.create(stream=True)` and `responses.stream()`.

VCR-recorded tests cover the live span; httpx `MockTransport` tests cover what a
cassette can't express (API errors, debug-replay cache HITs / MISSes).
"""

import json
from collections.abc import Iterator
from typing import Any, cast

import httpx
import pytest
from openai import AsyncOpenAI, BadRequestError, OpenAI
from openai.types.responses import FunctionToolParam, Response, ResponseStreamEvent
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import (
    OpenAIInstrumentor,
    rollout,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.rollout import (
    AsyncCachedResponsesEventStream,
    CachedResponsesEventStream,
    OpenAIRolloutWrapper,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.v1 import (
    responses_wrappers,
)
from lmnr.sdk.debug import replay
from lmnr.sdk.debug.outcome import CacheOutcome

TOOLS: list[FunctionToolParam] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }
            },
            "required": ["location"],
        },
        "strict": None,
    }
]


def _only_span(span_exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def _attributes(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _output_messages(attributes: dict[str, Any]) -> list[dict[str, Any]]:
    return json.loads(cast(str, attributes["gen_ai.output.messages"]))


def _assert_text_span(span: ReadableSpan, output_text: str):
    attributes = _attributes(span)
    assert span.name == "openai.response"
    assert span.status.status_code == StatusCode.OK
    assert attributes["lmnr.span.type"] == "LLM"
    assert attributes["gen_ai.system"] == "openai"
    assert attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert cast(str, attributes["gen_ai.response.model"]).startswith("gpt-4.1-nano")
    assert cast(str, attributes["gen_ai.response.id"]).startswith("resp_")
    assert attributes["gen_ai.prompt.0.content"] == "What is the capital of France?"
    assert attributes["gen_ai.prompt.0.role"] == "user"
    assert attributes["gen_ai.completion.0.content"] == output_text
    assert attributes["gen_ai.completion.0.role"] == "assistant"
    assert cast(int, attributes["gen_ai.usage.input_tokens"]) > 0
    assert cast(int, attributes["gen_ai.usage.output_tokens"]) > 0
    assert json.loads(cast(str, attributes["gen_ai.input.messages"])) == [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    assert _output_messages(attributes)[0]["content"][0]["text"] == output_text


# --- live streams (VCR) -------------------------------------------------------


@pytest.mark.vcr
def test_responses_streaming(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    stream = openai_client.responses.create(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
        stream=True,
    )
    output_text = ""
    for event in stream:
        # The span stays open until the stream is exhausted.
        assert span_exporter.get_finished_spans() == ()
        if event.type == "response.output_text.delta":
            output_text += event.delta

    assert "Paris" in output_text
    _assert_text_span(_only_span(span_exporter), output_text)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_responses_streaming_async(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    stream = await async_openai_client.responses.create(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
        stream=True,
    )
    output_text = ""
    async for event in stream:
        if event.type == "response.output_text.delta":
            output_text += event.delta

    assert "Paris" in output_text
    _assert_text_span(_only_span(span_exporter), output_text)


@pytest.mark.vcr
def test_responses_stream_helper(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    with openai_client.responses.stream(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
    ) as stream:
        for _ in stream:
            pass
        final = stream.get_final_response()

    assert "Paris" in final.output_text
    span = _only_span(span_exporter)
    _assert_text_span(span, final.output_text)
    assert _attributes(span)["gen_ai.response.id"] == final.id


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_responses_stream_helper_async(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    async with async_openai_client.responses.stream(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
    ) as stream:
        async for _ in stream:
            pass
        final = await stream.get_final_response()

    assert "Paris" in final.output_text
    span = _only_span(span_exporter)
    _assert_text_span(span, final.output_text)
    assert _attributes(span)["gen_ai.response.id"] == final.id


@pytest.mark.vcr
def test_responses_streaming_tool_calls(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    stream = openai_client.responses.create(
        model="gpt-4.1-nano",
        input=[{"type": "message", "role": "user", "content": "What's the weather in London?"}],
        tools=TOOLS,
        tool_choice="required",
        stream=True,
    )
    completed: Response | None = None
    for event in stream:
        if event.type == "response.completed":
            completed = event.response

    assert completed is not None
    function_call = completed.output[0]
    assert function_call.type == "function_call"
    attributes = _attributes(_only_span(span_exporter))
    assert attributes["gen_ai.completion.0.tool_calls.0.name"] == "get_weather"
    assert attributes["gen_ai.completion.0.tool_calls.0.id"] == function_call.id
    assert attributes["gen_ai.completion.0.tool_calls.0.arguments"] == function_call.arguments
    assert json.loads(cast(str, attributes["gen_ai.tool.definitions"])) == TOOLS
    assert "London" in function_call.arguments


@pytest.mark.vcr
def test_responses_streaming_early_close(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    stream = openai_client.responses.create(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
        stream=True,
    )
    with stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                break

    # Leaving the context manager ends the span with what was streamed so far.
    span = _only_span(span_exporter)
    attributes = _attributes(span)
    assert span.status.status_code == StatusCode.OK
    assert attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert cast(str, attributes["gen_ai.response.id"]).startswith("resp_")
    assert attributes["gen_ai.prompt.0.content"] == "What is the capital of France?"
    assert "gen_ai.usage.input_tokens" not in attributes


# --- mock transport -----------------------------------------------------------


def _response(text: str = "Paris", response_id: str = "resp_mock") -> Response:
    return Response.model_validate(
        {
            "id": response_id,
            "created_at": 0,
            "model": "gpt-4.1-nano-2025-04-14",
            "object": "response",
            "output": [
                {
                    "id": "msg_mock",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "usage": {
                "input_tokens": 14,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 16,
            },
        }
    )


def _sse(events: list[ResponseStreamEvent]) -> bytes:
    return b"".join(
        f"event: {event.type}\ndata: {event.model_dump_json()}\n\n".encode() for event in events
    )


def _events(response: Response) -> list[ResponseStreamEvent]:
    events = OpenAIRolloutWrapper().cached_response_to_responses_events(response)
    assert events is not None
    return events


def _transport(requests: list[httpx.Request], status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if status != 200:
            return httpx.Response(
                status, json={"error": {"message": "bad input", "type": "invalid_request_error"}}
            )
        return httpx.Response(
            200,
            content=_sse(_events(_response())),
            headers={"content-type": "text/event-stream"},
        )

    return httpx.MockTransport(handler)


def _client(requests: list[httpx.Request], status: int = 200) -> OpenAI:
    return OpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=_transport(requests, status)),
    )


def _async_client(requests: list[httpx.Request], status: int = 200) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=_transport(requests, status)),
    )


def test_responses_streaming_api_error_propagates(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
):
    requests: list[httpx.Request] = []
    with pytest.raises(BadRequestError):
        _ = _client(requests, status=400).responses.create(
            model="gpt-4.1-nano", input="What is the capital of France?", stream=True
        )

    span = _only_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert _attributes(span)["error.type"] == "BadRequestError"
    assert _attributes(span)["gen_ai.prompt.0.content"] == "What is the capital of France?"


@pytest.mark.asyncio
async def test_responses_api_error_propagates_async(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
):
    requests: list[httpx.Request] = []
    with pytest.raises(BadRequestError):
        _ = await _async_client(requests, status=400).responses.create(
            model="gpt-4.1-nano", input="What is the capital of France?"
        )

    assert _only_span(span_exporter).status.status_code == StatusCode.ERROR


def test_responses_streaming_mock_preserves_events(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
):
    requests: list[httpx.Request] = []
    stream = _client(requests).responses.create(
        model="gpt-4.1-nano", input="What is the capital of France?", stream=True
    )
    received = list(stream)

    assert [event.type for event in received] == [event.type for event in _events(_response())]
    _assert_text_span(_only_span(span_exporter), "Paris")
    # A finished stream doesn't leave its data in the background-response stash.
    assert "resp_mock" not in responses_wrappers.responses


def test_responses_streaming_mock_iteration_error(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
):
    def handler(request: httpx.Request) -> httpx.Response:
        events = _events(_response())
        # Cut the stream off mid-way with an `error` event: the SDK raises on it.
        body = _sse(events[:4]) + b'event: error\ndata: {"error": {"message": "boom"}}\n\n'
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = OpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    stream = client.responses.create(
        model="gpt-4.1-nano", input="What is the capital of France?", stream=True
    )
    with pytest.raises(Exception, match="boom"):
        for _ in stream:
            pass

    span = _only_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert _attributes(span)["gen_ai.response.id"] == "resp_mock"


def test_responses_stream_helper_early_exit_ends_span(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
):
    requests: list[httpx.Request] = []
    with _client(requests).responses.stream(
        model="gpt-4.1-nano", input="What is the capital of France?"
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                break
        assert span_exporter.get_finished_spans() == ()

    # `ResponseStream.__exit__` closes the httpx response, not our proxy; the
    # span still ends here, while `stream` is alive, not at garbage collection.
    span = _only_span(span_exporter)
    assert span.status.status_code == StatusCode.OK
    assert _attributes(span)["gen_ai.response.id"] == "resp_mock"
    assert stream is not None


@pytest.mark.asyncio
async def test_responses_stream_helper_early_exit_ends_span_async(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
):
    requests: list[httpx.Request] = []
    async with _async_client(requests).responses.stream(
        model="gpt-4.1-nano", input="What is the capital of France?"
    ) as stream:
        async for event in stream:
            if event.type == "response.output_text.delta":
                break
        assert span_exporter.get_finished_spans() == ()

    span = _only_span(span_exporter)
    assert span.status.status_code == StatusCode.OK
    assert _attributes(span)["gen_ai.response.id"] == "resp_mock"
    assert stream is not None


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
    monkeypatch.setattr(replay, "cache_outcome_for", outcome_for)
    monkeypatch.setattr(replay, "acache_outcome_for", aoutcome_for)
    yield outcomes


def _hit(text: str = "Rome") -> CacheOutcome:
    return CacheOutcome(
        kind="hit",
        cached={"type": "raw", "response": _response(text, "resp_cached").model_dump()},
    )


def test_replay_hit_serves_cached_stream(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    replay_outcome.append(_hit())
    requests: list[httpx.Request] = []
    stream = _client(requests).responses.create(
        model="gpt-4.1-nano", input="What is the capital of France?", stream=True
    )

    assert isinstance(stream, CachedResponsesEventStream)
    output_text = "".join(
        event.delta for event in stream if event.type == "response.output_text.delta"
    )
    assert output_text == "Rome"
    assert requests == []
    span = _only_span(span_exporter)
    attributes = _attributes(span)
    assert attributes["lmnr.span.type"] == "CACHED"
    assert attributes["gen_ai.completion.0.content"] == "Rome"
    assert "lmnr.sdk.raw.response" not in attributes


def test_replay_hit_serves_cached_stream_helper(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    replay_outcome.append(_hit())
    requests: list[httpx.Request] = []
    with _client(requests).responses.stream(
        model="gpt-4.1-nano", input="What is the capital of France?"
    ) as stream:
        deltas = [event.delta for event in stream if event.type == "response.output_text.delta"]
        final = stream.get_final_response()

    assert deltas == ["Rome"]
    assert final.output_text == "Rome"
    assert requests == []
    assert _attributes(_only_span(span_exporter))["lmnr.span.type"] == "CACHED"


@pytest.mark.asyncio
async def test_replay_hit_serves_cached_stream_async(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    replay_outcome.append(_hit())
    requests: list[httpx.Request] = []
    stream = await _async_client(requests).responses.create(
        model="gpt-4.1-nano", input="What is the capital of France?", stream=True
    )

    assert isinstance(stream, AsyncCachedResponsesEventStream)
    output_text = "".join(
        [event.delta async for event in stream if event.type == "response.output_text.delta"]
    )
    assert output_text == "Rome"
    assert requests == []
    assert _attributes(_only_span(span_exporter))["lmnr.span.type"] == "CACHED"


@pytest.mark.asyncio
async def test_replay_hit_serves_cached_stream_helper_async(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    replay_outcome.append(_hit())
    requests: list[httpx.Request] = []
    async with _async_client(requests).responses.stream(
        model="gpt-4.1-nano", input="What is the capital of France?"
    ) as stream:
        async for _ in stream:
            pass
        final = await stream.get_final_response()

    assert final.output_text == "Rome"
    assert requests == []
    assert _attributes(_only_span(span_exporter))["lmnr.span.type"] == "CACHED"


def test_replay_miss_traces_live_stream(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    replay_outcome.append(CacheOutcome(kind="miss"))
    requests: list[httpx.Request] = []
    stream = _client(requests).responses.create(
        model="gpt-4.1-nano", input="What is the capital of France?", stream=True
    )
    output_text = "".join(
        event.delta for event in stream if event.type == "response.output_text.delta"
    )

    assert output_text == "Paris"
    assert len(requests) == 1
    span = _only_span(span_exporter)
    attributes = _attributes(span)
    assert attributes["lmnr.span.type"] == "LLM"
    _assert_text_span(span, "Paris")
    # The completed response is recorded so this trace can be replayed later.
    raw = json.loads(cast(str, attributes["lmnr.sdk.raw.response"]))
    assert raw["id"] == "resp_mock"
    cached = OpenAIRolloutWrapper().cached_response_to_responses({"type": "raw", "response": raw})
    assert cached is not None and cached.output_text == "Paris"


def test_replay_hit_with_raw_response_runs_live(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    replay_outcome: list[CacheOutcome],
):
    # `with_raw_response` callers expect an `APIResponse` wrapper the cache can't
    # produce, so a HIT falls through to the live call.
    replay_outcome.append(_hit())
    requests: list[httpx.Request] = []
    raw = _client(requests).responses.with_raw_response.create(
        model="gpt-4.1-nano", input="What is the capital of France?", stream=True
    )
    events = list(raw.parse())

    assert len(requests) == 1
    assert events[-1].type == "response.completed"
    assert _attributes(_only_span(span_exporter))["lmnr.span.type"] == "LLM"

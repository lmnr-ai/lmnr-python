"""Tests for the Cohere instrumentation.

The SDK's HTTP client is swapped for an `httpx.MockTransport` that serves
canned API responses (JSON, newline-delimited v1 streams, v2 SSE streams), so
the full SDK runs, including response parsing, without network or cassettes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import cohere
import httpx
import pytest
from _wrapt_compat import is_wrapt_proxy
from cohere.base_client import BaseCohere
from cohere.v2.client import V2Client
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.cohere import CohereInstrumentor

MODEL = "command-a-03-2025"
HI: list[cohere.ChatMessageV2] = [cohere.UserChatMessageV2(content="hi")]

_META = {
    "billed_units": {"input_tokens": 5, "output_tokens": 2},
    "tokens": {"input_tokens": 12, "output_tokens": 3},
}
_USAGE = {**_META, "cached_tokens": 4}

V1_CHAT = {
    "text": "Hello there",
    "generation_id": "gen-1",
    "response_id": "resp-1",
    "finish_reason": "COMPLETE",
    "meta": _META,
}
V2_CHAT = {
    "id": "resp-2",
    "finish_reason": "COMPLETE",
    "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello there"}]},
    "usage": _USAGE,
}
V1_STREAM = [
    {"event_type": "stream-start", "generation_id": "gen-1", "is_finished": False},
    {"event_type": "text-generation", "text": "Hel", "is_finished": False},
    {"event_type": "text-generation", "text": "lo", "is_finished": False},
    {
        "event_type": "stream-end",
        "finish_reason": "COMPLETE",
        "is_finished": True,
        "response": {**V1_CHAT, "text": "Hello"},
    },
]
V2_STREAM = [
    {
        "type": "message-start",
        "id": "resp-2",
        "delta": {"message": {"role": "assistant", "content": [], "tool_calls": []}},
    },
    {
        "type": "content-start",
        "index": 0,
        "delta": {"message": {"content": {"type": "text", "text": ""}}},
    },
    {"type": "content-delta", "index": 0, "delta": {"message": {"content": {"text": "Hel"}}}},
    {"type": "content-delta", "index": 0, "delta": {"message": {"content": {"text": "lo"}}}},
    {"type": "content-end", "index": 0},
    {"type": "message-end", "delta": {"finish_reason": "COMPLETE", "usage": _USAGE}},
]
V2_TOOL_STREAM = [
    V2_STREAM[0],
    {"type": "tool-plan-delta", "delta": {"message": {"tool_plan": "I will check "}}},
    {"type": "tool-plan-delta", "delta": {"message": {"tool_plan": "the weather."}}},
    {
        "type": "tool-call-start",
        "index": 0,
        "delta": {
            "message": {
                "tool_calls": {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ""},
                }
            }
        },
    },
    {
        "type": "tool-call-delta",
        "index": 0,
        "delta": {"message": {"tool_calls": {"function": {"arguments": '{"city": '}}}},
    },
    {
        "type": "tool-call-delta",
        "index": 0,
        "delta": {"message": {"tool_calls": {"function": {"arguments": '"Paris"}'}}}},
    },
    {"type": "tool-call-end", "index": 0},
    {"type": "message-end", "delta": {"finish_reason": "TOOL_CALL", "usage": _USAGE}},
]
GENERATE = {
    "id": "gen-resp",
    "generations": [{"id": "g-1", "text": "Once upon a time"}],
    "meta": _META,
}
RERANK = {
    "id": "rerank-1",
    "results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}],
    "meta": {"billed_units": {"search_units": 1}},
}
V1_EMBED = {
    "response_type": "embeddings_floats",
    "id": "embed-1",
    "embeddings": [[0.1, 0.2]],
    "texts": ["hello"],
    "meta": {"billed_units": {"input_tokens": 3}},
}
V2_EMBED = {
    "id": "embed-2",
    "embeddings": {"float": [[0.1, 0.2]]},
    "texts": ["hello"],
    "meta": {"billed_units": {"input_tokens": 3}},
}

Handler = Callable[[httpx.Request], httpx.Response]


def _ndjson(events: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, content="\n".join(json.dumps(e) for e in events) + "\n")


def _sse(events: list[dict[str, Any]]) -> httpx.Response:
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def _api(v2_stream: list[dict[str, Any]] = V2_STREAM, v2_chat: dict[str, Any] = V2_CHAT) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        path = request.url.path
        if path == "/v1/chat":
            return _ndjson(V1_STREAM) if body.get("stream") else httpx.Response(200, json=V1_CHAT)
        if path == "/v2/chat":
            return _sse(v2_stream) if body.get("stream") else httpx.Response(200, json=v2_chat)
        responses = {
            "/v1/generate": GENERATE,
            "/v1/rerank": RERANK,
            "/v2/rerank": RERANK,
            "/v1/embed": V1_EMBED,
            "/v2/embed": V2_EMBED,
        }
        return httpx.Response(200, json=responses[path])

    return handle


def _client(handler: Handler | None = None) -> cohere.Client:
    transport = httpx.MockTransport(handler or _api())
    return cohere.Client(api_key="test", httpx_client=httpx.Client(transport=transport))


def _client_v2(handler: Handler | None = None) -> cohere.ClientV2:
    transport = httpx.MockTransport(handler or _api())
    return cohere.ClientV2(api_key="test", httpx_client=httpx.Client(transport=transport))


def _async_client(handler: Handler | None = None) -> cohere.AsyncClient:
    transport = httpx.MockTransport(handler or _api())
    return cohere.AsyncClient(api_key="test", httpx_client=httpx.AsyncClient(transport=transport))


def _async_client_v2(handler: Handler | None = None) -> cohere.AsyncClientV2:
    transport = httpx.MockTransport(handler or _api())
    return cohere.AsyncClientV2(api_key="test", httpx_client=httpx.AsyncClient(transport=transport))


def _only_span(span_exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, [s.name for s in spans]
    return spans[0]


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    assert span.attributes is not None
    return dict(span.attributes)


def _attr_json(span: ReadableSpan, key: str) -> Any:
    return json.loads(str(_attrs(span)[key]))


def _assert_usage(attrs: dict[str, Any], cached: bool = False):
    # actual `tokens` win over `billed_units`
    assert attrs["gen_ai.usage.input_tokens"] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 3
    assert attrs["llm.usage.total_tokens"] == 15
    if cached:
        assert attrs["gen_ai.usage.cache_read_input_tokens"] == 4


# --- v1 chat --------------------------------------------------------------------


def test_v1_chat(span_exporter: InMemorySpanExporter):
    response = _client().chat(
        model=MODEL,
        message="Say hello",
        preamble="Be terse.",
        chat_history=[
            cohere.UserMessage(message="Hi"),
            cohere.ChatbotMessage(message="Hello!"),
        ],
        temperature=0.3,
        max_tokens=64,
        p=0.9,
        k=5,
        seed=7,
        stop_sequences=["END"],
    )
    assert response.text == "Hello there"

    span = _only_span(span_exporter)
    attrs = _attrs(span)
    assert span.name == "cohere.chat"
    assert attrs["lmnr.span.type"] == "LLM"
    assert attrs["gen_ai.system"] == "cohere"
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == MODEL
    assert attrs["gen_ai.request.temperature"] == pytest.approx(0.3)
    assert attrs["gen_ai.request.max_tokens"] == 64
    assert attrs["gen_ai.request.top_p"] == pytest.approx(0.9)
    assert attrs["gen_ai.request.top_k"] == 5
    assert attrs["gen_ai.request.seed"] == 7
    assert attrs["gen_ai.request.stop_sequences"] == ("END",)
    assert attrs["llm.is_streaming"] is False
    assert attrs["gen_ai.response.id"] == "resp-1"
    assert attrs["gen_ai.response.finish_reasons"] == ("COMPLETE",)
    assert attrs["lmnr.span.instrumentation_scope.name"] == "cohere"
    _assert_usage(attrs)
    assert _attr_json(span, "gen_ai.input.messages") == [
        {"role": "system", "parts": [{"type": "text", "content": "Be terse."}]},
        {"role": "user", "parts": [{"type": "text", "content": "Hi"}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "Hello!"}]},
        {"role": "user", "parts": [{"type": "text", "content": "Say hello"}]},
    ]
    assert _attr_json(span, "gen_ai.output.messages") == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "Hello there"}],
            "finish_reason": "COMPLETE",
        }
    ]


def test_v1_chat_tool_calls_and_results(span_exporter: InMemorySpanExporter):
    weather_call = cohere.ToolCall(name="get_weather", parameters={"city": "Paris"})

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={**V1_CHAT, "text": "", "tool_calls": [weather_call.dict()]}
        )

    _ = _client(handle).chat(
        model=MODEL,
        message="Weather in Paris?",
        chat_history=[
            cohere.ChatbotMessage(message="Checking", tool_calls=[weather_call]),
            cohere.ToolMessage(
                tool_results=[cohere.ToolResult(call=weather_call, outputs=[{"temp": 20}])]
            ),
        ],
        tools=[
            cohere.Tool(
                name="get_weather",
                description="Get the weather",
                parameter_definitions={
                    "city": cohere.ToolParameterDefinitionsValue(type="str", required=True)
                },
            )
        ],
    )

    span = _only_span(span_exporter)
    messages = _attr_json(span, "gen_ai.input.messages")
    tool_call = {"type": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}
    assert messages[0] == {
        "role": "assistant",
        "parts": [{"type": "text", "content": "Checking"}, tool_call],
    }
    assert messages[1] == {
        "role": "tool",
        "parts": [{"type": "tool_call_response", "name": "get_weather", "result": [{"temp": 20}]}],
    }
    tools = _attr_json(span, "gen_ai.tool.definitions")
    assert tools[0]["name"] == "get_weather"
    assert tools[0]["description"] == "Get the weather"
    assert tools[0]["parameters"]["city"]["type"] == "str"
    output = _attr_json(span, "gen_ai.output.messages")
    assert output[0]["parts"] == [tool_call]


def test_v1_chat_stream(span_exporter: InMemorySpanExporter):
    events = list(_client().chat_stream(model=MODEL, message="Say hello"))
    assert [e.event_type for e in events][-1] == "stream-end"

    span = _only_span(span_exporter)
    attrs = _attrs(span)
    assert span.name == "cohere.chat"
    assert attrs["llm.is_streaming"] is True
    assert attrs["gen_ai.response.id"] == "resp-1"
    _assert_usage(attrs)
    # taken from the `stream-end` response
    assert _attr_json(span, "gen_ai.output.messages") == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "Hello"}],
            "finish_reason": "COMPLETE",
        }
    ]


def test_v1_chat_stream_early_break_ends_span(span_exporter: InMemorySpanExporter):
    stream = _client().chat_stream(model=MODEL, message="Say hello")
    for event in stream:
        if event.event_type == "text-generation":
            break
    assert span_exporter.get_finished_spans() == ()
    stream.close()  # pyright: ignore[reportAttributeAccessIssue]

    output = _attr_json(_only_span(span_exporter), "gen_ai.output.messages")
    assert output == [{"role": "assistant", "parts": [{"type": "text", "content": "Hel"}]}]


@pytest.mark.asyncio
async def test_v1_chat_async(span_exporter: InMemorySpanExporter):
    response = await _async_client().chat(model=MODEL, message="Say hello")
    assert response.text == "Hello there"

    span = _only_span(span_exporter)
    assert span.name == "cohere.chat"
    _assert_usage(_attrs(span))
    assert _attr_json(span, "gen_ai.output.messages")[0]["parts"][0]["content"] == "Hello there"


@pytest.mark.asyncio
async def test_v1_chat_stream_async(span_exporter: InMemorySpanExporter):
    events = [e async for e in _async_client().chat_stream(model=MODEL, message="Say hello")]
    assert len(events) == len(V1_STREAM)

    span = _only_span(span_exporter)
    assert _attrs(span)["llm.is_streaming"] is True
    assert _attr_json(span, "gen_ai.output.messages")[0]["parts"][0]["content"] == "Hello"


# --- v2 chat --------------------------------------------------------------------


def test_v2_chat(span_exporter: InMemorySpanExporter):
    response = _client_v2().chat(
        model=MODEL,
        messages=[
            cohere.SystemChatMessageV2(content="Be terse."),
            cohere.UserChatMessageV2(
                content=[
                    cohere.TextContent(text="Describe"),
                    cohere.ImageUrlContent(
                        image_url=cohere.ImageUrl(url="data:image/png;base64,iVBORw0K")
                    ),
                    cohere.ImageUrlContent(
                        image_url=cohere.ImageUrl(url="https://example.com/cat.png")
                    ),
                ]
            ),
        ],
        temperature=0.2,
        max_tokens=32,
    )
    assert response.message.content is not None

    span = _only_span(span_exporter)
    attrs = _attrs(span)
    assert span.name == "cohere.chat"
    assert attrs["gen_ai.request.temperature"] == pytest.approx(0.2)
    assert attrs["gen_ai.request.max_tokens"] == 32
    assert attrs["gen_ai.response.id"] == "resp-2"
    assert attrs["gen_ai.response.finish_reasons"] == ("COMPLETE",)
    _assert_usage(attrs, cached=True)
    assert _attr_json(span, "gen_ai.input.messages") == [
        {"role": "system", "parts": [{"type": "text", "content": "Be terse."}]},
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": "Describe"},
                {
                    "type": "blob",
                    "modality": "image",
                    "mime_type": "image/png",
                    "content": "iVBORw0K",
                },
                {"type": "uri", "modality": "image", "uri": "https://example.com/cat.png"},
            ],
        },
    ]
    assert _attr_json(span, "gen_ai.output.messages") == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "Hello there"}],
            "finish_reason": "COMPLETE",
        }
    ]


def test_v2_chat_tool_calls(span_exporter: InMemorySpanExporter):
    tool_call = cohere.ToolCallV2(
        id="call-1",
        type="function",
        function=cohere.ToolCallV2Function(name="get_weather", arguments='{"city": "Paris"}'),
    )
    response_body = {
        **V2_CHAT,
        "finish_reason": "TOOL_CALL",
        "message": {
            "role": "assistant",
            "tool_plan": "I will check the weather.",
            "tool_calls": [tool_call.dict()],
        },
    }
    _ = _client_v2(_api(v2_chat=response_body)).chat(
        model=MODEL,
        messages=[
            cohere.UserChatMessageV2(content="Weather in Paris?"),
            cohere.AssistantChatMessageV2(tool_plan="Checking", tool_calls=[tool_call]),
            cohere.ToolChatMessageV2(tool_call_id="call-1", content='{"temp": 20}'),
        ],
        tools=[
            cohere.ToolV2(
                type="function",
                function=cohere.ToolV2Function(
                    name="get_weather",
                    description="Get the weather",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                ),
            )
        ],
    )

    span = _only_span(span_exporter)
    expected_call = {
        "type": "tool_call",
        "id": "call-1",
        "name": "get_weather",
        "arguments": {"city": "Paris"},
    }
    messages = _attr_json(span, "gen_ai.input.messages")
    assert messages[1] == {
        "role": "assistant",
        "parts": [{"type": "text", "content": "Checking"}, expected_call],
    }
    assert messages[2] == {
        "role": "tool",
        "parts": [{"type": "tool_call_response", "id": "call-1", "result": '{"temp": 20}'}],
    }
    assert _attr_json(span, "gen_ai.tool.definitions") == [
        {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    assert _attr_json(span, "gen_ai.output.messages") == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "I will check the weather."}, expected_call],
            "finish_reason": "TOOL_CALL",
        }
    ]


def test_v2_chat_via_v1_client_property(span_exporter: InMemorySpanExporter):
    _ = _client().v2.chat(model=MODEL, messages=HI)
    span = _only_span(span_exporter)
    assert _attrs(span)["gen_ai.response.id"] == "resp-2"
    assert _attr_json(span, "gen_ai.input.messages") == [
        {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
    ]


def test_v2_chat_stream(span_exporter: InMemorySpanExporter):
    events = list(_client_v2().chat_stream(model=MODEL, messages=HI))
    assert events[-1].type == "message-end"

    span = _only_span(span_exporter)
    attrs = _attrs(span)
    assert attrs["llm.is_streaming"] is True
    assert attrs["gen_ai.response.id"] == "resp-2"
    assert attrs["gen_ai.response.finish_reasons"] == ("COMPLETE",)
    _assert_usage(attrs, cached=True)
    assert _attr_json(span, "gen_ai.output.messages") == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "Hello"}],
            "finish_reason": "COMPLETE",
        }
    ]


def test_v2_chat_stream_tool_calls(span_exporter: InMemorySpanExporter):
    client = _client_v2(_api(v2_stream=V2_TOOL_STREAM))
    _ = list(
        client.chat_stream(model=MODEL, messages=[cohere.UserChatMessageV2(content="weather?")])
    )

    output = _attr_json(_only_span(span_exporter), "gen_ai.output.messages")
    assert output == [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "I will check the weather."},
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "get_weather",
                    "arguments": {"city": "Paris"},
                },
            ],
            "finish_reason": "TOOL_CALL",
        }
    ]


def test_v2_chat_stream_early_break_ends_span(span_exporter: InMemorySpanExporter):
    stream = _client_v2().chat_stream(model=MODEL, messages=HI)
    for event in stream:
        if event.type == "content-delta":
            break
    stream.close()  # pyright: ignore[reportAttributeAccessIssue]

    output = _attr_json(_only_span(span_exporter), "gen_ai.output.messages")
    assert output == [{"role": "assistant", "parts": [{"type": "text", "content": "Hel"}]}]


@pytest.mark.asyncio
async def test_v2_chat_async(span_exporter: InMemorySpanExporter):
    _ = await _async_client_v2().chat(model=MODEL, messages=HI)
    span = _only_span(span_exporter)
    assert _attrs(span)["gen_ai.response.id"] == "resp-2"
    _assert_usage(_attrs(span), cached=True)


@pytest.mark.asyncio
async def test_v2_chat_stream_async(span_exporter: InMemorySpanExporter):
    stream = _async_client_v2().chat_stream(model=MODEL, messages=HI)
    events = [e async for e in stream]
    assert len(events) == len(V2_STREAM)

    span = _only_span(span_exporter)
    assert _attrs(span)["llm.is_streaming"] is True
    assert _attr_json(span, "gen_ai.output.messages")[0]["parts"] == [
        {"type": "text", "content": "Hello"}
    ]


# --- generate / rerank / embed --------------------------------------------------


def test_generate(span_exporter: InMemorySpanExporter):
    _ = _client().generate(model="command", prompt="Tell a story", max_tokens=20)

    span = _only_span(span_exporter)
    attrs = _attrs(span)
    assert span.name == "cohere.completion"
    assert attrs["gen_ai.operation.name"] == "text_completion"
    assert attrs["gen_ai.request.max_tokens"] == 20
    assert attrs["gen_ai.response.id"] == "gen-resp"
    _assert_usage(attrs)
    assert _attr_json(span, "gen_ai.input.messages") == [
        {"role": "user", "parts": [{"type": "text", "content": "Tell a story"}]}
    ]
    assert _attr_json(span, "gen_ai.output.messages") == [
        {"role": "assistant", "parts": [{"type": "text", "content": "Once upon a time"}]}
    ]


def test_rerank(span_exporter: InMemorySpanExporter):
    _ = _client().rerank(
        model="rerank-v3.5", query="capital of France", documents=["Berlin", "Paris"]
    )

    span = _only_span(span_exporter)
    attrs = _attrs(span)
    assert span.name == "cohere.rerank"
    assert attrs["gen_ai.operation.name"] == "rerank"
    assert attrs["gen_ai.response.id"] == "rerank-1"
    assert _attr_json(span, "gen_ai.input.messages") == [
        {"role": "system", "parts": [{"type": "text", "content": "Berlin"}]},
        {"role": "system", "parts": [{"type": "text", "content": "Paris"}]},
        {"role": "user", "parts": [{"type": "text", "content": "capital of France"}]},
    ]
    assert _attr_json(span, "gen_ai.output.messages") == [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "Doc 1, Score: 0.9"},
                {"type": "text", "content": "Doc 0, Score: 0.1"},
            ],
        }
    ]


def test_rerank_v2(span_exporter: InMemorySpanExporter):
    _ = _client_v2().rerank(model="rerank-v3.5", query="q", documents=["a", "b"])
    span = _only_span(span_exporter)
    assert span.name == "cohere.rerank"
    assert _attrs(span)["gen_ai.response.id"] == "rerank-1"


def test_embed(span_exporter: InMemorySpanExporter):
    _ = _client().embed(model="embed-v4.0", texts=["hello"], input_type="search_document")

    # `Client.embed` batches through the unwrapped base method: still one span
    span = _only_span(span_exporter)
    attrs = _attrs(span)
    assert span.name == "cohere.embed"
    assert attrs["gen_ai.operation.name"] == "embeddings"
    assert attrs["gen_ai.response.id"] == "embed-1"
    assert attrs["gen_ai.usage.input_tokens"] == 3
    assert _attr_json(span, "gen_ai.input.messages") == [
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
    ]
    assert "gen_ai.output.messages" not in attrs


def test_embed_v2(span_exporter: InMemorySpanExporter):
    _ = _client_v2().embed(
        model="embed-v4.0",
        texts=["hello"],
        input_type="search_document",
        embedding_types=["float"],
    )
    span = _only_span(span_exporter)
    assert span.name == "cohere.embed"
    assert _attrs(span)["gen_ai.response.id"] == "embed-2"


@pytest.mark.asyncio
async def test_embed_async(span_exporter: InMemorySpanExporter):
    _ = await _async_client().embed(
        model="embed-v4.0", texts=["hello"], input_type="search_document"
    )
    span = _only_span(span_exporter)
    assert span.name == "cohere.embed"
    assert _attrs(span)["gen_ai.response.id"] == "embed-1"


# --- errors, context, suppression -----------------------------------------------


def _failing(request: httpx.Request) -> httpx.Response:
    return httpx.Response(400, json={"message": "invalid model"})


def test_chat_error(span_exporter: InMemorySpanExporter):
    with pytest.raises(cohere.BadRequestError):
        _ = _client_v2(_failing).chat(model="nope", messages=HI)

    span = _only_span(span_exporter)
    assert span.status.is_ok is False
    assert _attrs(span)["error.type"] == "BadRequestError"
    assert [e.name for e in span.events] == ["exception"]


def test_stream_error(span_exporter: InMemorySpanExporter):
    # the request only starts on the first `next()`, so the error surfaces mid-iteration
    stream = _client_v2(_failing).chat_stream(model="nope", messages=HI)
    with pytest.raises(cohere.BadRequestError):
        _ = list(stream)

    span = _only_span(span_exporter)
    assert _attrs(span)["error.type"] == "BadRequestError"
    assert [e.name for e in span.events] == ["exception"]


def _assert_parented(spans: dict[str, ReadableSpan], child: str, parent: str):
    child_span, parent_span = spans[child], spans[parent]
    assert child_span.parent is not None and parent_span.context is not None
    assert child_span.parent.span_id == parent_span.context.span_id


def _with_transport_span(handler: Handler) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        with Laminar.start_as_current_span("transport"):
            return handler(request)

    return handle


def test_llm_span_is_current_during_the_call(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("outer"):
        _ = _client(_with_transport_span(_api())).chat(model=MODEL, message="hi")

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    _assert_parented(spans, "cohere.chat", "outer")
    _assert_parented(spans, "transport", "cohere.chat")


def test_llm_span_is_current_while_streaming(span_exporter: InMemorySpanExporter):
    client = _client_v2(_with_transport_span(_api()))
    with Laminar.start_as_current_span("outer"):
        for _ in client.chat_stream(model=MODEL, messages=HI):
            # the consumer's own work between events is not part of the LLM call
            with Laminar.start_as_current_span("consumer"):
                pass

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    _assert_parented(spans, "transport", "cohere.chat")
    _assert_parented(spans, "consumer", "outer")


@pytest.mark.asyncio
async def test_llm_span_is_current_while_async_streaming(span_exporter: InMemorySpanExporter):
    client = _async_client_v2(_with_transport_span(_api()))
    with Laminar.start_as_current_span("outer"):
        stream = client.chat_stream(model=MODEL, messages=HI)
        async for _ in stream:
            with Laminar.start_as_current_span("consumer"):
                pass

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    _assert_parented(spans, "transport", "cohere.chat")
    _assert_parented(spans, "consumer", "outer")


def test_suppressed_instrumentation(span_exporter: InMemorySpanExporter):
    token = context_api.attach(
        context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
        _ = _client().chat(model=MODEL, message="hi")
    finally:
        context_api.detach(token)
    assert span_exporter.get_finished_spans() == ()


def test_no_content_when_tracing_content_disabled(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LMNR_TRACE_CONTENT", "false")
    _ = _client().chat(model=MODEL, message="secret")

    attrs = _attrs(_only_span(span_exporter))
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs
    assert attrs["gen_ai.request.model"] == MODEL
    _assert_usage(attrs)


@pytest.fixture
def _uninstrumented() -> Iterator[CohereInstrumentor]:
    instrumentor = CohereInstrumentor()
    assert instrumentor.is_instrumented_by_opentelemetry
    try:
        instrumentor.uninstrument()
        yield instrumentor
    finally:
        instrumentor.instrument()


def test_uninstrument_removes_wrappers(_uninstrumented: CohereInstrumentor):
    assert not is_wrapt_proxy(BaseCohere.__dict__["chat"])
    assert not is_wrapt_proxy(V2Client.__dict__["chat_stream"])
    assert not is_wrapt_proxy(cohere.Client.__dict__["embed"])


def test_wrappers_restored_after_reinstrument():
    assert is_wrapt_proxy(BaseCohere.__dict__["chat"])
    assert is_wrapt_proxy(V2Client.__dict__["chat"])

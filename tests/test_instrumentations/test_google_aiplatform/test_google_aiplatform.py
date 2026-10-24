"""Tests for the google-cloud-aiplatform (`vertexai`) instrumentation.

The SDK talks gRPC, which VCR cannot record, so these tests stub the SDK's
private transport methods (`_GenerativeModel._generate_content*`, the
language-model endpoint) and assert on the spans the public methods produce.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import vertexai
import vertexai.language_models._language_models as language_models_module
from _wrapt_compat import is_wrapt_proxy
from google.auth.credentials import AnonymousCredentials
from google.cloud.aiplatform.models import Prediction
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Image,
    Part,
    Tool,
)
from vertexai.generative_models._generative_models import _GenerativeModel
from vertexai.language_models import ChatModel, TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel as PreviewGenerativeModel

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_aiplatform import (
    GoogleAIPlatformInstrumentor,
)

MODEL = "gemini-2.0-flash"

pytestmark = pytest.mark.filterwarnings("ignore:This feature is deprecated:UserWarning")


@pytest.fixture(autouse=True, scope="module")
def _vertexai_init():
    vertexai.init(
        project="test-project", location="us-central1", credentials=AnonymousCredentials()
    )


def _response(
    parts: list[dict[str, Any]] | None = None,
    finish_reason: str = "STOP",
    usage: dict[str, int] | None = None,
) -> GenerationResponse:
    response: dict[str, Any] = {
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": parts or [{"text": "Hello there"}]},
                "finish_reason": finish_reason,
            }
        ],
        "model_version": MODEL,
        "response_id": "resp-123",
    }
    if usage is not None:
        response["usage_metadata"] = usage
    return GenerationResponse.from_dict(response)


_USAGE = {"prompt_token_count": 5, "candidates_token_count": 3, "total_token_count": 8}


def _chunks() -> list[GenerationResponse]:
    return [
        _response([{"text": "Hel"}], finish_reason="FINISH_REASON_UNSPECIFIED"),
        _response([{"text": "lo"}], finish_reason="FINISH_REASON_UNSPECIFIED"),
        _response([{"text": "!"}], usage=_USAGE),
    ]


@pytest.fixture
def generate(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock(return_value=_response(usage=_USAGE))
    monkeypatch.setattr(_GenerativeModel, "_generate_content", mock)
    return mock


@pytest.fixture
def generate_stream(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock(side_effect=lambda *a, **k: iter(_chunks()))  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(_GenerativeModel, "_generate_content_streaming", mock)
    return mock


def _only_span(span_exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, [s.name for s in spans]
    return spans[0]


def _attr_json(span: ReadableSpan, key: str) -> Any:
    assert span.attributes is not None
    return json.loads(str(span.attributes[key]))


def test_generate_content(span_exporter: InMemorySpanExporter, generate: MagicMock):
    model = GenerativeModel(
        MODEL,
        system_instruction="Be terse.",
        generation_config=GenerationConfig(temperature=0.3, max_output_tokens=64, top_p=0.9),
    )
    response = model.generate_content("Say hello")
    assert response.candidates[0].text == "Hello there"

    span = _only_span(span_exporter)
    attrs = span.attributes
    assert attrs is not None
    assert span.name == "vertexai.generate_content"
    assert attrs["lmnr.span.type"] == "LLM"
    assert attrs["gen_ai.system"] == "gemini"
    assert attrs["gen_ai.request.model"] == MODEL
    assert attrs["gen_ai.response.model"] == MODEL
    assert attrs["gen_ai.response.id"] == "resp-123"
    assert attrs["gen_ai.request.temperature"] == pytest.approx(0.3)
    assert attrs["gen_ai.request.max_tokens"] == 64
    assert attrs["gen_ai.request.top_p"] == pytest.approx(0.9)
    assert attrs["gen_ai.usage.input_tokens"] == 5
    assert attrs["gen_ai.usage.output_tokens"] == 3
    assert attrs["llm.usage.total_tokens"] == 8
    assert attrs["lmnr.span.instrumentation_scope.name"] == "google-aiplatform"
    assert _attr_json(span, "gen_ai.input.messages") == [
        {"role": "system", "parts": [{"text": "Be terse."}]},
        {"role": "user", "parts": [{"text": "Say hello"}]},
    ]
    output = _attr_json(span, "gen_ai.output.messages")
    assert output[0]["content"] == {"role": "model", "parts": [{"text": "Hello there"}]}
    assert output[0]["finish_reason"] == "STOP"


def test_per_call_generation_config_replaces_model_default(
    span_exporter: InMemorySpanExporter, generate: MagicMock
):
    model = GenerativeModel(MODEL, generation_config=GenerationConfig(temperature=0.3, top_k=5))
    _ = model.generate_content("hi", generation_config={"temperature": 0.9})

    attrs = _only_span(span_exporter).attributes
    assert attrs is not None
    assert attrs["gen_ai.request.temperature"] == pytest.approx(0.9)
    # the SDK does not merge configs, so the model's top_k is not sent either
    assert "gen_ai.request.top_k" not in attrs


def test_multimodal_and_multi_turn_contents(
    span_exporter: InMemorySpanExporter, generate: MagicMock
):
    image = Image.from_bytes(b"\x89PNG fake")
    model = GenerativeModel(MODEL)
    _ = model.generate_content(
        [
            {"role": "user", "parts": [{"text": "Hi"}]},
            {"role": "model", "parts": [{"text": "Hello"}]},
            {"role": "user", "parts": [Part.from_text("Describe"), Part.from_image(image)]},
        ]
    )
    messages = _attr_json(_only_span(span_exporter), "gen_ai.input.messages")
    assert [m["role"] for m in messages] == ["user", "model", "user"]
    assert messages[2]["parts"][0] == {"text": "Describe"}
    inline = messages[2]["parts"][1]["inline_data"]
    assert inline["mime_type"] == Part.from_image(image).inline_data.mime_type
    assert inline["data"] == "iVBORyBmYWtl"  # standard base64 of the bytes


def test_list_of_parts_is_a_single_user_message(
    span_exporter: InMemorySpanExporter, generate: MagicMock
):
    _ = GenerativeModel(MODEL).generate_content(["Describe", Image.from_bytes(b"img")])
    messages = _attr_json(_only_span(span_exporter), "gen_ai.input.messages")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["parts"][0] == {"text": "Describe"}
    assert "inline_data" in messages[0]["parts"][1]


def test_function_call_response_and_tools(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    # `GenerationResponse.text` raises on a function-call part; the
    # instrumentation must not depend on it.
    monkeypatch.setattr(
        _GenerativeModel,
        "_generate_content",
        MagicMock(
            return_value=_response(
                [{"function_call": {"name": "get_weather", "args": {"city": "Paris"}}}]
            )
        ),
    )
    tool = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="get_weather",
                description="Weather for a city",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ]
    )
    _ = GenerativeModel(MODEL, tools=[tool]).generate_content("Weather in Paris?")

    span = _only_span(span_exporter)
    tools = _attr_json(span, "gen_ai.tool.definitions")
    assert tools[0]["name"] == "get_weather"
    output = _attr_json(span, "gen_ai.output.messages")
    assert output[0]["content"]["parts"][0]["function_call"] == {
        "name": "get_weather",
        "args": {"city": "Paris"},
    }


def test_generate_content_streaming(
    span_exporter: InMemorySpanExporter, generate_stream: MagicMock
):
    chunks = list(GenerativeModel(MODEL).generate_content("Say hello", stream=True))
    assert len(chunks) == 3

    span = _only_span(span_exporter)
    attrs = span.attributes
    assert attrs is not None
    assert span.name == "vertexai.generate_content"
    assert attrs["llm.is_streaming"] is True
    assert attrs["gen_ai.usage.input_tokens"] == 5
    assert attrs["gen_ai.usage.output_tokens"] == 3
    output = _attr_json(span, "gen_ai.output.messages")
    assert output[0]["content"]["parts"] == [{"text": "Hello!"}]
    assert output[0]["finish_reason"] == "STOP"


def test_streaming_early_break_ends_span(
    span_exporter: InMemorySpanExporter, generate_stream: MagicMock
):
    stream = GenerativeModel(MODEL).generate_content("Say hello", stream=True)
    first = next(iter(stream))
    assert first.candidates[0].content.parts[0].text == "Hel"
    assert span_exporter.get_finished_spans() == ()
    stream.close()  # pyright: ignore[reportAttributeAccessIssue]

    output = _attr_json(_only_span(span_exporter), "gen_ai.output.messages")
    assert output[0]["content"]["parts"] == [{"text": "Hel"}]


def test_streaming_error_mid_stream(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    def failing_stream(*_args: Any, **_kwargs: Any) -> Iterator[GenerationResponse]:
        yield _chunks()[0]
        raise RuntimeError("stream broke")

    monkeypatch.setattr(_GenerativeModel, "_generate_content_streaming", failing_stream)
    with pytest.raises(RuntimeError, match="stream broke"):
        for _ in GenerativeModel(MODEL).generate_content("hi", stream=True):
            pass

    span = _only_span(span_exporter)
    assert [e.name for e in span.events] == ["exception"]
    assert span.attributes is not None
    assert span.attributes["error.type"] == "RuntimeError"


def test_generate_content_error(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        _GenerativeModel, "_generate_content", MagicMock(side_effect=ValueError("bad request"))
    )
    with pytest.raises(ValueError, match="bad request"):
        _ = GenerativeModel(MODEL).generate_content("hi")

    span = _only_span(span_exporter)
    assert span.status.is_ok is False
    assert span.attributes is not None
    assert span.attributes["error.type"] == "ValueError"
    exception_event = span.events[0]
    assert exception_event.name == "exception"
    assert exception_event.attributes is not None
    assert exception_event.attributes["exception.message"] == "bad request"


def test_llm_span_is_current_during_the_call(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    def generate_with_child(*_args: Any, **_kwargs: Any) -> GenerationResponse:
        with Laminar.start_as_current_span("transport"):
            pass
        return _response()

    monkeypatch.setattr(_GenerativeModel, "_generate_content", generate_with_child)
    with Laminar.start_as_current_span("outer"):
        _ = GenerativeModel(MODEL).generate_content("hi")

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    llm = spans["vertexai.generate_content"]
    assert llm.parent is not None and spans["outer"].context is not None
    assert llm.parent.span_id == spans["outer"].context.span_id
    assert spans["transport"].parent is not None and llm.context is not None
    assert spans["transport"].parent.span_id == llm.context.span_id


@pytest.mark.asyncio
async def test_generate_content_async(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        _GenerativeModel,
        "_generate_content_async",
        AsyncMock(return_value=_response(usage=_USAGE)),
    )
    response = await GenerativeModel(MODEL).generate_content_async("Say hello")
    assert response.candidates[0].text == "Hello there"

    span = _only_span(span_exporter)
    assert span.name == "vertexai.generate_content_async"
    assert span.attributes is not None
    assert span.attributes["gen_ai.usage.input_tokens"] == 5


@pytest.mark.asyncio
async def test_generate_content_async_streaming(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    async def stream() -> AsyncIterator[GenerationResponse]:
        for chunk in _chunks():
            yield chunk

    async def generate_streaming_async(*_args: Any, **_kwargs: Any):
        return stream()

    monkeypatch.setattr(
        _GenerativeModel, "_generate_content_streaming_async", generate_streaming_async
    )
    response = await GenerativeModel(MODEL).generate_content_async("Say hello", stream=True)
    chunks = [chunk async for chunk in response]
    assert len(chunks) == 3

    span = _only_span(span_exporter)
    assert span.attributes is not None
    assert span.attributes["llm.is_streaming"] is True
    output = _attr_json(span, "gen_ai.output.messages")
    assert output[0]["content"]["parts"] == [{"text": "Hello!"}]


@pytest.mark.parametrize("model_class", [GenerativeModel, PreviewGenerativeModel])
def test_chat_session_emits_one_span(
    span_exporter: InMemorySpanExporter,
    generate: MagicMock,
    model_class: type[GenerativeModel],
):
    chat = model_class(MODEL).start_chat()
    _ = chat.send_message("First")
    span_exporter.clear()

    _ = chat.send_message("Second")

    # preview ChatSession subclasses the GA one: still exactly one span
    span = _only_span(span_exporter)
    assert span.name == "vertexai.send_message"
    messages = _attr_json(span, "gen_ai.input.messages")
    assert messages == [
        {"role": "user", "parts": [{"text": "First"}]},
        {"role": "model", "parts": [{"text": "Hello there"}]},
        {"role": "user", "parts": [{"text": "Second"}]},
    ]


def test_chat_session_streaming(span_exporter: InMemorySpanExporter, generate_stream: MagicMock):
    chat = GenerativeModel(MODEL).start_chat()
    chunks = list(chat.send_message("Say hello", stream=True))
    assert len(chunks) == 3

    span = _only_span(span_exporter)
    assert span.attributes is not None
    assert span.attributes["llm.is_streaming"] is True
    output = _attr_json(span, "gen_ai.output.messages")
    assert output[0]["content"]["parts"] == [{"text": "Hello!"}]


@pytest.mark.asyncio
async def test_chat_session_async(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        _GenerativeModel, "_generate_content_async", AsyncMock(return_value=_response())
    )
    chat = GenerativeModel(MODEL).start_chat()
    _ = await chat.send_message_async("hi")

    span = _only_span(span_exporter)
    assert span.name == "vertexai.send_message_async"


def test_suppressed_instrumentation(span_exporter: InMemorySpanExporter, generate: MagicMock):
    from opentelemetry import context as context_api

    token = context_api.attach(
        context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
        _ = GenerativeModel(MODEL).generate_content("hi")
    finally:
        context_api.detach(token)
    assert span_exporter.get_finished_spans() == ()


# --- legacy language models ---------------------------------------------------


def _text_prediction(text: str = "Bonjour") -> Prediction:
    return Prediction(
        predictions=[{"content": text, "safetyAttributes": {}}],
        deployed_model_id="",
        metadata={
            "tokenMetadata": {
                "inputTokenCount": {"totalTokens": 4},
                "outputTokenCount": {"totalTokens": 2},
            }
        },
    )


@pytest.fixture
def endpoint() -> MagicMock:
    endpoint = MagicMock()
    endpoint.predict.return_value = _text_prediction()
    endpoint.predict_async = AsyncMock(return_value=_text_prediction())
    return endpoint


def _text_model(endpoint: MagicMock) -> TextGenerationModel:
    model = TextGenerationModel(model_id="text-bison@002", endpoint_name="endpoint")
    model._endpoint = endpoint
    return model


def test_text_generation_predict(span_exporter: InMemorySpanExporter, endpoint: MagicMock):
    response = _text_model(endpoint).predict(
        "Translate hello", temperature=0.2, max_output_tokens=32
    )
    assert response.text == "Bonjour"

    span = _only_span(span_exporter)
    attrs = span.attributes
    assert attrs is not None
    assert span.name == "vertexai.predict"
    assert attrs["gen_ai.request.model"] == "text-bison@002"
    assert attrs["gen_ai.request.temperature"] == pytest.approx(0.2)
    assert attrs["gen_ai.request.max_tokens"] == 32
    assert attrs["gen_ai.usage.input_tokens"] == 4
    assert attrs["gen_ai.usage.output_tokens"] == 2
    assert attrs["llm.usage.total_tokens"] == 6
    assert _attr_json(span, "gen_ai.input.messages") == [
        {"role": "user", "parts": [{"text": "Translate hello"}]}
    ]
    assert _attr_json(span, "gen_ai.output.messages") == [
        {"content": {"role": "model", "parts": [{"text": "Bonjour"}]}}
    ]


@pytest.mark.asyncio
async def test_text_generation_predict_async(
    span_exporter: InMemorySpanExporter, endpoint: MagicMock
):
    response = await _text_model(endpoint).predict_async("Translate hello")
    assert response.text == "Bonjour"
    assert _only_span(span_exporter).name == "vertexai.predict_async"


def test_text_generation_predict_streaming(
    span_exporter: InMemorySpanExporter, endpoint: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        language_models_module._streaming_prediction,  # pyright: ignore[reportPrivateImportUsage]
        "predict_stream_of_dicts_from_single_dict",
        lambda **_kwargs: iter([{"content": "Bon"}, {"content": "jour"}]),  # pyright: ignore[reportUnknownLambdaType]
    )
    chunks = [c.text for c in _text_model(endpoint).predict_streaming("Translate hello")]
    assert chunks == ["Bon", "jour"]

    span = _only_span(span_exporter)
    assert span.name == "vertexai.predict_streaming"
    assert span.attributes is not None
    assert span.attributes["llm.is_streaming"] is True
    assert _attr_json(span, "gen_ai.output.messages") == [
        {"content": {"role": "model", "parts": [{"text": "Bonjour"}]}}
    ]


@pytest.mark.asyncio
async def test_text_generation_predict_streaming_async(
    span_exporter: InMemorySpanExporter, endpoint: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    async def stream(**_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        for content in ("Bon", "jour"):
            yield {"content": content}

    monkeypatch.setattr(
        language_models_module._streaming_prediction,  # pyright: ignore[reportPrivateImportUsage]
        "predict_stream_of_dicts_from_single_dict_async",
        stream,
    )
    model = _text_model(endpoint)
    chunks = [c.text async for c in model.predict_streaming_async("Translate hello")]
    assert chunks == ["Bon", "jour"]

    span = _only_span(span_exporter)
    assert span.name == "vertexai.predict_streaming_async"
    assert _attr_json(span, "gen_ai.output.messages") == [
        {"content": {"role": "model", "parts": [{"text": "Bonjour"}]}}
    ]


def test_language_chat_session(span_exporter: InMemorySpanExporter, endpoint: MagicMock):
    endpoint.predict.return_value = Prediction(
        predictions=[
            {"candidates": [{"author": "bot", "content": "Hi!"}], "safetyAttributes": [{}]}
        ],
        deployed_model_id="",
        metadata={},
    )
    model = ChatModel(model_id="chat-bison@002", endpoint_name="endpoint")
    model._endpoint = endpoint
    chat = model.start_chat(context="You are a pirate.")
    _ = chat.send_message("Hello")
    span_exporter.clear()

    _ = chat.send_message("How are you?")

    span = _only_span(span_exporter)
    assert span.name == "vertexai.send_message"
    assert span.attributes is not None
    assert span.attributes["gen_ai.request.model"] == "chat-bison@002"
    assert _attr_json(span, "gen_ai.input.messages") == [
        {"role": "system", "parts": [{"text": "You are a pirate."}]},
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi!"}]},
        {"role": "user", "parts": [{"text": "How are you?"}]},
    ]


def test_uninstrument_removes_wrappers():
    from vertexai.generative_models._generative_models import ChatSession

    instrumentor = GoogleAIPlatformInstrumentor()
    assert instrumentor.is_instrumented_by_opentelemetry
    assert is_wrapt_proxy(_GenerativeModel.__dict__["generate_content"])
    try:
        instrumentor.uninstrument()
        assert not is_wrapt_proxy(_GenerativeModel.__dict__["generate_content"])
        assert not is_wrapt_proxy(ChatSession.__dict__["send_message"])
        assert not is_wrapt_proxy(language_models_module._TextGenerationModel.__dict__["predict"])
    finally:
        instrumentor.instrument()
    assert is_wrapt_proxy(_GenerativeModel.__dict__["generate_content"])

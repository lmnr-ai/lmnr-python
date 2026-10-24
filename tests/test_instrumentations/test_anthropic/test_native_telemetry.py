"""anthropic >= 1.14's native OpenTelemetry spans, and the stream / error paths
that changed with the move to the 2.x dev pin.

The native span goes through the global tracer provider, i.e. Laminar's
`LaminarSpanProcessor`, which keeps it only under
`ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE=content` (and then our wrappers
are not installed). See `tracing/anthropic_native.py`. `MockTransport`, no
cassettes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import cast

import anthropic
import httpx2
import pytest
from _wrapt_compat import is_wrapt_proxy
from anthropic import Anthropic, AsyncAnthropic
from anthropic.lib.streaming import AsyncMessageStream, MessageStream
from anthropic.resources.messages import Messages
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic import (
    AnthropicInstrumentor,
)
from lmnr.opentelemetry_lib.tracing.anthropic_native import (
    is_anthropic_native_scope,
    native_spans_requested,
    use_native_spans,
)

from .test_raw_response import MESSAGES, MODEL, _message, _output_text, _sse

CONTENT_MODE = "ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE"


def _transport(status: int = 200) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if status != 200:
            return httpx2.Response(
                status, json={"type": "error", "error": {"type": "api_error", "message": "boom"}}
            )
        if json.loads(request.content).get("stream"):
            return httpx2.Response(
                200, content=_sse("Paris"), headers={"content-type": "text/event-stream"}
            )
        return httpx2.Response(200, json=_message("Paris", "msg_live"))

    return httpx2.MockTransport(handler)


def _client(status: int = 200) -> Anthropic:
    return Anthropic(
        api_key="test", max_retries=0, http_client=httpx2.Client(transport=_transport(status))
    )


def _async_client(status: int = 200) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key="test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=_transport(status)),
    )


def _only_span(span_exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["anthropic.chat"]
    return spans[0]


def _assert_error(span: ReadableSpan, error_type: str) -> None:
    # app-server marks a span as failed only when it has an `exception` event.
    assert [event.name for event in span.events] == ["exception"]
    assert (span.attributes or {})["error.type"] == error_type


@pytest.fixture
def default_mode(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(CONTENT_MODE, raising=False)
    monkeypatch.delenv("ANTHROPIC_OPEN_TELEMETRY", raising=False)
    native_spans_requested.cache_clear()
    yield
    native_spans_requested.cache_clear()


@pytest.fixture
def content_mode(monkeypatch: pytest.MonkeyPatch) -> Iterator[AnthropicInstrumentor]:
    """Install the instrumentor the way `Laminar.initialize` would, with the
    native spans opted into."""
    instrumentor = AnthropicInstrumentor()
    was_instrumented = instrumentor.is_instrumented_by_opentelemetry
    if was_instrumented:
        instrumentor.uninstrument()
    monkeypatch.setenv(CONTENT_MODE, "content")
    native_spans_requested.cache_clear()
    instrumentor.instrument()

    yield instrumentor

    instrumentor.uninstrument()
    monkeypatch.delenv(CONTENT_MODE)
    native_spans_requested.cache_clear()
    if was_instrumented:
        instrumentor.instrument()


# --- which spans are kept -----------------------------------------------------


def test_native_spans_are_dropped_by_default(
    default_mode: None, instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    _ = _client().messages.create(model=MODEL, max_tokens=10, messages=MESSAGES)

    # The SDK emits `anthropic.messages.create` too; counting it would bill the
    # call twice.
    assert _output_text(_only_span(span_exporter)) == "Paris"


def test_content_mode_uses_the_native_spans(
    content_mode: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    assert use_native_spans()
    assert not is_wrapt_proxy(Messages.create)
    # Still reported as instrumented, so LangChain hands `ChatAnthropic` runs
    # off (no LangChain LLM span next to the native one).
    assert content_mode.is_instrumented_by_opentelemetry

    _ = _client().messages.create(model=MODEL, max_tokens=10, messages=MESSAGES)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["anthropic.messages.create"]
    span = spans[0]
    assert span.instrumentation_scope is not None
    assert is_anthropic_native_scope(span.instrumentation_scope.name)
    attributes = span.attributes or {}
    # app-server reads the provider from `gen_ai.system`; the SDK only sets
    # `gen_ai.provider.name`.
    assert attributes["gen_ai.system"] == "anthropic"
    assert attributes["lmnr.span.path"] == ("anthropic.messages.create",)
    output = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert output[0]["parts"][0]["content"] == "Paris"


@pytest.mark.parametrize(
    ("env", "requested"),
    [
        ({}, False),
        ({CONTENT_MODE: "metadata_only"}, False),
        ({CONTENT_MODE: "content"}, True),
        ({CONTENT_MODE: "CONTENT"}, True),
        ({CONTENT_MODE: "content", "ANTHROPIC_OPEN_TELEMETRY": "false"}, False),
    ],
)
def test_native_spans_requested(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], requested: bool
):
    monkeypatch.delenv(CONTENT_MODE, raising=False)
    monkeypatch.delenv("ANTHROPIC_OPEN_TELEMETRY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    native_spans_requested.cache_clear()
    try:
        assert native_spans_requested() is requested
    finally:
        native_spans_requested.cache_clear()


# --- `messages.stream()` ------------------------------------------------------


def test_stream_manager_returns_the_message_stream(
    default_mode: None, instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    with _client().messages.stream(model=MODEL, max_tokens=10, messages=MESSAGES) as stream:
        # The SDK's helpers (`text_stream`, `get_final_message`, ...) must work
        # on what the context manager hands back.
        assert isinstance(stream, MessageStream)
        assert "".join(stream.text_stream) == "Paris"
        assert stream.get_final_message().content[0].text == "Paris"  # pyright: ignore[reportAttributeAccessIssue]

    span = _only_span(span_exporter)
    assert _output_text(span) == "Paris"
    assert (span.attributes or {})["gen_ai.usage.output_tokens"] == 3


@pytest.mark.asyncio
async def test_async_stream_manager_returns_the_message_stream(
    default_mode: None, instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    async with _async_client().messages.stream(
        model=MODEL, max_tokens=10, messages=MESSAGES
    ) as stream:
        assert isinstance(stream, AsyncMessageStream)
        assert "".join([text async for text in stream.text_stream]) == "Paris"
        message = await stream.get_final_message()
        assert message.content[0].text == "Paris"  # pyright: ignore[reportAttributeAccessIssue]

    assert _output_text(_only_span(span_exporter)) == "Paris"


def test_stream_manager_until_done(
    default_mode: None, instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    with _client().messages.stream(model=MODEL, max_tokens=10, messages=MESSAGES) as stream:
        stream.until_done()

    assert _output_text(_only_span(span_exporter)) == "Paris"


def test_raw_stream_message_stop_has_no_message(
    default_mode: None, instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    # `create(stream=True)` yields a `message_stop` event without `message`.
    events = list(
        _client().messages.create(model=MODEL, max_tokens=10, messages=MESSAGES, stream=True)
    )

    assert events[-1].type == "message_stop"
    span = _only_span(span_exporter)
    assert _output_text(span) == "Paris"
    assert (span.attributes or {})["gen_ai.response.id"] == "msg_live"


# --- errors -------------------------------------------------------------------


def test_api_error_is_recorded(
    default_mode: None, instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    with pytest.raises(anthropic.InternalServerError):
        _ = _client(500).messages.create(model=MODEL, max_tokens=10, messages=MESSAGES)

    _assert_error(_only_span(span_exporter), "InternalServerError")


@pytest.mark.asyncio
async def test_async_api_error_is_recorded(
    default_mode: None, instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    with pytest.raises(anthropic.BadRequestError):
        _ = await _async_client(400).messages.create(model=MODEL, max_tokens=10, messages=MESSAGES)

    _assert_error(_only_span(span_exporter), "BadRequestError")


def test_stream_manager_error_is_recorded(
    default_mode: None, instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    stream = _client(500).messages.stream(model=MODEL, max_tokens=10, messages=MESSAGES)
    with pytest.raises(anthropic.InternalServerError), stream:
        pass

    _assert_error(_only_span(span_exporter), "InternalServerError")


@pytest.mark.asyncio
async def test_async_stream_manager_error_is_recorded(
    default_mode: None, instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    with pytest.raises(anthropic.InternalServerError):
        async with _async_client(500).messages.stream(
            model=MODEL, max_tokens=10, messages=MESSAGES
        ):
            pass

    _assert_error(_only_span(span_exporter), "InternalServerError")


def test_error_inside_the_stream_block_is_recorded(
    default_mode: None, instrumentor: AnthropicInstrumentor, span_exporter: InMemorySpanExporter
):
    stream = _client().messages.stream(model=MODEL, max_tokens=10, messages=MESSAGES)
    with pytest.raises(ValueError), stream:
        raise ValueError("caller failed")

    _assert_error(_only_span(span_exporter), "ValueError")

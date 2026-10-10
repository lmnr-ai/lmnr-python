import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext

from lmnr import observe


def _ctx(span: ReadableSpan) -> SpanContext:
    ctx = span.get_span_context()
    assert ctx is not None
    return ctx


def _parent(span: ReadableSpan) -> SpanContext:
    assert span.parent is not None
    return span.parent


@pytest.mark.vcr
def test_openai_completion(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = OpenAI(api_key="test-key")
    _result = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "openai.chat"
    assert (spans[0].attributes or {})["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].name == "openai.chat"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("openai.chat",)


@pytest.mark.vcr
def test_openai_completion_in_observe(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = OpenAI(api_key="test-123")

    @observe()
    def foo() -> ChatCompletion:
        return client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

    _result = foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    openai_span = next(span for span in spans if span.name == "openai.chat")
    outer_span = next(span for span in spans if span.name == "foo")
    assert openai_span.name == "openai.chat"
    assert (openai_span.attributes or {})["gen_ai.request.model"] == "gpt-4.1-nano"
    assert openai_span.name == "openai.chat"
    assert (openai_span.attributes or {})["lmnr.span.path"] == (
        "foo",
        "openai.chat",
    )

    assert outer_span.parent is None or outer_span.parent.span_id == 0
    assert _parent(openai_span).span_id == _ctx(outer_span).span_id
    assert _parent(openai_span).trace_id == _ctx(outer_span).trace_id


@pytest.mark.vcr
def test_openai_multiple_requests_in_observe(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = OpenAI(api_key="test-123")

    @observe()
    def foo() -> ChatCompletion:
        _res = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )
        return client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

    _res = foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    openai_span = next(span for span in spans if span.name == "openai.chat")
    openai_span2 = [span for span in spans if span.name == "openai.chat"][1]
    outer_span = next(span for span in spans if span.name == "foo")
    assert openai_span.name == "openai.chat"
    assert (openai_span.attributes or {})["gen_ai.request.model"] == "gpt-4.1-nano"
    assert openai_span.name == "openai.chat"
    assert (openai_span.attributes or {})["lmnr.span.path"] == (
        "foo",
        "openai.chat",
    )
    assert (openai_span2.attributes or {})["lmnr.span.path"] == (
        "foo",
        "openai.chat",
    )

    assert outer_span.parent is None or outer_span.parent.span_id == 0
    assert _parent(openai_span).span_id == _ctx(outer_span).span_id
    assert _parent(openai_span).trace_id == _ctx(outer_span).trace_id

    assert _parent(openai_span2).span_id == _ctx(outer_span).span_id
    assert _parent(openai_span2).trace_id == _ctx(outer_span).trace_id


@pytest.mark.vcr
def test_openai_completion_after_observe(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = OpenAI(api_key="test-123")

    @observe()
    def foo() -> str:
        return "foo"

    _str = foo()

    _res = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    first_span = next(span for span in spans if span.name == "foo")
    openai_span = next(span for span in spans if span.name == "openai.chat")
    assert openai_span.name == "openai.chat"
    assert (openai_span.attributes or {})["gen_ai.request.model"] == "gpt-4.1-nano"
    assert openai_span.name == "openai.chat"
    assert (openai_span.attributes or {})["lmnr.span.path"] == ("openai.chat",)

    assert first_span.parent is None or first_span.parent.span_id == 0
    assert openai_span.parent is None or openai_span.parent.span_id == 0
    assert (
        _ctx(openai_span).trace_id
        != _ctx(first_span).trace_id
    )

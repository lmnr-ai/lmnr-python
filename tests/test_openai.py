import pytest

from lmnr import observe
from openai import OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.vcr
def test_openai_completion(exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = OpenAI(api_key="test-key")
    client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "openai.chat"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].name == "openai.chat"
    assert spans[0].attributes["lmnr.span.path"] == ("openai.chat",)


@pytest.mark.vcr
def test_openai_completion_in_observe(exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = OpenAI(api_key="test-123")

    @observe()
    def foo():
        return client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

    foo()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    openai_span = [span for span in spans if span.name == "openai.chat"][0]
    outer_span = [span for span in spans if span.name == "foo"][0]
    assert openai_span.name == "openai.chat"
    assert openai_span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert openai_span.name == "openai.chat"
    assert openai_span.attributes["lmnr.span.path"] == (
        "foo",
        "openai.chat",
    )

    assert outer_span.parent is None or outer_span.parent.span_id == 0
    assert openai_span.parent.span_id == outer_span.get_span_context().span_id
    assert openai_span.parent.trace_id == outer_span.get_span_context().trace_id

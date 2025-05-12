import pytest

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

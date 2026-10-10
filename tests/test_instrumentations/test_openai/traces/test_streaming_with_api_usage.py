from typing import cast

import pytest
from openai import OpenAI
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def api_usage_provider_client():
    """Client for testing API providers that include usage information in streaming responses, use deepseek here"""
    return OpenAI(api_key="test-api-key", base_url="https://api.deepseek.com/beta")


@pytest.mark.vcr
def test_streaming_with_api_usage_capture(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    api_usage_provider_client,
):
    """Test that streaming responses with API usage information are properly captured"""
    response = api_usage_provider_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    response_content = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            response_content += chunk.choices[0].delta.content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    attributes = span.attributes or {}
    attributes = span.attributes or {}
    assert span.name == "openai.chat"

    # Check that token usage is captured from API response
    assert cast(int, attributes.get("gen_ai.usage.input_tokens")) > 0
    assert cast(int, attributes.get("gen_ai.usage.output_tokens")) > 0

    # Verify that the response content is meaningful
    assert len(response_content) > 0
    assert attributes.get("gen_ai.response.model") == "deepseek-chat"

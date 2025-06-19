import json
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
    assert spans[0].attributes["lmnr.span.path"] == ("openai.chat",)


@pytest.mark.vcr
def test_openai_responses(exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = OpenAI(api_key="test-key")
    response = client.responses.create(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
    )
    # assert response
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openai.responses"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert span.attributes["lmnr.span.path"] == ("openai.responses",)
    assert (
        span.attributes["gen_ai.prompt.0.content"] == "What is the capital of France?"
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert span.attributes["gen_ai.completion.0.content"] == response.output_text
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"


@pytest.mark.vcr
def test_openai_responses_with_input_history(exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = OpenAI(api_key="test-key")
    user_message = "Come up with an adjective in English. Respond with just one word."
    first_response = client.responses.create(
        model="gpt-4.1-nano",
        input=user_message,
    )
    response = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {
                "role": "user",
                "content": user_message,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "input_text",
                        "text": first_response.output_text,
                    }
                ],
            },
            {"role": "user", "content": "Can you explain why you chose that word?"},
        ],
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    span = spans[1]
    assert span.name == "openai.responses"
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert span.attributes["lmnr.span.path"] == ("openai.responses",)
    assert (
        span.attributes["gen_ai.prompt.0.content"]
        == "Come up with an adjective in English. Respond with just one word."
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert json.loads(span.attributes["gen_ai.prompt.1.content"]) == [
        {
            "type": "input_text",
            "text": first_response.output_text,
        }
    ]
    assert span.attributes["gen_ai.prompt.1.role"] == "assistant"
    assert (
        span.attributes["gen_ai.prompt.2.content"]
        == "Can you explain why you chose that word?"
    )
    assert span.attributes["gen_ai.prompt.2.role"] == "user"
    assert span.attributes["gen_ai.completion.0.content"] == response.output_text
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"

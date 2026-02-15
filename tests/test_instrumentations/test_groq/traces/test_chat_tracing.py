import base64
import json
import pytest

from pathlib import Path
from opentelemetry.semconv_ai import SpanAttributes
from groq import Groq


image_base64 = base64.b64encode(
    open(
        Path(__file__).parent.parent.joinpath("data/logo.jpg"),
        "rb",
    ).read()
).decode("utf-8")


image_content_block = {
    "type": "image_url",
    "image_url": {
        "url": f"data:image/jpeg;base64,{image_base64}",
        "detail": "high",
    },
}


@pytest.mark.vcr
def test_chat_legacy(instrument_legacy, groq_client, span_exporter):
    groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert (
        groq_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert groq_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) > 0
    assert (
        groq_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-645691ff-34af-4d0f-a1c1-fe888f8685cc"
    )


@pytest.mark.vcr
def test_chat_legacy_image(instrument_legacy, groq_client: Groq, span_exporter):
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-guard-4-12b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is depicted in this image?"},
                    image_content_block,
                ],
            }
        ],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert json.loads(
        groq_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
    ) == [
        {"type": "text", "text": "What is depicted in this image?"},
        image_content_block,
    ]
    assert (
        groq_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.choices[0].message.content
    )
    assert (
        groq_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role")
        == "assistant"
    )
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) > 0
    assert (
        groq_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-741dc0bb-6df1-4e88-9d35-920d0dddc3c0"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chat_legacy(instrument_legacy, async_groq_client, span_exporter):
    await async_groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert (
        groq_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert groq_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS) > 0
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) > 0
    assert (
        groq_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-ec0a74e9-df7f-4e91-aa09-e9618451f5c9"
    )


@pytest.mark.vcr
def test_chat_streaming_legacy(instrument_legacy, groq_client, span_exporter):
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    content = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "groq.chat",
    ]
    groq_span = spans[0]
    assert (
        groq_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        groq_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == content
    )
    assert groq_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 18
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS) == 73
    assert groq_span.attributes.get(SpanAttributes.LLM_USAGE_TOTAL_TOKENS) == 91

import base64
import json

import pytest
import requests
from typing import cast
from openai import OpenAI
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.vcr
def test_vision(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://source.unsplash.com/8xznAGy4HcY/800x400"
                        },
                    },
                ],
            }
        ],
    )

    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://source.unsplash.com/8xznAGy4HcY/800x400"},
        },
    ]

    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert output_messages[0]["message"]["content"]
    assert (
        attributes["gen_ai.request.base_url"]
        == "https://api.openai.com/v1/"
    )
    assert (
        attributes.get("gen_ai.response.id")
        == "chatcmpl-8wq4EsSXTQC0JbGzob3SBHg6pS7Tt"
    )


@pytest.mark.vcr
def test_vision_base64(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    # Fetch the image from the URL
    response = requests.get(
        "https://upload.wikimedia.org/wikipedia/commons/"
        "thumb/d/dd/"
        "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
        "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    )
    image_data = response.content

    # Encode the image data to base64
    base64_image = base64.b64encode(image_data).decode("utf-8")

    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )

    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"][0] == {
        "type": "text",
        "text": "What is in this image?",
    }
    assert input_messages[0]["content"][1]["type"] == "image_url"
    assert input_messages[0]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )

    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert output_messages[0]["message"]["content"]
    assert (
        attributes["gen_ai.request.base_url"]
        == "https://api.openai.com/v1/"
    )
    assert (
        attributes.get("gen_ai.response.id")
        == "chatcmpl-AC7YAG2uy8c4VfbqJp4QkdHc5PDZ4"
    )

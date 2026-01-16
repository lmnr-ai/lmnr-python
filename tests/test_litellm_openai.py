import asyncio
import json
import litellm
import os
from opentelemetry.sdk.trace import ReadableSpan
import pytest
import time

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.litellm import LaminarLiteLLMCallback
from lmnr import Laminar

from pydantic import BaseModel

SLEEP_TO_FLUSH_SECONDS = 0.05

BASE64_IMAGE = ""
with open("tests/data/base64_png_blank_1024_768.txt", "r") as f:
    BASE64_IMAGE = f.read().strip()

EVENT_JSON_SCHEMA = {
    "name": "event",
    "schema": {
        "type": "object",
        "title": "Event",
        "properties": {
            "name": {"type": "string", "title": "Name"},
            "people": {
                "type": "array",
                "items": {"type": "string"},
                "title": "People",
            },
            "dayOfWeek": {"type": "string", "title": "Dayofweek"},
        },
        "required": ["name", "people", "dayOfWeek"],
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get the news in a given city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    },
]

TOOLS_FOR_RESPONSES = [{"type": "function", **tool["function"]} for tool in TOOLS]


class Event(BaseModel):
    name: str
    people: list[str]
    dayOfWeek: str


@pytest.mark.vcr
def test_litellm_openai_basic(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    # Wait for the callback to complete and flush the spans
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.usage.input_tokens"] == 14
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["llm.usage.total_tokens"] == 21
    assert (
        span.attributes["gen_ai.prompt.0.content"] == "What is the capital of France?"
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        span.attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.system"] == "openai"


@pytest.mark.vcr
def test_litellm_openai_with_metadata(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        metadata={
            "tags": ["test"],
            "user_id": "test_user_id",
            "session_id": "test_session_id",
        },
    )

    # Wait for the callback to complete and flush the spans
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.usage.input_tokens"] == 14
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["llm.usage.total_tokens"] == 21
    assert (
        span.attributes["gen_ai.prompt.0.content"] == "What is the capital of France?"
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        span.attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["lmnr.association.properties.tags"] == ("test",)
    assert span.attributes["lmnr.association.properties.user_id"] == "test_user_id"
    assert (
        span.attributes["lmnr.association.properties.session_id"] == "test_session_id"
    )


@pytest.mark.vcr
def test_litellm_openai_text_block(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of France?"}],
            }
        ],
    )

    # Wait for the callback to complete and flush the spans
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.usage.input_tokens"] == 14
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["llm.usage.total_tokens"] == 21
    assert json.loads(span.attributes["gen_ai.prompt.0.content"]) == [
        {"type": "text", "text": "What is the capital of France?"}
    ]
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        span.attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.system"] == "openai"


@pytest.mark.vcr
def test_litellm_openai_with_streaming(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
    )

    final_response = ""
    for chunk in response:
        final_response += chunk.choices[0].delta.content or ""

    # Wait for the callback to complete and flush the spans
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.usage.input_tokens"] == 14
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["llm.usage.total_tokens"] == 21
    assert (
        span.attributes["gen_ai.prompt.0.content"] == "What is the capital of France?"
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert span.attributes["gen_ai.completion.0.content"] == final_response
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.system"] == "openai"


@pytest.mark.vcr
def test_litellm_openai_with_streaming_and_metadata(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
        metadata={
            "tags": ["test"],
            "user_id": "test_user_id",
            "session_id": "test_session_id",
        },
    )

    final_response = ""
    for chunk in response:
        final_response += chunk.choices[0].delta.content or ""

    # Wait for the callback to complete and flush the spans
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.usage.input_tokens"] == 14
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["llm.usage.total_tokens"] == 21
    assert (
        span.attributes["gen_ai.prompt.0.content"] == "What is the capital of France?"
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert span.attributes["gen_ai.completion.0.content"] == final_response
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["lmnr.association.properties.tags"] == ("test",)
    assert span.attributes["lmnr.association.properties.user_id"] == "test_user_id"
    assert (
        span.attributes["lmnr.association.properties.session_id"] == "test_session_id"
    )


@pytest.mark.vcr
def test_litellm_openai_with_chat_history(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"
    litellm.callbacks = [litellm_callback]
    user_prompt = (
        "Generate a random adjective in English. Respond with only the adjective."
    )
    first_response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    second_response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            },
            first_response.choices[0].message,
            {"role": "user", "content": "Generate a sentence using the adjective."},
        ],
    )

    # Wait for the callback to complete and flush the spans
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    sorted_spans: list[ReadableSpan] = sorted(list(spans), key=lambda s: s.start_time)
    first_span = sorted_spans[0]
    second_span = sorted_spans[1]
    assert first_span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert second_span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"

    assert first_span.attributes["gen_ai.response.id"] == first_response.id
    assert second_span.attributes["gen_ai.response.id"] == second_response.id
    assert first_span.attributes["gen_ai.system"] == "openai"
    assert second_span.attributes["gen_ai.system"] == "openai"

    assert first_span.attributes["gen_ai.prompt.0.content"] == user_prompt
    assert first_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert second_span.attributes["gen_ai.prompt.0.content"] == user_prompt
    assert second_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        second_span.attributes["gen_ai.prompt.1.content"]
        == first_response.choices[0].message.content
    )
    assert second_span.attributes["gen_ai.prompt.1.role"] == "assistant"
    assert (
        second_span.attributes["gen_ai.prompt.2.content"]
        == "Generate a sentence using the adjective."
    )
    assert second_span.attributes["gen_ai.prompt.2.role"] == "user"

    assert (
        first_span.attributes["gen_ai.completion.0.content"]
        == first_response.choices[0].message.content
    )
    assert first_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert (
        second_span.attributes["gen_ai.completion.0.content"]
        == second_response.choices[0].message.content
    )
    assert second_span.attributes["gen_ai.completion.0.role"] == "assistant"


@pytest.mark.vcr
def test_litellm_openai_with_chat_history_and_tools(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"
    litellm.callbacks = [litellm_callback]

    user_prompt = "What is the weather in San Francisco? Also, any news in town?"
    first_response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        tools=TOOLS,
    )

    second_response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            },
            first_response.choices[0].message,
            {
                "role": "tool",
                "content": "Sunny as always!",
                "tool_call_id": first_response.choices[0].message.tool_calls[0].id,
            },
            {
                "role": "tool",
                "content": "The news in San Francisco is that the Giants are winning!",
                "tool_call_id": first_response.choices[0].message.tool_calls[1].id,
            },
        ],
        tools=TOOLS,
    )

    # Wait for the callback to complete and flush the spans
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    sorted_spans: list[ReadableSpan] = sorted(list(spans), key=lambda s: s.start_time)
    first_span = sorted_spans[0]
    second_span = sorted_spans[1]
    assert first_span.name == "litellm.completion"
    assert second_span.name == "litellm.completion"
    assert first_span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert second_span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"

    assert first_span.attributes["gen_ai.response.id"] == first_response.id
    assert second_span.attributes["gen_ai.response.id"] == second_response.id
    assert first_span.attributes["gen_ai.system"] == "openai"
    assert second_span.attributes["gen_ai.system"] == "openai"

    assert first_span.attributes["gen_ai.prompt.0.content"] == user_prompt
    assert first_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert second_span.attributes["gen_ai.prompt.0.content"] == user_prompt
    assert second_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert second_span.attributes["gen_ai.prompt.1.tool_calls.0.name"] == "get_weather"
    assert json.loads(
        second_span.attributes["gen_ai.prompt.1.tool_calls.0.arguments"]
    ) == {"city": "San Francisco"}
    assert (
        second_span.attributes["gen_ai.prompt.1.tool_calls.0.id"]
        == first_response.choices[0].message.tool_calls[0]["id"]
    )
    assert second_span.attributes["gen_ai.prompt.1.tool_calls.1.name"] == "get_news"
    assert json.loads(
        second_span.attributes["gen_ai.prompt.1.tool_calls.1.arguments"]
    ) == {"city": "San Francisco"}
    assert (
        second_span.attributes["gen_ai.prompt.1.tool_calls.1.id"]
        == first_response.choices[0].message.tool_calls[1]["id"]
    )

    assert second_span.attributes["gen_ai.prompt.1.role"] == "assistant"
    assert second_span.attributes["gen_ai.prompt.2.content"] == "Sunny as always!"
    assert second_span.attributes["gen_ai.prompt.2.role"] == "tool"
    assert (
        second_span.attributes["gen_ai.prompt.3.content"]
        == "The news in San Francisco is that the Giants are winning!"
    )
    assert second_span.attributes["gen_ai.prompt.3.role"] == "tool"

    assert (
        first_span.attributes["gen_ai.completion.0.tool_calls.0.name"] == "get_weather"
    )
    assert json.loads(
        first_span.attributes["gen_ai.completion.0.tool_calls.0.arguments"]
    ) == {"city": "San Francisco"}
    assert (
        first_span.attributes["gen_ai.completion.0.tool_calls.0.id"]
        == first_response.choices[0].message.tool_calls[0]["id"]
    )
    assert first_span.attributes["gen_ai.completion.0.tool_calls.1.name"] == "get_news"
    assert json.loads(
        first_span.attributes["gen_ai.completion.0.tool_calls.1.arguments"]
    ) == {"city": "San Francisco"}
    assert (
        first_span.attributes["gen_ai.completion.0.tool_calls.1.id"]
        == first_response.choices[0].message.tool_calls[1]["id"]
    )
    assert first_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert (
        second_span.attributes["gen_ai.completion.0.content"]
        == second_response.choices[0].message.content
    )
    assert second_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert (
        first_span.attributes["llm.request.functions.0.name"]
        == TOOLS[0]["function"]["name"]
    )
    assert (
        first_span.attributes["llm.request.functions.0.description"]
        == TOOLS[0]["function"]["description"]
    )
    assert json.loads(first_span.attributes["llm.request.functions.0.parameters"]) == (
        TOOLS[0]["function"]["parameters"]
    )
    assert (
        first_span.attributes["llm.request.functions.1.name"]
        == TOOLS[1]["function"]["name"]
    )
    assert (
        first_span.attributes["llm.request.functions.1.description"]
        == TOOLS[1]["function"]["description"]
    )
    assert json.loads(first_span.attributes["llm.request.functions.1.parameters"]) == (
        TOOLS[1]["function"]["parameters"]
    )
    assert first_span.attributes["gen_ai.usage.input_tokens"] == 82
    assert first_span.attributes["gen_ai.usage.output_tokens"] == 46
    assert first_span.attributes["llm.usage.total_tokens"] == 128
    assert second_span.attributes["gen_ai.usage.input_tokens"] == 157
    assert second_span.attributes["gen_ai.usage.output_tokens"] == 22
    assert second_span.attributes["llm.usage.total_tokens"] == 179


@pytest.mark.vcr
def test_litellm_openai_with_image_base64(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    # Sample image URL and base64 encoding
    with open("tests/data/base64_image.txt", "r") as f:
        image_data = f.read().strip()
    image_media_type = "image/jpeg"

    litellm.callbacks = [litellm_callback]
    user_prompt = "What's depicted in this image?"
    response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_media_type};base64,{image_data}"
                        },
                    },
                ],
            }
        ],
    )

    # Wait for the callback to complete and flush the spans
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        span.attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.system"] == "openai"
    # Check that the content includes both text and image
    prompt_content = json.loads(span.attributes["gen_ai.prompt.0.content"])
    assert len(prompt_content) == 2
    assert prompt_content[0]["type"] == "text"
    assert prompt_content[0]["text"] == user_prompt
    assert prompt_content[1]["type"] == "image_url"


@pytest.mark.vcr
def test_litellm_openai_with_image_url(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    # Direct image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/8/8e/MuseumOfFineArtsBoston_CopleySquare_19thc.jpg"

    litellm.callbacks = [litellm_callback]
    user_prompt = "What's depicted in this image?"
    response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
    )

    # Wait for the callback to complete and flush the spans
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        span.attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.system"] == "openai"
    # Check that the content includes both text and image
    prompt_content = json.loads(span.attributes["gen_ai.prompt.0.content"])
    assert len(prompt_content) == 2
    assert prompt_content[0]["type"] == "text"
    assert prompt_content[0]["text"] == user_prompt
    assert prompt_content[1]["type"] == "image_url"
    assert prompt_content[1]["image_url"]["url"] == image_url


@pytest.mark.vcr
def test_litellm_openai_with_structured_output(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday. Extract the event information.",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": EVENT_JSON_SCHEMA,
        },
    )
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 93
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 19
    assert spans[0].attributes["llm.usage.total_tokens"] == 112
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "Alice and Bob are going to a science fair on Friday. Extract the event information."
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        spans[0].attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert (
        json.loads(spans[0].attributes["gen_ai.request.structured_output_schema"])
        == EVENT_JSON_SCHEMA["schema"]
    )


@pytest.mark.vcr
def test_litellm_openai_with_structured_output_pydantic(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday. Extract the event information.",
            }
        ],
        response_format=Event,
    )
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 93
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 19
    assert spans[0].attributes["llm.usage.total_tokens"] == 112
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "Alice and Bob are going to a science fair on Friday. Extract the event information."
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        spans[0].attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    # For some reason, litellm adds "additionalProperties" to the schema if it's Pydantic
    # and OpenAI, but doesn't for Anthropic.
    assert json.loads(
        spans[0].attributes["gen_ai.request.structured_output_schema"]
    ) == {"additionalProperties": False, **EVENT_JSON_SCHEMA["schema"]}


@pytest.mark.vcr
def test_litellm_openai_responses(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    # litellm responses only supports strings for metadata and silently ignores the rest
    response = litellm.responses(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
        metadata={
            "tags": json.dumps(["test"]),
            "user_id": "test_user_id",
            "session_id": "test_session_id",
        },
    )
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 14
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 8
    assert spans[0].attributes["llm.usage.total_tokens"] == 22
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the capital of France?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert json.loads(spans[0].attributes["gen_ai.completion.0.content"]) == [
        {
            "type": "text",
            "text": response.output[0].content[0].text,
        }
    ]
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["lmnr.association.properties.tags"] == ("test",)
    assert spans[0].attributes["lmnr.association.properties.user_id"] == "test_user_id"
    assert (
        spans[0].attributes["lmnr.association.properties.session_id"]
        == "test_session_id"
    )


@pytest.mark.vcr
def test_litellm_openai_responses_with_streaming(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.responses(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
        stream=True,
    )
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    final_response = ""
    for chunk in response:
        if chunk.type == "response.output_text.delta":
            final_response += chunk.delta
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    spans = span_exporter.get_finished_spans()

    # TODO/FIXME: This looks more like a bug in LiteLLM than expected, so for
    # now filter out redundant spans.
    spans = [span for span in spans if span.attributes.get("gen_ai.response.id")]

    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the capital of France?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert json.loads(spans[0].attributes["gen_ai.completion.0.content"]) == [
        {
            "type": "text",
            "text": final_response,
        }
    ]
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"


@pytest.mark.vcr
def test_litellm_openai_responses_with_tools(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.responses(
        model="gpt-4.1-nano",
        input="What is the weather in San Francisco?",
        tools=TOOLS_FOR_RESPONSES,
    )

    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the weather in San Francisco?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 69
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 32
    assert spans[0].attributes["llm.usage.total_tokens"] == 101

    assert (
        spans[0].attributes["llm.request.functions.0.name"]
        == TOOLS_FOR_RESPONSES[0]["name"]
    )
    assert (
        spans[0].attributes["llm.request.functions.0.description"]
        == TOOLS_FOR_RESPONSES[0]["description"]
    )
    assert json.loads(spans[0].attributes["llm.request.functions.0.parameters"]) == (
        TOOLS_FOR_RESPONSES[0]["parameters"]
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert (
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.name"]
        == response.output[0].name
    )
    assert (
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.arguments"]
        == response.output[0].arguments
    )
    assert (
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.id"]
        == response.output[0].id
    )


@pytest.mark.vcr
def test_litellm_openai_responses_with_tools_and_streaming(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.responses(
        model="gpt-4.1-nano",
        input="What is the weather in San Francisco?",
        tools=TOOLS_FOR_RESPONSES,
        stream=True,
    )
    arguments = ""
    for chunk in response:
        if chunk.type == "response.function_call_arguments.delta":
            arguments += chunk.delta

    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    # TODO/FIXME: This looks more like a bug in LiteLLM than expected, so for
    # now filter out redundant spans.
    spans = [span for span in spans if span.attributes.get("gen_ai.response.id")]
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the weather in San Francisco?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"

    assert (
        spans[0].attributes["llm.request.functions.0.name"]
        == TOOLS_FOR_RESPONSES[0]["name"]
    )
    assert (
        spans[0].attributes["llm.request.functions.0.description"]
        == TOOLS_FOR_RESPONSES[0]["description"]
    )
    assert json.loads(spans[0].attributes["llm.request.functions.0.parameters"]) == (
        TOOLS_FOR_RESPONSES[0]["parameters"]
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["gen_ai.completion.0.tool_calls.0.name"] == "get_weather"
    assert (
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.arguments"] == arguments
    )
    assert (
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.id"]
        == "fc_68bf044c83fc819b9d1981bb37b722ea0504083095f7537f"
    )


@pytest.mark.vcr
def test_litellm_openai_responses_with_computer_tools(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"
    litellm.callbacks = [litellm_callback]

    user_prompt = "Take a screenshot of the desktop."
    first_response = litellm.responses(
        model="openai/computer-use-preview",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt,
                    }
                ],
            }
        ],
        truncation="auto",
        tools=[
            {
                "type": "computer_use_preview",
                "display_width": 1024,
                "display_height": 768,
                "environment": "linux",
            }
        ],
    )

    litellm.responses(
        model="openai/computer-use-preview",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt,
                    }
                ],
            },
            {
                "type": "computer_call",
                "call_id": first_response.output[-1].call_id,
                "id": first_response.output[-1].id,
                "action": {
                    "type": "screenshot",
                },
                "status": "completed",
            },
            {
                "type": "computer_call_output",
                "call_id": first_response.output[-1].call_id,
                "output": {
                    "type": "computer_screenshot",
                    "image_url": f"data:image/png;base64,{BASE64_IMAGE}",
                },
            },
        ],
        truncation="auto",
        tools=[
            {
                "type": "computer_use_preview",
                "display_width": 1024,
                "display_height": 768,
                "environment": "linux",
            }
        ],
    )

    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    sorted_spans: list[ReadableSpan] = sorted(list(spans), key=lambda s: s.start_time)
    first_span = sorted_spans[0]
    second_span = sorted_spans[1]
    assert first_span.name == "litellm.responses"
    assert first_span.attributes["gen_ai.request.model"] == "computer-use-preview"
    assert first_span.attributes["gen_ai.usage.input_tokens"] == 498
    assert first_span.attributes["gen_ai.usage.output_tokens"] == 7
    assert first_span.attributes["llm.usage.total_tokens"] == 505
    assert first_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert json.loads(first_span.attributes["gen_ai.prompt.0.content"]) == [
        {"type": "input_text", "text": user_prompt}
    ]
    assert first_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert first_span.attributes["gen_ai.system"] == "openai"
    assert (
        first_span.attributes["llm.request.functions.0.name"] == "computer_use_preview"
    )
    assert json.loads(first_span.attributes["llm.request.functions.0.parameters"]) == {
        "display_width": 1024,
        "display_height": 768,
        "environment": "linux",
    }
    assert (
        first_span.attributes["gen_ai.completion.0.tool_calls.0.name"]
        == "computer_call"
    )
    assert json.loads(
        first_span.attributes["gen_ai.completion.0.tool_calls.0.arguments"]
    ) == {
        "type": "screenshot",
    }
    assert (
        first_span.attributes["gen_ai.completion.0.tool_calls.0.id"]
        == first_response.output[-1].call_id
    )

    assert second_span.name == "litellm.responses"
    assert second_span.attributes["gen_ai.request.model"] == "computer-use-preview"
    assert second_span.attributes["gen_ai.usage.input_tokens"] == 2212
    assert second_span.attributes["gen_ai.usage.output_tokens"] == 38
    assert second_span.attributes["llm.usage.total_tokens"] == 2250
    assert second_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert json.loads(second_span.attributes["gen_ai.prompt.0.content"]) == [
        {"type": "input_text", "text": user_prompt}
    ]
    assert second_span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert second_span.attributes["gen_ai.system"] == "openai"
    assert (
        second_span.attributes["llm.request.functions.0.name"] == "computer_use_preview"
    )
    assert json.loads(second_span.attributes["llm.request.functions.0.parameters"]) == {
        "display_width": 1024,
        "display_height": 768,
        "environment": "linux",
    }
    assert (
        second_span.attributes["gen_ai.prompt.1.tool_calls.0.name"] == "computer_call"
    )
    assert json.loads(
        second_span.attributes["gen_ai.prompt.1.tool_calls.0.arguments"]
    ) == {
        "action": {"type": "screenshot"},
        "id": first_response.output[-1].id,
    }
    assert (
        second_span.attributes["gen_ai.prompt.1.tool_calls.0.id"]
        == first_response.output[-1].call_id
    )
    assert second_span.attributes["gen_ai.prompt.1.role"] == "assistant"
    assert json.loads(second_span.attributes["gen_ai.prompt.2.content"]) == [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{BASE64_IMAGE}"},
        }
    ]
    assert second_span.attributes["gen_ai.prompt.2.role"] == "computer_call_output"


@pytest.mark.vcr
def test_litellm_openai_with_structured_output_and_streaming(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday. Extract the event information.",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": EVENT_JSON_SCHEMA,
        },
        stream=True,
    )
    final_response = ""
    for chunk in response:
        final_response += chunk.choices[0].delta.content or ""
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 93
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 19
    assert spans[0].attributes["llm.usage.total_tokens"] == 112
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "Alice and Bob are going to a science fair on Friday. Extract the event information."
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert spans[0].attributes["gen_ai.completion.0.content"] == final_response
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert (
        json.loads(spans[0].attributes["gen_ai.request.structured_output_schema"])
        == EVENT_JSON_SCHEMA["schema"]
    )


@pytest.mark.vcr
def test_litellm_openai_with_structured_output_pydantic_and_streaming(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday. Extract the event information.",
            }
        ],
        response_format=Event,
        stream=True,
    )
    final_response = ""
    for chunk in response:
        final_response += chunk.choices[0].delta.content or ""
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 93
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 19
    assert spans[0].attributes["llm.usage.total_tokens"] == 112
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "Alice and Bob are going to a science fair on Friday. Extract the event information."
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert spans[0].attributes["gen_ai.completion.0.content"] == final_response
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    # For some reason, litellm adds "additionalProperties" to the schema if it's Pydantic
    # and OpenAI, but doesn't for Anthropic.
    assert json.loads(
        spans[0].attributes["gen_ai.request.structured_output_schema"]
    ) == {**EVENT_JSON_SCHEMA["schema"], "additionalProperties": False}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_with_image_base64(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    # Sample image URL and base64 encoding
    with open("tests/data/base64_image.txt", "r") as f:
        image_data = f.read().strip()
    image_media_type = "image/jpeg"

    litellm.callbacks = [litellm_callback]
    user_prompt = "What's depicted in this image?"
    response = await litellm.acompletion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_media_type};base64,{image_data}"
                        },
                    },
                ],
            }
        ],
    )

    # Wait for the callback to complete and flush the spans
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        span.attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.system"] == "openai"
    # Check that the content includes both text and image
    prompt_content = json.loads(span.attributes["gen_ai.prompt.0.content"])
    assert len(prompt_content) == 2
    assert prompt_content[0]["type"] == "text"
    assert prompt_content[0]["text"] == user_prompt
    assert prompt_content[1]["type"] == "image_url"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_with_image_url(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    # Direct image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/8/8e/MuseumOfFineArtsBoston_CopleySquare_19thc.jpg"

    litellm.callbacks = [litellm_callback]
    user_prompt = "What's depicted in this image?"
    response = await litellm.acompletion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
    )

    # Wait for the callback to complete and flush the spans
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        span.attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.system"] == "openai"
    # Check that the content includes both text and image
    prompt_content = json.loads(span.attributes["gen_ai.prompt.0.content"])
    assert len(prompt_content) == 2
    assert prompt_content[0]["type"] == "text"
    assert prompt_content[0]["text"] == user_prompt
    assert prompt_content[1]["type"] == "image_url"
    assert prompt_content[1]["image_url"]["url"] == image_url


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_basic(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    # Wait for the callback to complete and flush the spans
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.usage.input_tokens"] == 14
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["llm.usage.total_tokens"] == 21
    assert (
        span.attributes["gen_ai.prompt.0.content"] == "What is the capital of France?"
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        span.attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.system"] == "openai"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_text_block(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of France?"}],
            }
        ],
    )

    # Wait for the callback to complete and flush the spans
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.usage.input_tokens"] == 14
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["llm.usage.total_tokens"] == 21
    assert json.loads(span.attributes["gen_ai.prompt.0.content"]) == [
        {"type": "text", "text": "What is the capital of France?"}
    ]
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        span.attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.system"] == "openai"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_with_streaming(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
    )

    final_response = ""
    async for chunk in response:
        final_response += chunk.choices[0].delta.content or ""

    # Wait for the callback to complete and flush the spans
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.usage.input_tokens"] == 14
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["llm.usage.total_tokens"] == 21
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the capital of France?"
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert span.attributes["gen_ai.completion.0.content"] == final_response
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.system"] == "openai"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_with_streaming_and_metadata(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
        metadata={
            "tags": ["test"],
            "user_id": "test_user_id",
            "session_id": "test_session_id",
        },
    )

    final_response = ""
    async for chunk in response:
        final_response += chunk.choices[0].delta.content or ""

    # Wait for the callback to complete and flush the spans
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert span.attributes["gen_ai.usage.input_tokens"] == 14
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["llm.usage.total_tokens"] == 21
    assert (
        span.attributes["gen_ai.prompt.0.content"] == "What is the capital of France?"
    )
    assert span.attributes["gen_ai.prompt.0.role"] == "user"
    assert span.attributes["gen_ai.completion.0.content"] == final_response
    assert span.attributes["gen_ai.completion.0.role"] == "assistant"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["lmnr.association.properties.tags"] == ("test",)
    assert span.attributes["lmnr.association.properties.user_id"] == "test_user_id"
    assert (
        span.attributes["lmnr.association.properties.session_id"] == "test_session_id"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_with_structured_output(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday. Extract the event information.",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": EVENT_JSON_SCHEMA,
        },
    )
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 93
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 19
    assert spans[0].attributes["llm.usage.total_tokens"] == 112
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "Alice and Bob are going to a science fair on Friday. Extract the event information."
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        spans[0].attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert (
        json.loads(spans[0].attributes["gen_ai.request.structured_output_schema"])
        == EVENT_JSON_SCHEMA["schema"]
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_with_structured_output_pydantic(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday. Extract the event information.",
            }
        ],
        response_format=Event,
    )
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 93
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 19
    assert spans[0].attributes["llm.usage.total_tokens"] == 112
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "Alice and Bob are going to a science fair on Friday. Extract the event information."
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        spans[0].attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    # For some reason, litellm adds "additionalProperties" to the schema if it's Pydantic
    # and OpenAI, but doesn't for Anthropic.
    assert json.loads(
        spans[0].attributes["gen_ai.request.structured_output_schema"]
    ) == {"additionalProperties": False, **EVENT_JSON_SCHEMA["schema"]}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_with_structured_output_and_streaming(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday. Extract the event information.",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": EVENT_JSON_SCHEMA,
        },
        stream=True,
    )
    final_response = ""
    async for chunk in response:
        final_response += chunk.choices[0].delta.content or ""
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 93
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 19
    assert spans[0].attributes["llm.usage.total_tokens"] == 112
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "Alice and Bob are going to a science fair on Friday. Extract the event information."
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert spans[0].attributes["gen_ai.completion.0.content"] == final_response
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert (
        json.loads(spans[0].attributes["gen_ai.request.structured_output_schema"])
        == EVENT_JSON_SCHEMA["schema"]
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_with_structured_output_pydantic_and_streaming(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday. Extract the event information.",
            }
        ],
        response_format=Event,
        stream=True,
    )
    final_response = ""
    async for chunk in response:
        final_response += chunk.choices[0].delta.content or ""
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 93
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 19
    assert spans[0].attributes["llm.usage.total_tokens"] == 112
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "Alice and Bob are going to a science fair on Friday. Extract the event information."
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert spans[0].attributes["gen_ai.completion.0.content"] == final_response
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    # For some reason, litellm adds "additionalProperties" to the schema if it's Pydantic
    # and OpenAI, but doesn't for Anthropic.
    assert json.loads(
        spans[0].attributes["gen_ai.request.structured_output_schema"]
    ) == {"additionalProperties": False, **EVENT_JSON_SCHEMA["schema"]}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_responses(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    # litellm responses only supports strings for metadata and silently ignores the rest
    response = await litellm.aresponses(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
        metadata={
            "tags": json.dumps(["test"]),
            "user_id": "test_user_id",
            "session_id": "test_session_id",
        },
    )
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 14
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 8
    assert spans[0].attributes["llm.usage.total_tokens"] == 22
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the capital of France?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert json.loads(spans[0].attributes["gen_ai.completion.0.content"]) == [
        {
            "type": "text",
            "text": response.output[0].content[0].text,
        }
    ]
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["lmnr.association.properties.tags"] == ("test",)
    assert spans[0].attributes["lmnr.association.properties.user_id"] == "test_user_id"
    assert (
        spans[0].attributes["lmnr.association.properties.session_id"]
        == "test_session_id"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_responses_with_streaming(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.aresponses(
        model="gpt-4.1-nano",
        input="What is the capital of France?",
        stream=True,
    )
    final_response = ""
    async for chunk in response:
        if chunk.type == "response.output_text.delta":
            final_response += chunk.delta
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    spans = span_exporter.get_finished_spans()
    # TODO/FIXME: This looks more like a bug in LiteLLM than expected, so for
    # now filter out redundant spans.
    spans = [span for span in spans if span.attributes.get("gen_ai.response.id")]
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the capital of France?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert json.loads(spans[0].attributes["gen_ai.completion.0.content"]) == [
        {
            "type": "text",
            "text": final_response,
        }
    ]
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_responses_with_tools(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.aresponses(
        model="gpt-4.1-nano",
        input="What is the weather in San Francisco?",
        tools=TOOLS_FOR_RESPONSES,
    )

    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the weather in San Francisco?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 69
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 16
    assert spans[0].attributes["llm.usage.total_tokens"] == 85

    assert (
        spans[0].attributes["llm.request.functions.0.name"]
        == TOOLS_FOR_RESPONSES[0]["name"]
    )
    assert (
        spans[0].attributes["llm.request.functions.0.description"]
        == TOOLS_FOR_RESPONSES[0]["description"]
    )
    assert json.loads(spans[0].attributes["llm.request.functions.0.parameters"]) == (
        TOOLS_FOR_RESPONSES[0]["parameters"]
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert (
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.name"]
        == response.output[0].name
    )
    assert (
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.arguments"]
        == response.output[0].arguments
    )
    assert (
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.id"]
        == response.output[0].id
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_responses_with_tools_and_streaming(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.aresponses(
        model="gpt-4.1-nano",
        input="What is the weather in San Francisco?",
        tools=TOOLS_FOR_RESPONSES,
        stream=True,
    )
    arguments = ""
    async for chunk in response:
        if chunk.type == "response.function_call_arguments.delta":
            arguments += chunk.delta
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    # TODO/FIXME: This looks more like a bug in LiteLLM than expected, so for
    # now filter out redundant spans.
    spans = [span for span in spans if span.attributes.get("gen_ai.response.id")]
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano-2025-04-14"
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the weather in San Francisco?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"

    assert (
        spans[0].attributes["llm.request.functions.0.name"]
        == TOOLS_FOR_RESPONSES[0]["name"]
    )
    assert (
        spans[0].attributes["llm.request.functions.0.description"]
        == TOOLS_FOR_RESPONSES[0]["description"]
    )
    assert json.loads(spans[0].attributes["llm.request.functions.0.parameters"]) == (
        TOOLS_FOR_RESPONSES[0]["parameters"]
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["gen_ai.completion.0.tool_calls.0.name"] == "get_weather"
    assert (
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.arguments"] == arguments
    )
    assert (
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.id"]
        == "fc_68bf063821bc819bae60706ae40e403504275f3c4bd02f58"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_openai_responses_with_computer_tools(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["OPENAI_API_KEY"] = "test-key"
    litellm.callbacks = [litellm_callback]
    await litellm.aresponses(
        model="openai/computer-use-preview",
        input="Take a screenshot of the desktop.",
        truncation="auto",
        tools=[
            {
                "type": "computer_use_preview",
                "display_width": 1024,
                "display_height": 768,
                "environment": "linux",
            }
        ],
    )

    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.request.model"] == "computer-use-preview"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 498
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 7
    assert spans[0].attributes["llm.usage.total_tokens"] == 505
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "Take a screenshot of the desktop."
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["llm.request.functions.0.name"] == "computer_use_preview"
    assert json.loads(spans[0].attributes["llm.request.functions.0.parameters"]) == {
        "display_width": 1024,
        "display_height": 768,
        "environment": "linux",
    }
    assert (
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.name"] == "computer_call"
    )
    assert json.loads(
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.arguments"]
    ) == {
        "type": "screenshot",
    }
    assert (
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.id"]
        == "call_ORDTqhMq9WvShpSJYjsugxXY"
    )

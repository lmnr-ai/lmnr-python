import asyncio
import json
import litellm
import os
import pytest
import time

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.litellm import LaminarLiteLLMCallback
from lmnr import Laminar


SLEEP_TO_FLUSH_SECONDS = 0.05


@pytest.mark.vcr
def test_litellm_anthropic_basic(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="claude-3-5-haiku-latest",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    # Wait for the callback to complete and flush the spans
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert spans[0].attributes["gen_ai.response.model"] == "claude-3-5-haiku-20241022"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 14
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 10
    assert spans[0].attributes["llm.usage.total_tokens"] == 24
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the capital of France?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        spans[0].attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "anthropic"


@pytest.mark.vcr
def test_litellm_anthropic_with_metadata(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="claude-3-5-haiku-latest",
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
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert spans[0].attributes["gen_ai.response.model"] == "claude-3-5-haiku-20241022"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 14
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 10
    assert spans[0].attributes["llm.usage.total_tokens"] == 24
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the capital of France?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        spans[0].attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "anthropic"
    assert spans[0].attributes["lmnr.association.properties.tags"] == ("test",)
    assert spans[0].attributes["lmnr.association.properties.user_id"] == "test_user_id"
    assert (
        spans[0].attributes["lmnr.association.properties.session_id"]
        == "test_session_id"
    )


@pytest.mark.vcr
def test_litellm_anthropic_text_block(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="claude-3-5-haiku-latest",
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
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert spans[0].attributes["gen_ai.response.model"] == "claude-3-5-haiku-20241022"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 14
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 10
    assert spans[0].attributes["llm.usage.total_tokens"] == 24
    assert json.loads(spans[0].attributes["gen_ai.prompt.0.content"]) == [
        {"type": "text", "text": "What is the capital of France?"}
    ]
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        spans[0].attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "anthropic"


@pytest.mark.vcr
def test_litellm_anthropic_with_streaming(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="claude-3-5-haiku-latest",
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
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 14
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 10
    assert spans[0].attributes["llm.usage.total_tokens"] == 24
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the capital of France?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert spans[0].attributes["gen_ai.completion.0.content"] == final_response
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "anthropic"


@pytest.mark.vcr
def test_litellm_anthropic_with_streaming_and_metadata(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="claude-3-5-haiku-latest",
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
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 14
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 10
    assert spans[0].attributes["llm.usage.total_tokens"] == 24
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the capital of France?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert spans[0].attributes["gen_ai.completion.0.content"] == final_response
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "anthropic"
    assert spans[0].attributes["lmnr.association.properties.tags"] == ("test",)
    assert spans[0].attributes["lmnr.association.properties.user_id"] == "test_user_id"
    assert (
        spans[0].attributes["lmnr.association.properties.session_id"]
        == "test_session_id"
    )


@pytest.mark.vcr
def test_litellm_anthropic_with_chat_history(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    litellm.callbacks = [litellm_callback]
    user_prompt = (
        "Generate a random adjective in English. Respond with only the adjective."
    )
    first_response = litellm.completion(
        model="claude-3-5-haiku-latest",
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    second_response = litellm.completion(
        model="claude-3-5-haiku-latest",
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
    first_span = sorted(spans, key=lambda x: x.start_time)[0]
    second_span = sorted(spans, key=lambda x: x.start_time)[1]
    assert first_span.name == "litellm.completion"
    assert second_span.name == "litellm.completion"
    assert first_span.attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert second_span.attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"

    assert first_span.attributes["gen_ai.response.id"] == first_response.id
    assert second_span.attributes["gen_ai.response.id"] == second_response.id
    assert first_span.attributes["gen_ai.system"] == "anthropic"
    assert second_span.attributes["gen_ai.system"] == "anthropic"

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
def test_litellm_anthropic_with_chat_history_and_tools(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    litellm.callbacks = [litellm_callback]
    tools = [
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
    user_prompt = "What is the weather in San Francisco? Also, any news in town?"
    first_response = litellm.completion(
        model="claude-3-5-haiku-latest",
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        tools=tools,
    )

    second_response = litellm.completion(
        model="claude-3-5-haiku-latest",
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
        ],
        tools=tools,
    )

    # Wait for the callback to complete and flush the spans
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    first_span = spans[0]
    second_span = spans[1]
    assert first_span.name == "litellm.completion"
    assert second_span.name == "litellm.completion"
    assert first_span.attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert second_span.attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"

    assert first_span.attributes["gen_ai.response.id"] == first_response.id
    assert second_span.attributes["gen_ai.response.id"] == second_response.id
    assert first_span.attributes["gen_ai.system"] == "anthropic"
    assert second_span.attributes["gen_ai.system"] == "anthropic"

    assert first_span.attributes["gen_ai.prompt.0.content"] == user_prompt
    assert first_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert second_span.attributes["gen_ai.prompt.0.content"] == user_prompt
    assert second_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        second_span.attributes["gen_ai.prompt.1.content"]
        == first_response.choices[0].message.content
    )
    assert second_span.attributes["gen_ai.prompt.1.role"] == "assistant"
    assert second_span.attributes["gen_ai.prompt.2.content"] == "Sunny as always!"
    assert second_span.attributes["gen_ai.prompt.2.role"] == "tool"

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

    assert (
        first_span.attributes["llm.request.functions.0.name"]
        == tools[0]["function"]["name"]
    )
    assert (
        first_span.attributes["llm.request.functions.0.description"]
        == tools[0]["function"]["description"]
    )
    assert json.loads(first_span.attributes["llm.request.functions.0.parameters"]) == (
        tools[0]["function"]["parameters"]
    )
    assert (
        first_span.attributes["llm.request.functions.1.name"]
        == tools[1]["function"]["name"]
    )
    assert (
        first_span.attributes["llm.request.functions.1.description"]
        == tools[1]["function"]["description"]
    )
    assert json.loads(first_span.attributes["llm.request.functions.1.parameters"]) == (
        tools[1]["function"]["parameters"]
    )


@pytest.mark.vcr
def test_litellm_anthropic_with_image_base64(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    with open("tests/data/base64_image.txt", "r") as f:
        image_data = f.read().strip()
    image_media_type = "image/jpeg"

    litellm.callbacks = [litellm_callback]
    user_prompt = "What's depicted in this image?"
    response = litellm.completion(
        model="claude-3-5-haiku-latest",
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
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert spans[0].attributes["gen_ai.response.model"] == "claude-3-5-haiku-20241022"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        spans[0].attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "anthropic"
    # Check that the content includes both text and image
    prompt_content = json.loads(spans[0].attributes["gen_ai.prompt.0.content"])
    assert len(prompt_content) == 2
    assert prompt_content[0]["type"] == "text"
    assert prompt_content[0]["text"] == user_prompt
    assert prompt_content[1]["type"] == "image_url"


@pytest.mark.vcr
def test_litellm_anthropic_with_image_url(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    # Direct image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/8/8e/MuseumOfFineArtsBoston_CopleySquare_19thc.jpg"

    litellm.callbacks = [litellm_callback]
    user_prompt = "What's depicted in this image?"
    response = litellm.completion(
        model="claude-3-5-haiku-latest",
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
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert spans[0].attributes["gen_ai.response.model"] == "claude-3-5-haiku-20241022"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        spans[0].attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "anthropic"
    # Check that the content includes both text and image
    prompt_content = json.loads(spans[0].attributes["gen_ai.prompt.0.content"])
    assert len(prompt_content) == 2
    assert prompt_content[0]["type"] == "text"
    assert prompt_content[0]["text"] == user_prompt
    assert prompt_content[1]["type"] == "image_url"
    assert prompt_content[1]["image_url"]["url"] == image_url


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_anthropic_with_image_base64(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    with open("tests/data/base64_image.txt", "r") as f:
        image_data = f.read()
    image_media_type = "image/jpeg"

    litellm.callbacks = [litellm_callback]
    user_prompt = "What's depicted in this image?"
    response = await litellm.acompletion(
        model="claude-3-5-haiku-latest",
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
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert spans[0].attributes["gen_ai.response.model"] == "claude-3-5-haiku-20241022"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        spans[0].attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "anthropic"
    # Check that the content includes both text and image
    prompt_content = json.loads(spans[0].attributes["gen_ai.prompt.0.content"])
    assert len(prompt_content) == 2
    assert prompt_content[0]["type"] == "text"
    assert prompt_content[0]["text"] == user_prompt
    assert prompt_content[1]["type"] == "image_url"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_anthropic_with_image_url(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    # Direct image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/8/8e/MuseumOfFineArtsBoston_CopleySquare_19thc.jpg"

    litellm.callbacks = [litellm_callback]
    user_prompt = "What's depicted in this image?"
    response = await litellm.acompletion(
        model="claude-3-5-haiku-latest",
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
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert spans[0].attributes["gen_ai.response.model"] == "claude-3-5-haiku-20241022"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        spans[0].attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "anthropic"
    # Check that the content includes both text and image
    prompt_content = json.loads(spans[0].attributes["gen_ai.prompt.0.content"])
    assert len(prompt_content) == 2
    assert prompt_content[0]["type"] == "text"
    assert prompt_content[0]["text"] == user_prompt
    assert prompt_content[1]["type"] == "image_url"
    assert prompt_content[1]["image_url"]["url"] == image_url


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_anthropic_basic(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-123"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="claude-3-5-haiku-latest",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    # Wait for the callback to complete and flush the spans
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert spans[0].attributes["gen_ai.response.model"] == "claude-3-5-haiku-20241022"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 14
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 10
    assert spans[0].attributes["llm.usage.total_tokens"] == 24
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the capital of France?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        spans[0].attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "anthropic"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_anthropic_text_block(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-123"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="claude-3-5-haiku-latest",
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
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert spans[0].attributes["gen_ai.response.model"] == "claude-3-5-haiku-20241022"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 14
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 10
    assert spans[0].attributes["llm.usage.total_tokens"] == 24
    assert json.loads(spans[0].attributes["gen_ai.prompt.0.content"]) == [
        {"type": "text", "text": "What is the capital of France?"}
    ]
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        spans[0].attributes["gen_ai.completion.0.content"]
        == response.choices[0].message.content
    )
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "anthropic"


@pytest.mark.vcr(record_mode="once")
@pytest.mark.asyncio
async def test_async_litellm_anthropic_with_streaming(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="claude-3-5-haiku-latest",
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
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 14
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 10
    assert spans[0].attributes["llm.usage.total_tokens"] == 24
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the capital of France?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert spans[0].attributes["gen_ai.completion.0.content"] == final_response
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "anthropic"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_litellm_anthropic_with_streaming_and_metadata(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="claude-3-5-haiku-latest",
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
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-haiku-latest"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 14
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 10
    assert spans[0].attributes["llm.usage.total_tokens"] == 24
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "What is the capital of France?"
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert spans[0].attributes["gen_ai.completion.0.content"] == final_response
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "anthropic"
    assert spans[0].attributes["lmnr.association.properties.tags"] == ("test",)
    assert spans[0].attributes["lmnr.association.properties.user_id"] == "test_user_id"
    assert (
        spans[0].attributes["lmnr.association.properties.session_id"]
        == "test_session_id"
    )

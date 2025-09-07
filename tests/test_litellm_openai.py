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
    first_span = sorted(spans, key=lambda s: s.start_time)[0]
    second_span = sorted(spans, key=lambda s: s.start_time)[1]
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


@pytest.mark.vcr(record_mode="once")
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

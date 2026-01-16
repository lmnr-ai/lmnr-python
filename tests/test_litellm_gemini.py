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
def test_litellm_gemini_thinking(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["GEMINI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {"role": "system", "content": "Think deep and thoroughly step by step."},
            {
                "role": "user",
                "content": "How many times does the letter 'r' appear in the word strawberry?",
            },
        ],
        thinking={"type": "enabled", "budget_tokens": 512, "include_thoughts": True},
    )

    # Wait for the callback to complete and flush the spans
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 24
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 289
    assert spans[0].attributes["gen_ai.usage.reasoning_tokens"] == 223
    assert spans[0].attributes["llm.usage.total_tokens"] == 313
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "Think deep and thoroughly step by step."
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "system"
    assert (
        spans[0].attributes["gen_ai.prompt.1.content"]
        == "How many times does the letter 'r' appear in the word strawberry?"
    )
    assert spans[0].attributes["gen_ai.prompt.1.role"] == "user"
    assert json.loads(spans[0].attributes["gen_ai.completion.0.content"]) == [
        {"type": "text", "text": response.choices[0].message.reasoning_content},
        {"type": "text", "text": response.choices[0].message.content},
    ]
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "gemini"


@pytest.mark.vcr
def test_litellm_gemini_thinking_with_streaming(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["GEMINI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = litellm.completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {"role": "system", "content": "Think deep and thoroughly step by step."},
            {
                "role": "user",
                "content": "How many times does the letter 'r' appear in the word strawberry?",
            },
        ],
        thinking={"type": "enabled", "budget_tokens": 512, "include_thoughts": True},
        stream=True,
    )

    final_response = ""
    final_reasoning_response = ""
    for chunk in response:
        final_response += chunk.choices[0].delta.content or ""
        if hasattr(chunk.choices[0].delta, "reasoning_content"):
            final_reasoning_response += chunk.choices[0].delta.reasoning_content or ""
    assert final_reasoning_response

    # Wait for the callback to complete and flush the spans
    time.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    time.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 24
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 303
    assert spans[0].attributes["gen_ai.usage.reasoning_tokens"] == 220
    assert spans[0].attributes["llm.usage.total_tokens"] == 327
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "Think deep and thoroughly step by step."
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "system"
    assert (
        spans[0].attributes["gen_ai.prompt.1.content"]
        == "How many times does the letter 'r' appear in the word strawberry?"
    )
    assert spans[0].attributes["gen_ai.prompt.1.role"] == "user"
    assert json.loads(spans[0].attributes["gen_ai.completion.0.content"]) == [
        {"type": "text", "text": final_reasoning_response},
        {"type": "text", "text": final_response},
    ]
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "gemini"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_gemini_thinking_async(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["GEMINI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {"role": "system", "content": "Think deep and thoroughly step by step."},
            {
                "role": "user",
                "content": "How many times does the letter 'r' appear in the word strawberry?",
            },
        ],
        thinking={"type": "enabled", "budget_tokens": 512, "include_thoughts": True},
    )

    # Wait for the callback to complete and flush the spans
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 24
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 313
    assert spans[0].attributes["gen_ai.usage.reasoning_tokens"] == 235
    assert spans[0].attributes["llm.usage.total_tokens"] == 337
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "Think deep and thoroughly step by step."
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "system"
    assert (
        spans[0].attributes["gen_ai.prompt.1.content"]
        == "How many times does the letter 'r' appear in the word strawberry?"
    )
    assert spans[0].attributes["gen_ai.prompt.1.role"] == "user"
    assert json.loads(spans[0].attributes["gen_ai.completion.0.content"]) == [
        {"type": "text", "text": response.choices[0].message.reasoning_content},
        {"type": "text", "text": response.choices[0].message.content},
    ]
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "gemini"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_gemini_thinking_with_streaming_async(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    os.environ["GEMINI_API_KEY"] = "test-key"

    litellm.callbacks = [litellm_callback]
    response = await litellm.acompletion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {"role": "system", "content": "Think deep and thoroughly step by step."},
            {
                "role": "user",
                "content": "How many times does the letter 'r' appear in the word strawberry?",
            },
        ],
        thinking={"type": "enabled", "budget_tokens": 512, "include_thoughts": True},
        stream=True,
    )

    final_response = ""
    final_reasoning_response = ""
    async for chunk in response:
        final_response += chunk.choices[0].delta.content or ""
        if hasattr(chunk.choices[0].delta, "reasoning_content"):
            final_reasoning_response += chunk.choices[0].delta.reasoning_content or ""

    # Wait for the callback to complete and flush the spans
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)
    Laminar.flush()
    await asyncio.sleep(SLEEP_TO_FLUSH_SECONDS)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 24
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 269
    assert spans[0].attributes["gen_ai.usage.reasoning_tokens"] == 222
    assert spans[0].attributes["llm.usage.total_tokens"] == 293
    assert (
        spans[0].attributes["gen_ai.prompt.0.content"]
        == "Think deep and thoroughly step by step."
    )
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "system"
    assert (
        spans[0].attributes["gen_ai.prompt.1.content"]
        == "How many times does the letter 'r' appear in the word strawberry?"
    )
    assert spans[0].attributes["gen_ai.prompt.1.role"] == "user"
    assert json.loads(spans[0].attributes["gen_ai.completion.0.content"]) == [
        {"type": "text", "text": final_reasoning_response},
        {"type": "text", "text": final_response},
    ]
    assert spans[0].attributes["gen_ai.completion.0.role"] == "assistant"
    assert spans[0].attributes["gen_ai.system"] == "gemini"

import json
from typing import Any, cast

import pytest
from anthropic import Anthropic, AsyncAnthropic
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from .utils import successful_spans


@pytest.mark.vcr
def test_anthropic_thinking_legacy(
    instrumentor: Any,
    anthropic_client: Anthropic,
    span_exporter: InMemorySpanExporter,
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        anthropic_client.messages.create(  # pyright: ignore[reportCallIssue] intentional
            unknown_parameter="unknown",
        )
    except Exception:
        print("expected exception")

    response = anthropic_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024,
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
    )

    spans = successful_spans(span_exporter)
    anthropic_span = spans[0]
    attributes = anthropic_span.attributes or {}

    assert anthropic_span.name == "anthropic.chat"
    assert attributes["gen_ai.request.thinking_type"] == "enabled"
    assert attributes["gen_ai.request.thinking_budget"] == 1024

    # Verify input messages
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["role"] == "user"
    content = input_messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == prompt

    # Verify output messages - should contain thinking and text blocks
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"

    content_blocks = output_messages[0]["content"]
    thinking_blocks = [b for b in content_blocks if b.get("type") == "thinking"]
    text_blocks = [b for b in content_blocks if b.get("type") == "text"]

    assert len(thinking_blocks) >= 1
    assert thinking_blocks[0]["thinking"] == response.content[0].thinking  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert len(text_blocks) >= 1
    assert text_blocks[0]["text"] == response.content[1].text  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_thinking_legacy(
    instrumentor: Any,
    async_anthropic_client: AsyncAnthropic,
    span_exporter: InMemorySpanExporter,
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        await async_anthropic_client.messages.create(  # pyright: ignore[reportCallIssue] intentional
            unknown_parameter="unknown",
        )
    except Exception:
        print("expected exception")

    response = await async_anthropic_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024,
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
    )

    spans = successful_spans(span_exporter)
    anthropic_span = spans[0]
    attributes = anthropic_span.attributes or {}

    assert anthropic_span.name == "anthropic.chat"
    assert attributes["gen_ai.request.thinking_type"] == "enabled"
    assert attributes["gen_ai.request.thinking_budget"] == 1024

    # Verify input messages
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["role"] == "user"

    # Verify output messages - should contain thinking and text blocks
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"

    content_blocks = output_messages[0]["content"]
    thinking_blocks = [b for b in content_blocks if b.get("type") == "thinking"]
    text_blocks = [b for b in content_blocks if b.get("type") == "text"]

    assert len(thinking_blocks) >= 1
    assert thinking_blocks[0]["thinking"] == response.content[0].thinking  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert len(text_blocks) >= 1
    assert text_blocks[0]["text"] == response.content[1].text  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]


@pytest.mark.vcr
def test_anthropic_thinking_streaming_legacy(
    instrumentor: Any,
    anthropic_client: Anthropic,
    span_exporter: InMemorySpanExporter,
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        anthropic_client.messages.create(  # pyright: ignore[reportCallIssue] intentional
            unknown_parameter="unknown",
        )
    except Exception:
        print("expected exception")

    response = anthropic_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        stream=True,
        max_tokens=2048,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024,
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
    )

    text = ""
    thinking = ""

    for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            text += event.delta.text
        elif event.type == "content_block_delta" and event.delta.type == "thinking_delta":
            thinking += event.delta.thinking

    spans = successful_spans(span_exporter)
    anthropic_span = spans[0]
    attributes = anthropic_span.attributes or {}

    assert anthropic_span.name == "anthropic.chat"
    assert attributes["gen_ai.request.thinking_type"] == "enabled"
    assert attributes["gen_ai.request.thinking_budget"] == 1024

    # Verify input messages
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["role"] == "user"

    # Verify output messages - should contain thinking and text blocks
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"

    content_blocks = output_messages[0]["content"]
    thinking_blocks = [b for b in content_blocks if b.get("type") == "thinking"]
    text_blocks = [b for b in content_blocks if b.get("type") == "text"]

    assert len(thinking_blocks) >= 1
    assert thinking_blocks[0]["thinking"] == thinking
    assert len(text_blocks) >= 1
    assert text_blocks[0]["text"] == text


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_thinking_streaming_legacy(
    instrumentor: Any,
    async_anthropic_client: AsyncAnthropic,
    span_exporter: InMemorySpanExporter,
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        await async_anthropic_client.messages.create(  # pyright: ignore[reportCallIssue] intentional
            unknown_parameter="unknown",
        )
    except Exception:
        print("expected exception")

    response = await async_anthropic_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        stream=True,
        max_tokens=2048,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024,
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
    )

    text = ""
    thinking = ""

    async for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            text += event.delta.text
        elif event.type == "content_block_delta" and event.delta.type == "thinking_delta":
            thinking += event.delta.thinking

    spans = successful_spans(span_exporter)
    anthropic_span = spans[0]
    attributes = anthropic_span.attributes or {}

    assert anthropic_span.name == "anthropic.chat"
    assert attributes["gen_ai.request.thinking_type"] == "enabled"
    assert attributes["gen_ai.request.thinking_budget"] == 1024

    # Verify input messages
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["role"] == "user"

    # Verify output messages - should contain thinking and text blocks
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"

    content_blocks = output_messages[0]["content"]
    thinking_blocks = [b for b in content_blocks if b.get("type") == "thinking"]
    text_blocks = [b for b in content_blocks if b.get("type") == "text"]

    assert len(thinking_blocks) >= 1
    assert thinking_blocks[0]["thinking"] == thinking
    assert len(text_blocks) >= 1
    assert text_blocks[0]["text"] == text


@pytest.mark.vcr
def test_anthropic_thinking_adaptive_effort(
    instrumentor: Any,
    anthropic_client: Anthropic,
    span_exporter: InMemorySpanExporter,
):
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        messages=[
            {
                "role": "user",
                "content": "How many times does the letter 'r' appear in the word strawberry?",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    anthropic_span = spans[0]
    attributes = anthropic_span.attributes or {}

    assert anthropic_span.name == "anthropic.chat"
    assert attributes["gen_ai.request.thinking_type"] == "adaptive"
    assert attributes["gen_ai.request.effort"] == "low"
    # Adaptive thinking has no token budget
    assert "gen_ai.request.thinking_budget" not in attributes
    assert attributes["gen_ai.response.id"] == response.id

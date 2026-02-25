import json

import pytest


@pytest.mark.vcr
def test_anthropic_thinking_legacy(instrument_legacy, anthropic_client, span_exporter):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

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

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"

    # Verify input messages
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert input_messages[0]["role"] == "user"
    content = input_messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == prompt

    # Verify output messages - should contain thinking and text blocks
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"

    content_blocks = output_messages[0]["content"]
    thinking_blocks = [b for b in content_blocks if b.get("type") == "thinking"]
    text_blocks = [b for b in content_blocks if b.get("type") == "text"]

    assert len(thinking_blocks) >= 1
    assert thinking_blocks[0]["thinking"] == response.content[0].thinking
    assert len(text_blocks) >= 1
    assert text_blocks[0]["text"] == response.content[1].text


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_thinking_legacy(
    instrument_legacy, async_anthropic_client, span_exporter
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

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

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"

    # Verify input messages
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert input_messages[0]["role"] == "user"

    # Verify output messages - should contain thinking and text blocks
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"

    content_blocks = output_messages[0]["content"]
    thinking_blocks = [b for b in content_blocks if b.get("type") == "thinking"]
    text_blocks = [b for b in content_blocks if b.get("type") == "text"]

    assert len(thinking_blocks) >= 1
    assert thinking_blocks[0]["thinking"] == response.content[0].thinking
    assert len(text_blocks) >= 1
    assert text_blocks[0]["text"] == response.content[1].text


@pytest.mark.vcr
def test_anthropic_thinking_streaming_legacy(
    instrument_legacy, anthropic_client, span_exporter
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

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
        elif (
            event.type == "content_block_delta" and event.delta.type == "thinking_delta"
        ):
            thinking += event.delta.thinking

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"

    # Verify input messages
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert input_messages[0]["role"] == "user"

    # Verify output messages - should contain thinking and text blocks
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
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
    instrument_legacy, async_anthropic_client, span_exporter
):
    prompt = "How many times does the letter 'r' appear in the word strawberry?"

    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

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
        elif (
            event.type == "content_block_delta" and event.delta.type == "thinking_delta"
        ):
            thinking += event.delta.thinking

    spans = span_exporter.get_finished_spans()
    anthropic_span = spans[0]

    assert anthropic_span.name == "anthropic.chat"

    # Verify input messages
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert input_messages[0]["role"] == "user"

    # Verify output messages - should contain thinking and text blocks
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"

    content_blocks = output_messages[0]["content"]
    thinking_blocks = [b for b in content_blocks if b.get("type") == "thinking"]
    text_blocks = [b for b in content_blocks if b.get("type") == "text"]

    assert len(thinking_blocks) >= 1
    assert thinking_blocks[0]["thinking"] == thinking
    assert len(text_blocks) >= 1
    assert text_blocks[0]["text"] == text

import pytest
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.mark.vcr
def test_anthropic_thinking_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter
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
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["gen_ai.prompt.0.content"] == prompt

    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "thinking"
    assert (
        anthropic_span.attributes["gen_ai.completion.0.content"]
        == response.content[0].thinking
    )

    assert anthropic_span.attributes["gen_ai.completion.1.role"] == "assistant"
    assert (
        anthropic_span.attributes["gen_ai.completion.1.content"]
        == response.content[1].text
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_thinking_legacy(
    instrument_legacy, async_anthropic_client, span_exporter, log_exporter
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
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["gen_ai.prompt.0.content"] == prompt

    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "thinking"
    assert (
        anthropic_span.attributes["gen_ai.completion.0.content"]
        == response.content[0].thinking
    )

    assert anthropic_span.attributes["gen_ai.completion.1.role"] == "assistant"
    assert (
        anthropic_span.attributes["gen_ai.completion.1.content"]
        == response.content[1].text
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_anthropic_thinking_streaming_legacy(
    instrument_legacy, anthropic_client, span_exporter, log_exporter
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
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["gen_ai.prompt.0.content"] == prompt

    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "thinking"
    assert anthropic_span.attributes["gen_ai.completion.0.content"] == thinking

    assert anthropic_span.attributes["gen_ai.completion.1.role"] == "assistant"
    assert anthropic_span.attributes["gen_ai.completion.1.content"] == text

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_thinking_streaming_legacy(
    instrument_legacy, async_anthropic_client, span_exporter, log_exporter
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
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["gen_ai.prompt.0.content"] == prompt

    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "thinking"
    assert anthropic_span.attributes["gen_ai.completion.0.content"] == thinking

    assert anthropic_span.attributes["gen_ai.completion.1.role"] == "assistant"
    assert anthropic_span.attributes["gen_ai.completion.1.content"] == text

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"

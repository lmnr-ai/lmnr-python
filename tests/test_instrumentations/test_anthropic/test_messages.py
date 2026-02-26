import asyncio
import base64
import json
from pathlib import Path

import pytest

image_content_block = {
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": base64.b64encode(
            open(
                Path(__file__).parent.joinpath("data/logo.jpg"),
                "rb",
            ).read()
        ).decode("utf-8"),
    },
}


TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_time",
        "description": "Get the current time in a given time zone",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The IANA time zone name, e.g. America/Los_Angeles",
                }
            },
            "required": ["timezone"],
        },
    },
]


@pytest.mark.vcr
def test_anthropic_message_create_legacy(
    instrument_legacy, anthropic_client, span_exporter
):
    response = anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
        service_tier="standard_only",
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.chat" for span in spans)

    anthropic_span = spans[0]

    # Verify input messages in new format
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["content"] == "Tell me a joke about OpenTelemetry"

    # Verify output messages in new format
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert any(
        block["type"] == "text" and block["text"] == response.content[0].text
        for block in output_messages[0]["content"]
    )

    assert anthropic_span.attributes["gen_ai.usage.input_tokens"] == 17
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01NgS2sXcQRKUKbwKFx1vVxC"
    )
    assert (
        anthropic_span.attributes.get("anthropic.request.service_tier")
        == "standard_only"
    )
    assert (
        anthropic_span.attributes.get("anthropic.response.service_tier") == "standard"
    )


@pytest.mark.vcr
def test_anthropic_multi_modal_legacy(
    instrument_legacy, anthropic_client, span_exporter
):
    response = anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see?",
                    },
                    image_content_block,
                ],
            },
        ],
        model="claude-3-opus-20240229",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]

    # Verify input messages in new format
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    # Content should be a list with text and image parts
    content = input_messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "What do you see?"
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"

    # Verify output messages
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert any(
        block["type"] == "text" and block["text"] == response.content[0].text
        for block in output_messages[0]["content"]
    )

    assert anthropic_span.attributes["gen_ai.usage.input_tokens"] == 1381
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01B37ySLPzYj8KY6uZmiPoxd"
    )


@pytest.mark.vcr
def test_anthropic_image_with_history(
    instrument_legacy, anthropic_client, span_exporter
):
    system_message = "You are a helpful assistant. Be concise and to the point."
    user_message1 = {
        "role": "user",
        "content": "Are you capable of describing an image?",
    }
    user_message2 = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see?"},
            image_content_block,
        ],
    }

    response1 = anthropic_client.messages.create(
        max_tokens=1024,
        model="claude-3-5-haiku-latest",
        system=system_message,
        messages=[
            user_message1,
        ],
    )

    response2 = anthropic_client.messages.create(
        max_tokens=1024,
        model="claude-3-5-haiku-latest",
        system=system_message,
        messages=[
            user_message1,
            {"role": "assistant", "content": response1.content},
            user_message2,
        ],
    )

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.chat" for span in spans)

    # First span: system + 1 user message
    input_messages_0 = json.loads(spans[0].attributes["gen_ai.input.messages"])
    assert input_messages_0[0]["role"] == "system"
    assert input_messages_0[0]["content"] == system_message
    assert input_messages_0[1]["role"] == "user"
    assert input_messages_0[1]["content"] == "Are you capable of describing an image?"

    output_messages_0 = json.loads(spans[0].attributes["gen_ai.output.messages"])
    assert output_messages_0[0]["role"] == "assistant"
    assert any(
        block.get("type") == "text" and block.get("text") == response1.content[0].text
        for block in output_messages_0[0]["content"]
    )

    # Second span: system + 3 messages (user, assistant, user with image)
    input_messages_1 = json.loads(spans[1].attributes["gen_ai.input.messages"])
    assert input_messages_1[0]["role"] == "system"
    assert input_messages_1[0]["content"] == system_message
    assert input_messages_1[1]["role"] == "user"
    assert input_messages_1[1]["content"] == "Are you capable of describing an image?"
    assert input_messages_1[2]["role"] == "assistant"
    assert input_messages_1[3]["role"] == "user"
    # Last user message should have text and image parts
    content_3 = input_messages_1[3]["content"]
    assert isinstance(content_3, list)
    assert content_3[0]["type"] == "text"
    assert content_3[0]["text"] == "What do you see?"

    output_messages_1 = json.loads(spans[1].attributes["gen_ai.output.messages"])
    assert output_messages_1[0]["role"] == "assistant"
    assert any(
        block.get("type") == "text" and block.get("text") == response2.content[0].text
        for block in output_messages_1[0]["content"]
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_async_multi_modal_legacy(
    instrument_legacy, async_anthropic_client, span_exporter
):
    response = await async_anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see?",
                    },
                    image_content_block,
                ],
            },
        ],
        model="claude-3-opus-20240229",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]

    # Verify input messages in new format
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    content = input_messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "What do you see?"
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"

    # Verify output messages
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert any(
        block["type"] == "text" and block["text"] == response.content[0].text
        for block in output_messages[0]["content"]
    )

    assert anthropic_span.attributes["gen_ai.usage.input_tokens"] == 1311
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01DWnmUo9hWk4Fk7V7Ddfa2w"
    )


@pytest.mark.vcr
def test_anthropic_message_streaming_legacy(
    instrument_legacy, anthropic_client, span_exporter
):
    response = anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
        stream=True,
        service_tier="standard_only",
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response_content = ""
    for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]

    # Verify input messages
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["content"] == "Tell me a joke about OpenTelemetry"

    # Verify output messages
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert any(
        block["type"] == "text" and block["text"] == response_content
        for block in output_messages[0]["content"]
    )

    assert anthropic_span.attributes["gen_ai.usage.input_tokens"] == 17
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_019bVafnfSbR9K5SGmoy6gcX"
    )
    assert (
        anthropic_span.attributes.get("anthropic.request.service_tier")
        == "standard_only"
    )
    assert (
        anthropic_span.attributes.get("anthropic.response.service_tier") == "standard"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_message_create_legacy(
    instrument_legacy, async_anthropic_client, span_exporter
):
    response = await async_anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
        service_tier="standard_only",
    )
    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]

    # Verify input messages
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["content"] == "Tell me a joke about OpenTelemetry"

    # Verify output messages
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert any(
        block["type"] == "text" and block["text"] == response.content[0].text
        for block in output_messages[0]["content"]
    )

    assert anthropic_span.attributes["gen_ai.usage.input_tokens"] == 17
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01HCcR31VQpZtUqtJ6gZnX34"
    )
    assert (
        anthropic_span.attributes.get("anthropic.request.service_tier")
        == "standard_only"
    )
    assert (
        anthropic_span.attributes.get("anthropic.response.service_tier") == "standard"
    )


@pytest.mark.vcr(record_mode="once")
@pytest.mark.asyncio
async def test_async_anthropic_message_streaming_legacy(
    instrument_legacy, async_anthropic_client, span_exporter
):
    try:
        await async_anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    response = await async_anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
        service_tier="standard_only",
        stream=True,
    )
    response_content = ""
    async for event in response:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]

    # Verify input messages
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["content"] == "Tell me a joke about OpenTelemetry"

    # Verify output messages
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert any(
        block["type"] == "text" and block["text"] == response_content
        for block in output_messages[0]["content"]
    )

    assert anthropic_span.attributes["gen_ai.usage.input_tokens"] == 17
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01TwQxsi2T5Pat3DNJW7m2wx"
    )
    assert (
        anthropic_span.attributes.get("anthropic.request.service_tier")
        == "standard_only"
    )
    assert (
        anthropic_span.attributes.get("anthropic.response.service_tier") == "standard"
    )


@pytest.mark.vcr
def test_anthropic_tools_legacy(instrument_legacy, anthropic_client, span_exporter):
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=TOOLS,
        messages=[
            {
                "role": "user",
                "content": "What is the weather like right now in New York? Also what time is it there now?",
            }
        ],
    )
    try:
        anthropic_client.messages.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify usage
    assert anthropic_span.attributes["gen_ai.usage.input_tokens"] == 514

    # verify input messages
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert (
        input_messages[0]["content"]
        == "What is the weather like right now in New York? Also what time is it there now?"
    )

    # verify tools are still set as individual attributes
    assert json.loads(anthropic_span.attributes["gen_ai.tool.definitions"]) == TOOLS

    # verify output messages
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0].get("stop_reason") == response.stop_reason

    # Should have text block and tool_use blocks in content
    content_blocks = output_messages[0]["content"]
    text_blocks = [b for b in content_blocks if b["type"] == "text"]
    tool_blocks = [b for b in content_blocks if b["type"] == "tool_use"]

    assert len(text_blocks) >= 1
    assert text_blocks[0]["text"] == response.content[0].text
    assert len(tool_blocks) >= 2
    assert tool_blocks[0]["id"] == response.content[1].id
    assert tool_blocks[0]["name"] == response.content[1].name
    assert tool_blocks[0]["input"] == response.content[1].input
    assert tool_blocks[1]["id"] == response.content[2].id
    assert tool_blocks[1]["name"] == response.content[2].name
    assert tool_blocks[1]["input"] == response.content[2].input


@pytest.mark.vcr
def test_anthropic_tools_history_legacy(
    instrument_legacy, anthropic_client, span_exporter
):
    response = anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        tools=TOOLS,
        messages=[
            {
                "role": "user",
                "content": "What is the weather and current time in San Francisco?",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I'll help you get the weather and current time in San Francisco.",
                    },
                    {
                        "id": "call_1",
                        "type": "tool_use",
                        "name": "get_weather",
                        "input": {"location": "San Francisco, CA"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "Sunny and 65 degrees Fahrenheit",
                        "tool_use_id": "call_1",
                    }
                ],
            },
        ],
    )

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify input messages
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 3

    # First message: user
    assert input_messages[0]["role"] == "user"
    assert (
        input_messages[0]["content"]
        == "What is the weather and current time in San Francisco?"
    )

    # Second message: assistant with text and tool_use
    assert input_messages[1]["role"] == "assistant"
    content_1 = input_messages[1]["content"]
    assert isinstance(content_1, list)
    assert content_1[0]["type"] == "text"
    assert (
        content_1[0]["text"]
        == "I'll help you get the weather and current time in San Francisco."
    )
    assert content_1[1]["type"] == "tool_use"
    assert content_1[1]["id"] == "call_1"
    assert content_1[1]["name"] == "get_weather"

    # Third message: user with tool_result
    assert input_messages[2]["role"] == "user"
    content_2 = input_messages[2]["content"]
    assert isinstance(content_2, list)
    assert content_2[0]["type"] == "tool_result"
    assert content_2[0]["content"] == "Sunny and 65 degrees Fahrenheit"
    assert content_2[0]["tool_use_id"] == "call_1"

    # verify output messages
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0].get("stop_reason") == response.stop_reason

    tool_blocks = [b for b in output_messages[0]["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) >= 1
    assert tool_blocks[0]["id"] == response.content[0].id
    assert tool_blocks[0]["name"] == response.content[0].name


@pytest.mark.vcr
def test_anthropic_tools_streaming_legacy(
    instrument_legacy, anthropic_client, span_exporter
):
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=TOOLS,
        messages=[
            {
                "role": "user",
                "content": "What is the weather and current time in San Francisco?",
            }
        ],
        stream=True,
    )

    # consume the streaming iterator
    [event for event in response]

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify input messages
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert (
        input_messages[0]["content"]
        == "What is the weather and current time in San Francisco?"
    )

    # verify output messages
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"

    # Should have text and tool_use blocks
    content_blocks = output_messages[0]["content"]
    text_blocks = [b for b in content_blocks if b["type"] == "text"]
    tool_blocks = [b for b in content_blocks if b["type"] == "tool_use"]

    assert len(text_blocks) >= 1
    assert len(tool_blocks) >= 2
    assert tool_blocks[0]["id"] == "toolu_014x5X91kx3fvdhpLvwXZWE2"
    assert tool_blocks[0]["name"] == "get_weather"
    assert tool_blocks[1]["id"] == "toolu_0121kXsENLvoDZ72LCuAnCCz"
    assert tool_blocks[1]["name"] == "get_time"


@pytest.mark.vcr
def test_with_asyncio_run_legacy(
    instrument_legacy, async_anthropic_client, span_exporter
):
    asyncio.run(
        async_anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": "You help generate concise summaries of news articles and blog posts that user sends you.",
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": "What is the weather in San Francisco?",
                },
            ],
        )
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]


@pytest.mark.vcr(record_mode="once")
def test_anthropic_message_stream_manager(
    instrument_legacy, anthropic_client, span_exporter
):
    response_content = ""
    with anthropic_client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
        service_tier="standard_only",
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]

    # Verify input messages
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["content"] == "Tell me a joke about OpenTelemetry"

    # Verify output messages
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert any(
        block["type"] == "text" and block["text"] == response_content
        for block in output_messages[0]["content"]
    )

    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01MCkQZZtEKF3nVbFaExwATe"
    )
    assert (
        anthropic_span.attributes.get("anthropic.request.service_tier")
        == "standard_only"
    )
    assert (
        anthropic_span.attributes.get("anthropic.response.service_tier") == "standard"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_message_stream_manager(
    instrument_legacy, async_anthropic_client, span_exporter
):
    response_content = ""
    async with async_anthropic_client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-5-haiku-20241022",
        service_tier="standard_only",
    ) as stream:
        async for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                response_content += event.delta.text

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "anthropic.chat",
    ]
    anthropic_span = spans[0]

    # Verify input messages
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["content"] == "Tell me a joke about OpenTelemetry"

    # Verify output messages
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert any(
        block["type"] == "text" and block["text"] == response_content
        for block in output_messages[0]["content"]
    )

    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01QnFxEDGs7cHJegKR367cJ7"
    )
    assert (
        anthropic_span.attributes.get("anthropic.request.service_tier")
        == "standard_only"
    )
    assert (
        anthropic_span.attributes.get("anthropic.response.service_tier") == "standard"
    )

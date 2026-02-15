import asyncio
import base64
import json
from pathlib import Path

import pytest
from opentelemetry.semconv_ai import SpanAttributes

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
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.content[0].text
    )
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 17
    assert (
        anthropic_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        + anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
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
    assert anthropic_span.attributes[
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    ] == json.dumps(
        [
            {"type": "text", "text": "What do you see?"},
            {"type": "image_url", "image_url": {"url": "/some/url"}},
        ]
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.content[0].text
    )
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 1381
    assert (
        anthropic_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        + anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
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
    assert (
        spans[0].attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == system_message
    )
    assert spans[0].attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert (
        spans[0].attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == "Are you capable of describing an image?"
    )
    assert spans[0].attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"
    assert (
        spans[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == response1.content[0].text
    )
    assert (
        spans[0].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    )
    assert (
        spans[0].attributes.get("gen_ai.response.id") == "msg_01Ctc62hUPvikvYASXZqTo9q"
    )

    assert (
        spans[1].attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == system_message
    )
    assert spans[1].attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert (
        spans[1].attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == "Are you capable of describing an image?"
    )
    assert spans[1].attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"
    assert (
        spans[1].attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"]
        == response1.content[0].text
    )
    assert spans[1].attributes[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "assistant"
    assert json.loads(
        spans[1].attributes[f"{SpanAttributes.LLM_PROMPTS}.3.content"]
    ) == [
        {"type": "text", "text": "What do you see?"},
        {"type": "image_url", "image_url": {"url": "/some/url"}},
    ]
    assert spans[1].attributes[f"{SpanAttributes.LLM_PROMPTS}.3.role"] == "user"

    assert (
        spans[1].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == response2.content[0].text
    )
    assert (
        spans[1].attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    )
    assert (
        spans[1].attributes.get("gen_ai.response.id") == "msg_01EtAvxHCWn5jjdUCnG4wEAd"
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
    assert anthropic_span.attributes[
        f"{SpanAttributes.LLM_PROMPTS}.0.content"
    ] == json.dumps(
        [
            {"type": "text", "text": "What do you see?"},
            {"type": "image_url", "image_url": {"url": "/some/url"}},
        ]
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.content[0].text
    )
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 1311
    assert (
        anthropic_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        + anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
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
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response_content
    )
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 17
    assert (
        anthropic_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        + anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
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
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response.content[0].text
    )
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 17
    assert (
        anthropic_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        + anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
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
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response_content
    )
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role")
        == "assistant"
    )
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 17
    assert (
        anthropic_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS]
        + anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS]
        == anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )
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
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify usage
    assert anthropic_span.attributes["gen_ai.usage.prompt_tokens"] == 514
    assert (
        anthropic_span.attributes["gen_ai.usage.completion_tokens"]
        + anthropic_span.attributes["gen_ai.usage.prompt_tokens"]
        == anthropic_span.attributes["llm.usage.total_tokens"]
    )

    # verify request and inputs
    assert (
        anthropic_span.attributes["gen_ai.prompt.0.content"]
        == "What is the weather like right now in New York? Also what time is it there now?"
    )
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["llm.request.functions.0.name"] == "get_weather"
    assert (
        anthropic_span.attributes["llm.request.functions.0.description"]
        == "Get the current weather in a given location"
    )
    assert anthropic_span.attributes[
        "llm.request.functions.0.input_schema"
    ] == json.dumps(
        {
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
        }
    )
    assert anthropic_span.attributes["llm.request.functions.1.name"] == "get_time"
    assert (
        anthropic_span.attributes["llm.request.functions.1.description"]
        == "Get the current time in a given time zone"
    )
    assert anthropic_span.attributes[
        "llm.request.functions.1.input_schema"
    ] == json.dumps(
        {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The IANA time zone name, e.g. America/Los_Angeles",
                }
            },
            "required": ["timezone"],
        }
    )

    # verify response and output
    assert (
        anthropic_span.attributes["gen_ai.completion.0.finish_reason"]
        == response.stop_reason
    )
    assert (
        anthropic_span.attributes["gen_ai.completion.0.content"]
        == response.content[0].text
    )
    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.id"]
    ) == response.content[1].id
    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.name"]
    ) == response.content[1].name
    response_input = json.dumps(response.content[1].input)
    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.arguments"]
        == response_input
    )

    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.1.id"]
    ) == response.content[2].id
    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.1.name"]
    ) == response.content[2].name
    response_input = json.dumps(response.content[2].input)
    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.1.arguments"]
        == response_input
    )
    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01RBkXFe9TmDNNWThMz2HmGt"
    )


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
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify usage
    assert anthropic_span.attributes["gen_ai.usage.prompt_tokens"] == 568
    assert (
        anthropic_span.attributes["gen_ai.usage.completion_tokens"]
        + anthropic_span.attributes["gen_ai.usage.prompt_tokens"]
        == anthropic_span.attributes["llm.usage.total_tokens"]
    )

    # verify request and inputs
    assert (
        anthropic_span.attributes["gen_ai.prompt.0.content"]
        == "What is the weather and current time in San Francisco?"
    )
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert (
        anthropic_span.attributes["gen_ai.prompt.1.content"]
        == "I'll help you get the weather and current time in San Francisco."
    )
    assert anthropic_span.attributes["gen_ai.prompt.1.role"] == "assistant"
    assert json.loads(anthropic_span.attributes["gen_ai.prompt.2.content"]) == [
        {
            "type": "tool_result",
            "content": "Sunny and 65 degrees Fahrenheit",
            "tool_use_id": "call_1",
        }
    ]
    assert anthropic_span.attributes["gen_ai.prompt.2.role"] == "user"
    assert anthropic_span.attributes["llm.request.functions.0.name"] == "get_weather"
    assert (
        anthropic_span.attributes["llm.request.functions.0.description"]
        == "Get the current weather in a given location"
    )
    assert anthropic_span.attributes[
        "llm.request.functions.0.input_schema"
    ] == json.dumps(
        {
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
        }
    )
    assert anthropic_span.attributes["llm.request.functions.1.name"] == "get_time"
    assert (
        anthropic_span.attributes["llm.request.functions.1.description"]
        == "Get the current time in a given time zone"
    )
    assert anthropic_span.attributes[
        "llm.request.functions.1.input_schema"
    ] == json.dumps(
        {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The IANA time zone name, e.g. America/Los_Angeles",
                }
            },
            "required": ["timezone"],
        }
    )

    # verify response and output
    assert (
        anthropic_span.attributes["gen_ai.completion.0.finish_reason"]
        == response.stop_reason
    )
    assert "gen_ai.completion.0.content" not in anthropic_span.attributes
    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.id"]
    ) == response.content[0].id
    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.name"]
    ) == response.content[0].name
    response_input = json.dumps(response.content[0].input)
    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.arguments"]
        == response_input
    )

    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_01QJDheQSo4hSrxgtLpEJFkA"
    )


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
    # verify overall shape
    assert all(span.name == "anthropic.chat" for span in spans)
    assert len(spans) == 1

    anthropic_span = spans[0]

    # verify usage
    assert anthropic_span.attributes["gen_ai.usage.prompt_tokens"] == 506
    assert (
        anthropic_span.attributes["gen_ai.usage.completion_tokens"]
        + anthropic_span.attributes["gen_ai.usage.prompt_tokens"]
        == anthropic_span.attributes["llm.usage.total_tokens"]
    )

    # verify request and inputs
    assert (
        anthropic_span.attributes["gen_ai.prompt.0.content"]
        == "What is the weather and current time in San Francisco?"
    )
    assert anthropic_span.attributes["gen_ai.prompt.0.role"] == "user"
    assert anthropic_span.attributes["llm.request.functions.0.name"] == "get_weather"
    assert (
        anthropic_span.attributes["llm.request.functions.0.description"]
        == "Get the current weather in a given location"
    )
    assert anthropic_span.attributes[
        "llm.request.functions.0.input_schema"
    ] == json.dumps(
        {
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
        }
    )
    assert anthropic_span.attributes["llm.request.functions.1.name"] == "get_time"
    assert (
        anthropic_span.attributes["llm.request.functions.1.description"]
        == "Get the current time in a given time zone"
    )
    assert anthropic_span.attributes[
        "llm.request.functions.1.input_schema"
    ] == json.dumps(
        {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The IANA time zone name, e.g. America/Los_Angeles",
                }
            },
            "required": ["timezone"],
        }
    )

    # verify response and output
    assert (
        anthropic_span.attributes["gen_ai.completion.0.content"]
        == "Certainly! I can help you with that information. To get the weather and current time in San Francisco, I'll need to use two separate functions. Let me fetch that data for you."
    )
    assert anthropic_span.attributes["gen_ai.completion.0.role"] == "assistant"

    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.id"]
    ) == "toolu_0121kXsENLvoDZ72LCuAnCCz"
    assert (
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.name"]
    ) == "get_time"
    assert json.loads(
        anthropic_span.attributes["gen_ai.completion.0.tool_calls.0.arguments"]
    ) == {"timezone": "America/Los_Angeles"}

    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "msg_0138UNF3YbNp49KkqZtUBWqz"
    )


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
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response_content
    )
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role")
        == "assistant"
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
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "user"
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        == response_content
    )
    assert (
        anthropic_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.role")
        == "assistant"
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

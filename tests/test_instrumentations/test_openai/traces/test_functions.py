import json

import pytest
from opentelemetry.semconv_ai import SpanAttributes


@pytest.fixture
def openai_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
    ]


@pytest.mark.vcr
def test_open_ai_function_calls(instrument_legacy, span_exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        function_call="auto",
    )

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]
    input_messages = json.loads(open_ai_span.attributes["gen_ai.input.messages"])
    assert input_messages[0]["content"] == "What's the weather like in Boston?"
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.description"]
        == "Get the current weather in a given location"
    )
    output_messages = json.loads(open_ai_span.attributes["gen_ai.output.messages"])
    assert (
        output_messages[0]["message"]["function_call"]["name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-8wq4AUDD36geK9Za8cccowhObkV9H"
    )


@pytest.mark.vcr
def test_open_ai_function_calls_tools(
    instrument_legacy, span_exporter, openai_client, openai_tools
):
    openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        tools=openai_tools,
        tool_choice="auto",
    )

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]
    input_messages = json.loads(open_ai_span.attributes["gen_ai.input.messages"])
    assert input_messages[0]["content"] == "What's the weather like in Boston?"
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.description"]
        == "Get the current weather"
    )
    output_messages = json.loads(open_ai_span.attributes["gen_ai.output.messages"])
    assert isinstance(
        output_messages[0]["message"]["tool_calls"][0]["id"],
        str,
    )
    assert (
        output_messages[0]["message"]["tool_calls"][0]["function"]["name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_OPENAI_API_BASE]
        == "https://api.openai.com/v1/"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-934OqhoorTmk1VnovIRXQCPk8PUTd"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_open_ai_function_calls_tools_streaming(
    instrument_legacy, span_exporter, async_openai_client, openai_tools
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ],
        tools=openai_tools,
        stream=True,
    )

    async for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]

    output_messages = json.loads(open_ai_span.attributes["gen_ai.output.messages"])
    assert isinstance(
        output_messages[0]["message"]["tool_calls"][0]["id"],
        str,
    )
    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name")
        == "get_current_weather"
    )
    assert output_messages[0]["finish_reason"] == "tool_calls"
    assert (
        output_messages[0]["message"]["tool_calls"][0]["function"]["name"]
        == "get_current_weather"
    )
    assert (
        output_messages[0]["message"]["tool_calls"][0]["function"]["arguments"]
        == '{"location":"San Francisco, CA"}'
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9g4TmLd49mPoD6c0EnGlhNAp8b0on"
    )


@pytest.mark.vcr
def test_open_ai_function_calls_tools_parallel(
    instrument_legacy, span_exporter, openai_client, openai_tools
):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in San Francisco and Boston?",
            }
        ],
        tools=openai_tools,
    )

    for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name")
        == "get_current_weather"
    )
    output_messages = json.loads(open_ai_span.attributes["gen_ai.output.messages"])
    assert output_messages[0]["finish_reason"] == "tool_calls"

    assert isinstance(
        output_messages[0]["message"]["tool_calls"][0]["id"],
        str,
    )
    assert (
        output_messages[0]["message"]["tool_calls"][0]["function"]["name"]
        == "get_current_weather"
    )
    assert (
        output_messages[0]["message"]["tool_calls"][0]["function"]["arguments"]
        == '{"location": "San Francisco"}'
    )

    assert isinstance(
        output_messages[0]["message"]["tool_calls"][1]["id"],
        str,
    )
    assert (
        output_messages[0]["message"]["tool_calls"][1]["function"]["name"]
        == "get_current_weather"
    )
    assert (
        output_messages[0]["message"]["tool_calls"][1]["function"]["arguments"]
        == '{"location": "Boston"}'
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9g4cZhrW9CsqihSvXslk0EUtjASsO"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_open_ai_function_calls_tools_streaming_parallel(
    instrument_legacy, span_exporter, async_openai_client, openai_tools
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in San Francisco and Boston?",
            }
        ],
        tools=openai_tools,
        stream=True,
    )

    async for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]

    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name")
        == "get_current_weather"
    )
    output_messages = json.loads(open_ai_span.attributes["gen_ai.output.messages"])
    assert output_messages[0]["finish_reason"] == "tool_calls"

    assert isinstance(
        output_messages[0]["message"]["tool_calls"][0]["id"],
        str,
    )
    assert (
        output_messages[0]["message"]["tool_calls"][0]["function"]["name"]
        == "get_current_weather"
    )
    assert (
        output_messages[0]["message"]["tool_calls"][0]["function"]["arguments"]
        == '{"location": "San Francisco"}'
    )

    assert isinstance(
        output_messages[0]["message"]["tool_calls"][1]["id"],
        str,
    )
    assert (
        output_messages[0]["message"]["tool_calls"][1]["function"]["name"]
        == "get_current_weather"
    )
    assert (
        output_messages[0]["message"]["tool_calls"][1]["function"]["arguments"]
        == '{"location": "Boston"}'
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9g58noIjRkOeNNxfFsFfcNjhXlul7"
    )

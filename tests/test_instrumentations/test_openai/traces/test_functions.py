import json
from typing import Any, cast

import pytest
from openai import AsyncOpenAI, OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import (
    OpenAIInstrumentor,
)


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
def test_open_ai_function_calls(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    functions = [
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
    ]
    _ = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        functions=functions,  # pyright: ignore[reportArgumentType]
        function_call="auto",
    )

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "What's the weather like in Boston?"
    assert json.loads(cast(str, attributes["gen_ai.tool.definitions"])) == functions
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert (
        output_messages[0]["message"]["function_call"]["name"] == "get_current_weather"
    )
    assert (
        attributes["gen_ai.request.base_url"]
        == "https://api.openai.com/v1/"
    )
    assert (
        attributes.get("gen_ai.response.id")
        == "chatcmpl-8wq4AUDD36geK9Za8cccowhObkV9H"
    )


@pytest.mark.vcr
def test_open_ai_function_calls_tools(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
    openai_tools: Any,
):
    _ = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        tools=openai_tools,
        tool_choice="auto",
    )

    spans = span_exporter.get_finished_spans()
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "What's the weather like in Boston?"
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert isinstance(
        output_messages[0]["message"]["tool_calls"][0]["id"],
        str,
    )
    assert (
        output_messages[0]["message"]["tool_calls"][0]["function"]["name"]
        == "get_current_weather"
    )
    assert (
        json.loads(cast(str, attributes["gen_ai.tool.definitions"])) == openai_tools
    )
    assert (
        attributes["gen_ai.request.base_url"]
        == "https://api.openai.com/v1/"
    )
    assert (
        attributes.get("gen_ai.response.id")
        == "chatcmpl-934OqhoorTmk1VnovIRXQCPk8PUTd"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_open_ai_function_calls_tools_streaming(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
    openai_tools: Any,
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
    attributes = open_ai_span.attributes or {}

    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert isinstance(
        output_messages[0]["message"]["tool_calls"][0]["id"],
        str,
    )
    assert (
        json.loads(cast(str, attributes["gen_ai.tool.definitions"])) == openai_tools
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
        attributes.get("gen_ai.response.id")
        == "chatcmpl-9g4TmLd49mPoD6c0EnGlhNAp8b0on"
    )


@pytest.mark.vcr
def test_open_ai_function_calls_tools_parallel(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
    openai_tools: Any,
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
    attributes = open_ai_span.attributes or {}
    assert (
        json.loads(cast(str, attributes["gen_ai.tool.definitions"])) == openai_tools
    )

    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
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
        attributes.get("gen_ai.response.id")
        == "chatcmpl-9g4cZhrW9CsqihSvXslk0EUtjASsO"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_open_ai_function_calls_tools_streaming_parallel(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
    openai_tools: Any,
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
    attributes = open_ai_span.attributes or {}

    assert (
        json.loads(cast(str, attributes["gen_ai.tool.definitions"])) == openai_tools
    )

    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
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
        attributes.get("gen_ai.response.id")
        == "chatcmpl-9g58noIjRkOeNNxfFsFfcNjhXlul7"
    )

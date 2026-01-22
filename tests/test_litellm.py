import asyncio
import base64
import json
import os
import pytest
import uuid

import litellm

from importlib.metadata import version

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from lmnr import LaminarLiteLLMCallback
from lmnr.version import __version__


class Event(BaseModel):
    name: str
    people: list[str]
    dayOfWeek: str


schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "event",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "title": "Event",
            "properties": {
                "name": {"type": "string", "description": "The name of the event"},
                "people": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "The name of the person attending the event",
                    },
                },
                "dayOfWeek": {
                    "type": "string",
                    "description": "The day of the week the event is on",
                },
            },
            "required": ["name", "people", "dayOfWeek"],
        },
    },
}

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location of the weather to get, e.g. San Francisco, CA or Paris, France",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "name": "get_news",
        "function": {
            "name": "get_news",
            "description": "Get the latest news for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location of the news to get, e.g. San Francisco, CA or Paris, France",
                    }
                },
                "required": ["location"],
            },
        },
    },
]

responses_tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Location of the weather to get, e.g. San Francisco, CA or Paris, France",
                }
            },
            "required": ["location"],
        },
    },
    {
        "type": "function",
        "name": "get_news",
        "description": "Get the latest news for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Location of the news to get, e.g. San Francisco, CA or Paris, France",
                }
            },
            "required": ["location"],
        },
    },
]


def check_span_has_basic_attributes(span: ReadableSpan):
    assert span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert span.attributes["lmnr.span.sdk_version"] == __version__
    assert span.attributes["lmnr.span.instrumentation_scope.name"] == "litellm"
    assert span.attributes["lmnr.span.instrumentation_scope.version"] == version(
        "litellm"
    )


def is_encoded_litellm_response_id(response_id: str) -> bool:
    """
    Check if a response ID is base64-encoded by litellm.

    Encoded IDs will have base64 characters after 'resp_' prefix and
    can be successfully decoded to contain semicolons.
    """
    if not response_id.startswith("resp_"):
        return False

    cleaned_id = response_id.replace("resp_", "")
    try:
        decoded = base64.b64decode(cleaned_id.encode("utf-8")).decode("utf-8")
        # Litellm encoded IDs contain semicolons separating the parts
        return ";" in decoded
    except Exception:
        return False


def extract_original_response_id(response_id: str) -> str:
    """
    Extract the original response ID from a litellm-encoded response ID.

    If the ID is not encoded, returns it as-is.
    Format: resp_<base64_of_"litellm:custom_llm_provider:{};model_id:{};response_id:{}">
    """
    if not is_encoded_litellm_response_id(response_id):
        return response_id

    cleaned_id = response_id.replace("resp_", "")
    decoded = base64.b64decode(cleaned_id.encode("utf-8")).decode("utf-8")

    # Parse format: litellm:custom_llm_provider:{};model_id:{};response_id:{}
    parts = decoded.split(";")
    if len(parts) >= 3:
        response_part = parts[2]
        return response_part.replace("response_id:", "")

    return response_id


@pytest.mark.vcr
def test_litellm_completion_basic(span_exporter: InMemorySpanExporter):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = litellm.completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 8
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 8
    assert spans[0].attributes["llm.usage.total_tokens"] == 16
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": "The capital of France is **Paris**.",
            "tool_calls": None,
            "function_call": None,
            "images": [],
            "thinking_blocks": [],
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ]
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.vcr(record_mode="once")
@pytest.mark.asyncio
async def test_litellm_completion_callback_doesnt_create_double_spans(
    span_exporter: InMemorySpanExporter, litellm_callback: LaminarLiteLLMCallback
):
    litellm.callbacks = [litellm_callback]
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"

    await litellm.acompletion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    # litellm callback is slow, so we need to sleep for a bit for this test to
    # properly catch errors and not dump a lot of errors to the console
    await asyncio.sleep(0.2)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    check_span_has_basic_attributes(spans[0])


@pytest.mark.vcr(record_mode="once")
def test_litellm_completion_streaming(span_exporter: InMemorySpanExporter):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = litellm.completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
    )
    response_id = None
    for chunk in response:
        if chunk.id:
            response_id = chunk.id
        # consume the stream
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response_id
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": "The capital of France is **Paris**.",
            "tool_calls": None,
            "finish_reason": "stop",
            "index": 0,
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ]
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.vcr(record_mode="once")
def test_litellm_completion_streaming_openai(span_exporter: InMemorySpanExporter):
    # LiteLLM calls OpenAI through OpenAI SDK, and other providers via raw HTTP,
    # so for streaming, we return an async generator from OpenAI SDK; thus, separate test.
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test-key"
    response = litellm.completion(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
    )
    response_id = None
    for chunk in response:
        if chunk.id:
            response_id = chunk.id
        # consume the stream
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.id"] == response_id
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": "The capital of France is Paris.",
            "tool_calls": None,
            "finish_reason": "stop",
            "index": 0,
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ]
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.vcr(record_mode="once")
def test_litellm_completion_with_structured_output(span_exporter: InMemorySpanExporter):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = litellm.completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": "Alice and Bob go to a Science fair on Friday. Extract the event information.",
            }
        ],
        response_format=schema,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 17
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 32
    assert spans[0].attributes["llm.usage.total_tokens"] == 49
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": '{\n  "name": "Science fair",\n  "people": ["Alice", "Bob"],\n  "dayOfWeek": "Friday"\n}',
            "tool_calls": None,
            "function_call": None,
            "images": [],
            "thinking_blocks": [],
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "Alice and Bob go to a Science fair on Friday. Extract the event information.",
        }
    ]
    assert (
        json.loads(spans[0].attributes["gen_ai.request.structured_output_schema"])
        == schema
    )
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.vcr(record_mode="once")
def test_litellm_completion_with_structured_output_pydantic(
    span_exporter: InMemorySpanExporter,
):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = litellm.completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": "Alice and Bob go to a Science fair on Friday. Extract the event information.",
            }
        ],
        response_format=Event,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 17
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 32
    assert spans[0].attributes["llm.usage.total_tokens"] == 49
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": '{\n  "name": "Science fair",\n  "people": ["Alice", "Bob"],\n  "dayOfWeek": "Friday"\n}',
            "tool_calls": None,
            "function_call": None,
            "images": [],
            "thinking_blocks": [],
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "Alice and Bob go to a Science fair on Friday. Extract the event information.",
        }
    ]
    assert (
        json.loads(spans[0].attributes["gen_ai.request.structured_output_schema"])
        == Event.model_json_schema()
    )
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.vcr(record_mode="once")
def test_litellm_completion_with_tool_call(span_exporter: InMemorySpanExporter):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = litellm.completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Tokyo?",
            }
        ],
        tools=[tools[0]],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 65
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 15
    assert spans[0].attributes["llm.usage.total_tokens"] == 80
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": response.choices[0].message.tool_calls[0].id,
                    "index": 0,
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    },
                }
            ],
            "function_call": None,
            "images": [],
            "thinking_blocks": [],
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the weather in Tokyo?",
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.tool.definitions"]) == [tools[0]]
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.vcr(record_mode="once")
def test_litellm_completion_with_parallel_tool_calls(
    span_exporter: InMemorySpanExporter,
):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Tokyo? What are the latest news there?",
            }
        ],
        tools=tools,
        thinking={
            "type": "enabled",
            "budget_tokens": 0,
        },
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 129
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 34
    assert spans[0].attributes["llm.usage.total_tokens"] == 163
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": response.choices[0].message.tool_calls[0].id,
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo, Japan"}',
                    },
                    "index": 0,
                    "type": "function",
                },
                {
                    "id": response.choices[0].message.tool_calls[1].id,
                    "function": {
                        "name": "get_news",
                        "arguments": '{"location": "Tokyo, Japan"}',
                    },
                    "index": 1,
                    "type": "function",
                },
            ],
            "function_call": None,
            "images": [],
            "thinking_blocks": [],
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the weather in Tokyo? What are the latest news there?",
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.tool.definitions"]) == tools
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    assert spans[0].attributes["lmnr.span.instrumentation_source"] == "python"
    assert spans[0].attributes["lmnr.span.sdk_version"] == __version__


@pytest.mark.vcr(record_mode="once")
def test_litellm_completion_with_tool_call_history(span_exporter: InMemorySpanExporter):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"

    response1 = litellm.completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Tokyo?",
            }
        ],
        tools=[tools[0]],
    )
    response2 = litellm.completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Tokyo?",
            },
            response1.choices[0].message,
            {
                "role": "tool",
                "content": "Sunny, 22°C.",
                "tool_call_id": response1.choices[0].message.tool_calls[0].id,
            },
        ],
        tools=[tools[0]],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span1 = sorted(spans, key=lambda x: x.start_time)[0]
    span2 = sorted(spans, key=lambda x: x.start_time)[1]

    for span in spans:
        assert span.name == "litellm.completion"
        assert span.attributes["gen_ai.system"] == "gemini"
        assert span.attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash-lite"
        assert span.attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
        assert json.loads(span.attributes["gen_ai.tool.definitions"]) == [tools[0]]
        assert span.attributes["lmnr.span.path"] == ("litellm.completion",)
        assert span.attributes["lmnr.span.ids_path"] == (
            str(uuid.UUID(int=span.get_span_context().span_id)),
        )
        check_span_has_basic_attributes(span)

    assert span1.attributes["gen_ai.response.id"] == response1.id
    assert span2.attributes["gen_ai.response.id"] == response2.id

    assert span1.attributes["gen_ai.usage.input_tokens"] == 65
    assert span1.attributes["gen_ai.usage.output_tokens"] == 15
    assert span1.attributes["llm.usage.total_tokens"] == 80

    assert span2.attributes["gen_ai.usage.input_tokens"] == 101
    assert span2.attributes["gen_ai.usage.output_tokens"] == 13
    assert span2.attributes["llm.usage.total_tokens"] == 114

    assert json.loads(span1.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": response1.choices[0].message.tool_calls[0]["id"],
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    },
                    "index": 0,
                    "type": "function",
                }
            ],
            "function_call": None,
            "images": [],
            "thinking_blocks": [],
        }
    ]
    assert json.loads(span1.attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the weather in Tokyo?",
        }
    ]

    assert json.loads(span2.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": "The weather in Tokyo is sunny and 22°C.",
            "tool_calls": None,
            "function_call": None,
            "images": [],
            "thinking_blocks": [],
        }
    ]
    assert json.loads(span2.attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the weather in Tokyo?",
        },
        response1.choices[0].message.model_dump(mode="json"),
        {
            "role": "tool",
            "content": "Sunny, 22°C.",
            "tool_call_id": response1.choices[0].message.tool_calls[0]["id"],
        },
    ]


@pytest.mark.vcr(record_mode="once")
def test_litellm_completion_streaming_with_tool_call(
    span_exporter: InMemorySpanExporter,
):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = litellm.completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Tokyo?",
            }
        ],
        stream=True,
        tools=[tools[0]],
    )

    response_id = None
    for chunk in response:
        if chunk.id:
            response_id = chunk.id
        if chunk.choices[0].delta.tool_calls:
            tool_call_id = chunk.choices[0].delta.tool_calls[0].id
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response_id
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": None,
            "finish_reason": "tool_calls",
            "index": 0,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    },
                    "index": 0,
                    "type": "function",
                }
            ],
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the weather in Tokyo?",
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.tool.definitions"]) == [tools[0]]
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.asyncio
@pytest.mark.vcr(record_mode="once")
async def test_litellm_completion_basic_async(span_exporter: InMemorySpanExporter):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = await litellm.acompletion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 8
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 8
    assert spans[0].attributes["llm.usage.total_tokens"] == 16
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": "The capital of France is **Paris**.",
            "tool_calls": None,
            "function_call": None,
            "images": [],
            "thinking_blocks": [],
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ]
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.asyncio
@pytest.mark.vcr(record_mode="once")
async def test_litellm_completion_streaming_async(span_exporter: InMemorySpanExporter):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = await litellm.acompletion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
    )
    response_id = None
    async for chunk in response:
        if chunk.id:
            response_id = chunk.id
        # consume the stream
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response_id
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": "The capital of France is **Paris**.",
            "tool_calls": None,
            "finish_reason": "stop",
            "index": 0,
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ]
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.asyncio
@pytest.mark.vcr(record_mode="once")
async def test_litellm_completion_streaming_async_openai(
    span_exporter: InMemorySpanExporter,
):
    # LiteLLM calls OpenAI through OpenAI SDK, and other providers via raw HTTP,
    # so for streaming, we return an async generator from OpenAI SDK; thus, separate test.
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test-key"
    response = await litellm.acompletion(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
    )
    response_id = None
    async for chunk in response:
        if chunk.id:
            response_id = chunk.id
        # consume the stream
        pass

    spans = span_exporter.get_finished_spans()
    print([span.name for span in spans])
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.id"] == response_id
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": "The capital of France is Paris.",
            "tool_calls": None,
            "finish_reason": "stop",
            "index": 0,
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ]
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.asyncio
@pytest.mark.vcr(record_mode="once")
async def test_litellm_completion_with_structured_output_async(
    span_exporter: InMemorySpanExporter,
):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = await litellm.acompletion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": "Alice and Bob go to a Science fair on Friday. Extract the event information.",
            }
        ],
        response_format=schema,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 17
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 29
    assert spans[0].attributes["llm.usage.total_tokens"] == 46
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": '{\n"name": "Science fair",\n"people": ["Alice", "Bob"],\n"dayOfWeek": "Friday"\n}',
            "tool_calls": None,
            "function_call": None,
            "images": [],
            "thinking_blocks": [],
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "Alice and Bob go to a Science fair on Friday. Extract the event information.",
        }
    ]
    assert (
        json.loads(spans[0].attributes["gen_ai.request.structured_output_schema"])
        == schema
    )
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.asyncio
@pytest.mark.vcr(record_mode="once")
async def test_litellm_completion_with_structured_output_pydantic_async(
    span_exporter: InMemorySpanExporter,
):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = await litellm.acompletion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": "Alice and Bob go to a Science fair on Friday. Extract the event information.",
            }
        ],
        response_format=Event,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 17
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 32
    assert spans[0].attributes["llm.usage.total_tokens"] == 49
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": '{\n  "name": "Science fair",\n  "people": ["Alice", "Bob"],\n  "dayOfWeek": "Friday"\n}',
            "tool_calls": None,
            "function_call": None,
            "images": [],
            "thinking_blocks": [],
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "Alice and Bob go to a Science fair on Friday. Extract the event information.",
        }
    ]
    assert (
        json.loads(spans[0].attributes["gen_ai.request.structured_output_schema"])
        == Event.model_json_schema()
    )
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.asyncio
@pytest.mark.vcr(record_mode="once")
async def test_litellm_completion_with_tool_call_async(
    span_exporter: InMemorySpanExporter,
):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = await litellm.acompletion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Tokyo?",
            }
        ],
        tools=[tools[0]],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 65
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 15
    assert spans[0].attributes["llm.usage.total_tokens"] == 80
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": response.choices[0].message.tool_calls[0].id,
                    "index": 0,
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    },
                }
            ],
            "function_call": None,
            "images": [],
            "thinking_blocks": [],
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the weather in Tokyo?",
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.tool.definitions"]) == [tools[0]]
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.asyncio
@pytest.mark.vcr(record_mode="once")
async def test_litellm_completion_with_parallel_tool_calls_async(
    span_exporter: InMemorySpanExporter,
):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = await litellm.acompletion(
        model="gemini/gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Tokyo? What are the latest news there?",
            }
        ],
        tools=tools,
        thinking={
            "type": "enabled",
            "budget_tokens": 0,
        },
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 129
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 34
    assert spans[0].attributes["llm.usage.total_tokens"] == 163
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": response.choices[0].message.tool_calls[0].id,
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo, Japan"}',
                    },
                    "index": 0,
                    "type": "function",
                },
                {
                    "id": response.choices[0].message.tool_calls[1].id,
                    "function": {
                        "name": "get_news",
                        "arguments": '{"location": "Tokyo, Japan"}',
                    },
                    "index": 1,
                    "type": "function",
                },
            ],
            "function_call": None,
            "images": [],
            "thinking_blocks": [],
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the weather in Tokyo? What are the latest news there?",
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.tool.definitions"]) == tools
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.asyncio
@pytest.mark.vcr(record_mode="once")
async def test_litellm_completion_streaming_with_tool_call_async(
    span_exporter: InMemorySpanExporter,
):
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key"
    response = await litellm.acompletion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Tokyo?",
            }
        ],
        stream=True,
        tools=[tools[0]],
    )

    response_id = None
    tool_call_id = None
    async for chunk in response:
        if chunk.id:
            response_id = chunk.id
        if chunk.choices[0].delta.tool_calls:
            tool_call_id = chunk.choices[0].delta.tool_calls[0].id
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.completion"
    assert spans[0].attributes["gen_ai.system"] == "gemini"
    assert spans[0].attributes["gen_ai.request.model"] == "gemini/gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    assert spans[0].attributes["gen_ai.response.id"] == response_id
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "content": None,
            "finish_reason": "tool_calls",
            "index": 0,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    },
                    "index": 0,
                    "type": "function",
                }
            ],
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "content": "What is the weather in Tokyo?",
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.tool.definitions"]) == [tools[0]]
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.completion",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.vcr(record_mode="once")
def test_litellm_responses_basic(span_exporter: InMemorySpanExporter):
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test-key"

    response = litellm.responses(
        model="gpt-4.1-nano",
        input=[
            {"content": "Be very crisp in your response.", "role": "system"},
            {"content": "What is the capital of France?", "role": "user"},
        ],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "id": response.output[0].id,
            "content": [
                {
                    "annotations": [],
                    "text": "Paris.",
                    "type": "output_text",
                    "logprobs": [],
                }
            ],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {"content": "Be very crisp in your response.", "role": "system"},
        {"content": "What is the capital of France?", "role": "user"},
    ]
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 25
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 3
    assert spans[0].attributes["llm.usage.total_tokens"] == 28
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.responses",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.vcr(record_mode="once")
def test_litellm_responses_streaming(span_exporter: InMemorySpanExporter):
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test-key"

    response = litellm.responses(
        model="gpt-4.1-nano",
        input=[
            {"content": "Be very crisp in your response.", "role": "system"},
            {"content": "What is the capital of France?", "role": "user"},
        ],
        stream=True,
    )
    final_response = None
    for chunk in response:
        if chunk.type == "response.completed":
            final_response = chunk.response

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.id"] == final_response.id
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "id": final_response.output[0].id,
            "content": [
                {
                    "annotations": [],
                    "text": "Paris",
                    "type": "output_text",
                    "logprobs": [],
                }
            ],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {"content": "Be very crisp in your response.", "role": "system"},
        {"content": "What is the capital of France?", "role": "user"},
    ]
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 25
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 2
    assert spans[0].attributes["llm.usage.total_tokens"] == 27
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.responses",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.vcr(record_mode="once")
def test_litellm_responses_with_tool_call(span_exporter: InMemorySpanExporter):
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test-key"

    response = litellm.responses(
        model="gpt-4.1-nano",
        input=[
            {"content": "Be very crisp in your response.", "role": "system"},
            {"content": "What is the weather in Tokyo?", "role": "user"},
        ],
        tools=[responses_tools[0]],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.id"] == response.id
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "arguments": '{"location":"Tokyo"}',
            "call_id": response.output[0].call_id,
            "name": "get_weather",
            "type": "function_call",
            "id": response.output[0].id,
            "status": "completed",
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {"content": "Be very crisp in your response.", "role": "system"},
        {"content": "What is the weather in Tokyo?", "role": "user"},
    ]
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 77
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 31
    assert spans[0].attributes["llm.usage.total_tokens"] == 108
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.responses",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.asyncio
@pytest.mark.vcr(record_mode="once")
async def test_litellm_responses_basic_async(span_exporter: InMemorySpanExporter):
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test-key"

    response = await litellm.aresponses(
        model="gpt-4.1-nano",
        input=[
            {"content": "Be very crisp in your response.", "role": "system"},
            {"content": "What is the capital of France?", "role": "user"},
        ],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.id"] == extract_original_response_id(
        response.id
    )
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "id": response.output[0].id,
            "content": [
                {
                    "annotations": [],
                    "text": "Paris",
                    "type": "output_text",
                    "logprobs": [],
                }
            ],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {"content": "Be very crisp in your response.", "role": "system"},
        {"content": "What is the capital of France?", "role": "user"},
    ]
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 25
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 2
    assert spans[0].attributes["llm.usage.total_tokens"] == 27
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.responses",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.asyncio
@pytest.mark.vcr(record_mode="once")
async def test_litellm_responses_streaming_async(span_exporter: InMemorySpanExporter):
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test-key"

    response = await litellm.aresponses(
        model="gpt-4.1-nano",
        input=[
            {"content": "Be very crisp in your response.", "role": "system"},
            {"content": "What is the capital of France?", "role": "user"},
        ],
        stream=True,
    )
    final_response = None
    async for chunk in response:
        if chunk.type == "response.completed":
            final_response = chunk.response

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    # In streaming, we only have a chance to get the response ID after the LiteLLM magic,
    # so we check for the exact match. This is not too critical, users may extract the response ID manually,
    # if needed for debugging purposes.
    assert spans[0].attributes["gen_ai.response.id"] == final_response.id
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "id": final_response.output[0].id,
            "content": [
                {
                    "annotations": [],
                    "text": "Paris",
                    "type": "output_text",
                    "logprobs": [],
                }
            ],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {"content": "Be very crisp in your response.", "role": "system"},
        {"content": "What is the capital of France?", "role": "user"},
    ]
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 25
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 2
    assert spans[0].attributes["llm.usage.total_tokens"] == 27
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.responses",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])


@pytest.mark.asyncio
@pytest.mark.vcr(record_mode="once")
async def test_litellm_responses_with_tool_call_async(
    span_exporter: InMemorySpanExporter,
):
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test-key"

    response = await litellm.aresponses(
        model="gpt-4.1-nano",
        input=[
            {"content": "Be very crisp in your response.", "role": "system"},
            {"content": "What is the weather in Tokyo?", "role": "user"},
        ],
        tools=[responses_tools[0]],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "litellm.responses"
    assert spans[0].attributes["gen_ai.system"] == "openai"
    assert spans[0].attributes["gen_ai.request.model"] == "gpt-4.1-nano"
    assert spans[0].attributes["gen_ai.response.id"] == extract_original_response_id(
        response.id
    )
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "arguments": '{"location":"Tokyo"}',
            "call_id": response.output[0].call_id,
            "name": "get_weather",
            "type": "function_call",
            "id": response.output[0].id,
            "status": "completed",
        }
    ]
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {"content": "Be very crisp in your response.", "role": "system"},
        {"content": "What is the weather in Tokyo?", "role": "user"},
    ]
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 77
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 31
    assert spans[0].attributes["llm.usage.total_tokens"] == 108
    assert spans[0].attributes["lmnr.span.path"] == ("litellm.responses",)
    assert spans[0].attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=spans[0].get_span_context().span_id)),
    )
    check_span_has_basic_attributes(spans[0])

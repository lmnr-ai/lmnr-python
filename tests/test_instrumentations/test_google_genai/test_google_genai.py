import base64
import json
from typing import cast

import httpx
import pydantic
import pytest
from google.genai import Client, types
from google.genai.errors import ClientError
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
    merge_text_parts,
)

image_url = "https://upload.wikimedia.org/wikipedia/commons/8/8e/MuseumOfFineArtsBoston_CopleySquare_19thc.jpg"
image_media_type = "image/jpeg"
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
image_data_raw_bytes = base64.b64encode(httpx.get(image_url).content).decode()

get_weather_declaration: types.FunctionDeclaration = types.FunctionDeclaration(
    name="get_weather",
    description="Gets the weather in a given city.",
    parameters=types.Schema(
        type=types.Type("object"),
        properties={
            "location": types.Schema(
                type=types.Type("string"),
                description="The location to get the weather for.",
            ),
        },
        required=["location"],
    ),
)


@pytest.mark.vcr
def test_google_genai(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point. Use tools as much as possible."
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "What is the capital of France?"},
                ],
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": system_instruction},
        ),
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    assert (
        (spans[0].attributes or {})["gen_ai.request.model"] == "gemini-2.5-flash-preview-05-20"
    )
    assert (
        (spans[0].attributes or {})["gen_ai.response.model"]
        == "models/gemini-2.5-flash-preview-05-20"
    )
    messages = json.loads(cast(str, (spans[0].attributes or {})["gen_ai.input.messages"]))
    assert messages[0] == {"role": "system", "parts": [{"text": system_instruction}]}
    assert messages[1] == {
        "role": "user",
        "parts": [{"text": "What is the capital of France?"}],
    }
    assert json.loads(cast(str, (spans[0].attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {"role": "model", "parts": [{"text": response.text}]},
            "finish_reason": "STOP",
            "index": 0,
        }
    ]


@pytest.mark.vcr
def test_google_genai_multiturn(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point. Use tools as much as possible."
    adjective_response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Come up with an adjective in English. Respond with only the adjective."
                    },
                ],
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": system_instruction},
        ),
    )
    haiku_response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Come up with an adjective in English. Respond with only the adjective."
                    },
                ],
            },
            {
                "role": "model",
                "parts": [
                    {"text": adjective_response.text},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"text": "Now generate a haiku using this adjective."},
                ],
            },
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": system_instruction},
        ),
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        assert span.name == "gemini.generate_content"
        assert (
            (span.attributes or {})["gen_ai.request.model"] == "gemini-2.5-flash-preview-05-20"
        )
        assert (
            (span.attributes or {})["gen_ai.response.model"]
            == "models/gemini-2.5-flash-preview-05-20"
        )
        input_messages = json.loads(cast(str, (span.attributes or {})["gen_ai.input.messages"]))
        assert input_messages[0] == {
            "role": "system",
            "parts": [{"text": system_instruction}],
        }

    spans = sorted(spans, key=lambda x: x.start_time or 0)
    adjective_span = spans[0]
    haiku_span = spans[1]

    adj_messages = json.loads(cast(str, (adjective_span.attributes or {})["gen_ai.input.messages"]))
    assert adj_messages[1] == {
        "role": "user",
        "parts": [
            {
                "text": "Come up with an adjective in English. Respond with only the adjective."
            }
        ],
    }
    assert json.loads(cast(str, (adjective_span.attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {"role": "model", "parts": [{"text": adjective_response.text}]},
            "finish_reason": "STOP",
            "index": 0,
        }
    ]

    haiku_messages = json.loads(cast(str, (haiku_span.attributes or {})["gen_ai.input.messages"]))
    assert haiku_messages[1] == {
        "role": "user",
        "parts": [
            {
                "text": "Come up with an adjective in English. Respond with only the adjective."
            }
        ],
    }
    assert haiku_messages[2]["role"] == "model"
    assert haiku_messages[2]["parts"][0]["text"] == adjective_response.text
    assert haiku_messages[3] == {
        "role": "user",
        "parts": [{"text": "Now generate a haiku using this adjective."}],
    }
    assert json.loads(cast(str, (haiku_span.attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {"role": "model", "parts": [{"text": haiku_response.text}]},
            "finish_reason": "STOP",
            "index": 0,
        }
    ]


@pytest.mark.vcr
def test_google_genai_tool_calls(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point. Use tools as much as possible."
    _res = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "What is the weather in Tokyo?"},
                ],
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": system_instruction},
            tools=[types.Tool(function_declarations=[get_weather_declaration])],
        ),
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    assert (spans[0].attributes or {})["gen_ai.request.model"] == "gemini-2.5-flash-lite"
    assert (spans[0].attributes or {})["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    messages = json.loads(cast(str, (spans[0].attributes or {})["gen_ai.input.messages"]))
    assert messages[0] == {"role": "system", "parts": [{"text": system_instruction}]}
    assert messages[1] == {
        "role": "user",
        "parts": [{"text": "What is the weather in Tokyo?"}],
    }
    assert json.loads(cast(str, (spans[0].attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "name": "get_weather",
                            "args": {
                                "location": "Tokyo",
                            },
                        }
                    }
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
    ]


@pytest.mark.vcr(record_mode="once")
def test_google_genai_tool_calls_history(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point. Use tools as much as possible."
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "What is the weather in Tokyo?"},
                ],
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": system_instruction},
            tools=[types.Tool(function_declarations=[get_weather_declaration])],
        ),
    )
    _res = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "What is the weather in Tokyo?"},
                ],
            },
            {
                "role": "model",
                "parts": response.parts,
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": "get_weather",
                            "response": {"output": "Sunny, 22°C."},
                        }
                    }
                ],
            },
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": system_instruction},
            tools=[types.Tool(function_declarations=[get_weather_declaration])],
        ),
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span1 = min(spans, key=lambda x: x.start_time or 0)
    span2 = sorted(spans, key=lambda x: x.start_time or 0)[1]

    assert json.loads(cast(str, (span1.attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "name": "get_weather",
                            "args": {
                                "location": "Tokyo",
                            },
                        }
                    }
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
    ]

    for span in spans:
        assert span.name == "gemini.generate_content"
        assert (span.attributes or {})["gen_ai.request.model"] == "gemini-2.5-flash-lite"
        assert (span.attributes or {})["gen_ai.response.model"] == "gemini-2.5-flash-lite"
        input_messages = json.loads(cast(str, (span.attributes or {})["gen_ai.input.messages"]))
        assert input_messages[0] == {
            "role": "system",
            "parts": [{"text": system_instruction}],
        }
        assert input_messages[1] == {
            "role": "user",
            "parts": [{"text": "What is the weather in Tokyo?"}],
        }

    messages2 = json.loads(cast(str, (span2.attributes or {})["gen_ai.input.messages"]))
    assert messages2[2]["role"] == "model"
    assert messages2[2]["parts"][0]["function_call"]["name"] == "get_weather"
    assert messages2[2]["parts"][0]["function_call"]["args"] == {"location": "Tokyo"}
    assert messages2[3]["role"] == "user"
    assert messages2[3]["parts"] == [
        {
            "function_response": {
                "name": "get_weather",
                "response": {"output": "Sunny, 22°C."},
            }
        }
    ]
    assert json.loads(cast(str, (span2.attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": "The weather in Tokyo is sunny with a temperature of 22°C.",
                    }
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
    ]


@pytest.mark.vcr(record_mode="once")
def test_google_genai_tool_calls_history_from_function_response(
    span_exporter: InMemorySpanExporter,
):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point. Use tools as much as possible."
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "What is the weather in Tokyo?"},
                ],
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": system_instruction},
            tools=[types.Tool(function_declarations=[get_weather_declaration])],
        ),
    )
    _res = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "What is the weather in Tokyo?"},
                ],
            },
            {
                "role": "model",
                "parts": response.parts,
            },
            {
                "role": "user",
                "parts": [
                    types.Part.from_function_response(
                        name="get_weather",
                        response={"output": "Sunny, 22°C."},
                    )
                ],
            },
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": system_instruction},
            tools=[types.Tool(function_declarations=[get_weather_declaration])],
        ),
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span1 = min(spans, key=lambda x: x.start_time or 0)
    span2 = sorted(spans, key=lambda x: x.start_time or 0)[1]

    assert json.loads(cast(str, (span1.attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "name": "get_weather",
                            "args": {
                                "location": "Tokyo",
                            },
                        }
                    }
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
    ]

    for span in spans:
        assert span.name == "gemini.generate_content"
        assert (span.attributes or {})["gen_ai.request.model"] == "gemini-2.5-flash-lite"
        assert (span.attributes or {})["gen_ai.response.model"] == "gemini-2.5-flash-lite"
        input_messages = json.loads(cast(str, (span.attributes or {})["gen_ai.input.messages"]))
        assert input_messages[0] == {
            "role": "system",
            "parts": [{"text": system_instruction}],
        }
        assert input_messages[1] == {
            "role": "user",
            "parts": [{"text": "What is the weather in Tokyo?"}],
        }

    messages2 = json.loads(cast(str, (span2.attributes or {})["gen_ai.input.messages"]))
    assert messages2[2]["role"] == "model"
    assert messages2[2]["parts"][0]["function_call"]["name"] == "get_weather"
    assert messages2[2]["parts"][0]["function_call"]["args"] == {"location": "Tokyo"}
    assert messages2[3]["role"] == "user"
    assert messages2[3]["parts"][0]["function_response"] == {
        "name": "get_weather",
        "response": {"output": "Sunny, 22°C."},
    }
    assert json.loads(cast(str, (span2.attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": "The weather in Tokyo is sunny with a temperature of 22°C.",
                    }
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
    ]


@pytest.mark.vcr
def test_google_genai_multiple_tool_calls(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point. Use tools as much as possible."
    _res = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "What is the weather in Tokyo and Paris?"},
                ],
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": system_instruction},
            tools=[types.Tool(function_declarations=[get_weather_declaration])],
        ),
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    assert (
        (spans[0].attributes or {})["gen_ai.request.model"] == "gemini-2.5-flash-preview-05-20"
    )
    assert (
        (spans[0].attributes or {})["gen_ai.response.model"]
        == "models/gemini-2.5-flash-preview-05-20"
    )
    messages = json.loads(cast(str, (spans[0].attributes or {})["gen_ai.input.messages"]))
    assert messages[0] == {"role": "system", "parts": [{"text": system_instruction}]}
    assert messages[1] == {
        "role": "user",
        "parts": [{"text": "What is the weather in Tokyo and Paris?"}],
    }


@pytest.mark.vcr
def test_google_genai_tool_calls_and_text_part(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point"
    user_message = (
        "What is the opposite of 'bright'? Also, what is the weather in Tokyo?"
    )
    _res = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": user_message},
                ],
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": system_instruction},
            tools=[types.Tool(function_declarations=[get_weather_declaration])],
        ),
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    assert (spans[0].attributes or {})["gen_ai.request.model"] == "gemini-2.5-flash-lite"
    assert (spans[0].attributes or {})["gen_ai.response.model"] == "gemini-2.5-flash-lite"
    messages = json.loads(cast(str, (spans[0].attributes or {})["gen_ai.input.messages"]))
    assert messages[0] == {"role": "system", "parts": [{"text": system_instruction}]}
    assert messages[1] == {"role": "user", "parts": [{"text": user_message}]}
    assert json.loads(cast(str, (spans[0].attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": "The opposite of 'bright' is 'dim'.",
                    },
                    {
                        "function_call": {
                            "name": "get_weather",
                            "args": {"location": "Tokyo"},
                        }
                    },
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
    ]


@pytest.mark.vcr
def test_google_genai_image(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point. Use tools as much as possible."
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "Describe this image"},
                    {
                        "inline_data": {
                            "mime_type": image_media_type,
                            "data": image_data,
                        }
                    },
                ],
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": system_instruction},
        ),
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    assert (
        (spans[0].attributes or {})["gen_ai.request.model"] == "gemini-2.5-flash-preview-05-20"
    )
    assert (
        (spans[0].attributes or {})["gen_ai.response.model"]
        == "models/gemini-2.5-flash-preview-05-20"
    )
    messages = json.loads(cast(str, (spans[0].attributes or {})["gen_ai.input.messages"]))
    assert messages[0] == {"role": "system", "parts": [{"text": system_instruction}]}
    assert messages[1]["role"] == "user"
    assert messages[1]["parts"][0] == {"text": "Describe this image"}
    assert messages[1]["parts"][1] == {
        "inline_data": {"mime_type": image_media_type, "data": image_data}
    }
    assert json.loads(cast(str, (spans[0].attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": response.text,
                    }
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
    ]


@pytest.mark.vcr
def test_google_genai_image_raw_bytes(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point. Use tools as much as possible."
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "Describe this image"},
                    {
                        "inline_data": {
                            "mime_type": image_media_type,
                            "data": image_data_raw_bytes,
                        }
                    },
                ],
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": system_instruction},
        ),
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    assert (
        (spans[0].attributes or {})["gen_ai.request.model"] == "gemini-2.5-flash-preview-05-20"
    )
    assert (
        (spans[0].attributes or {})["gen_ai.response.model"]
        == "models/gemini-2.5-flash-preview-05-20"
    )
    messages = json.loads(cast(str, (spans[0].attributes or {})["gen_ai.input.messages"]))
    assert messages[0] == {"role": "system", "parts": [{"text": system_instruction}]}
    assert messages[1]["role"] == "user"
    assert messages[1]["parts"][0] == {"text": "Describe this image"}
    assert messages[1]["parts"][1] == {
        "inline_data": {"mime_type": image_media_type, "data": image_data_raw_bytes}
    }
    assert json.loads(cast(str, (spans[0].attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": response.text,
                    }
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
    ]


class CalendarEvent(pydantic.BaseModel):
    name: str
    dayOfWeek: str
    participants: list[str]


EXPECTED_SCHEMA = {
    "type": "object",
    "title": "CalendarEvent",
    "properties": {
        "name": {"type": "string", "title": "Name"},
        "dayOfWeek": {"type": "string", "title": "Dayofweek"},
        "participants": {
            "type": "array",
            "title": "Participants",
            "items": {"type": "string"},
        },
    },
    "required": ["name", "dayOfWeek", "participants"],
}


@pytest.mark.vcr
def test_google_genai_output_schema(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    prompt = "Alice and Bob are going to a science fair on Friday. Extract the event information."
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite-preview-06-17",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                ],
            }
        ],
        config=types.GenerateContentConfig(
            response_schema=CalendarEvent,
            response_mime_type="application/json",
        ),
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    assert (
        (spans[0].attributes or {})["gen_ai.request.model"]
        == "gemini-2.5-flash-lite-preview-06-17"
    )
    assert (
        (spans[0].attributes or {})["gen_ai.response.model"]
        == "gemini-2.5-flash-lite-preview-06-17"
    )

    messages = json.loads(cast(str, (spans[0].attributes or {})["gen_ai.input.messages"]))
    assert messages[0] == {"role": "user", "parts": [{"text": prompt}]}
    assert json.loads(cast(str, (spans[0].attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": response.text,
                    }
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
    ]
    assert (
        json.loads(cast(str, (spans[0].attributes or {})["gen_ai.request.structured_output_schema"]))
        == EXPECTED_SCHEMA
    )


@pytest.mark.vcr
def test_google_genai_output_json_schema(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    prompt = "Alice and Bob are going to a science fair on Friday. Extract the event information."
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite-preview-06-17",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                ],
            }
        ],
        config=types.GenerateContentConfig(
            response_json_schema=EXPECTED_SCHEMA,
            response_mime_type="application/json",
        ),
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    assert (
        (spans[0].attributes or {})["gen_ai.request.model"]
        == "gemini-2.5-flash-lite-preview-06-17"
    )
    assert (
        (spans[0].attributes or {})["gen_ai.response.model"]
        == "gemini-2.5-flash-lite-preview-06-17"
    )

    messages = json.loads(cast(str, (spans[0].attributes or {})["gen_ai.input.messages"]))
    assert messages[0] == {"role": "user", "parts": [{"text": prompt}]}

    assert json.loads(cast(str, (spans[0].attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": response.text,
                    }
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
    ]
    assert (
        json.loads(cast(str, (spans[0].attributes or {})["gen_ai.request.structured_output_schema"]))
        == EXPECTED_SCHEMA
    )


@pytest.mark.vcr
def test_google_genai_reasoning_tokens(span_exporter: InMemorySpanExporter):
    client = Client(api_key="123")
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "text": "How many times does the letter 'r' appear in the word strawberry?"
                    },
                ],
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": "Think deep and thoroughly step by step."},
            thinking_config=types.ThinkingConfig(thinking_budget=512),
        ),
    )

    usage = response.usage_metadata
    assert usage is not None
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    assert (
        (spans[0].attributes or {})["gen_ai.usage.reasoning_tokens"]
        == usage.thoughts_token_count
    )
    assert (
        (spans[0].attributes or {})["gen_ai.usage.output_tokens"]
        == (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    )
    assert (
        (spans[0].attributes or {})["gen_ai.usage.input_tokens"]
        == usage.prompt_token_count
    )
    assert (
        (spans[0].attributes or {})["llm.usage.total_tokens"]
        == usage.total_token_count
    )
    assert (
        (spans[0].attributes or {})["llm.usage.total_tokens"]
        == cast(int, (spans[0].attributes or {})["gen_ai.usage.input_tokens"])
        + cast(int, (spans[0].attributes or {})["gen_ai.usage.output_tokens"])
    )


@pytest.mark.vcr
def test_google_genai_reasoning_tokens_with_include_thoughts(
    span_exporter: InMemorySpanExporter,
):
    client = Client(api_key="123")
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "text": "How many times does the letter 'r' appear in the word strawberry?"
                    },
                ],
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": "Think deep and thoroughly step by step."},
            thinking_config=types.ThinkingConfig(
                thinking_budget=512, include_thoughts=True
            ),
        ),
    )

    usage = response.usage_metadata
    assert usage is not None
    assert response.parts is not None
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    assert (
        (spans[0].attributes or {})["gen_ai.usage.reasoning_tokens"]
        == usage.thoughts_token_count
    )
    assert (
        (spans[0].attributes or {})["gen_ai.usage.output_tokens"]
        == (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    )
    assert (
        (spans[0].attributes or {})["gen_ai.usage.input_tokens"]
        == usage.prompt_token_count
    )
    assert (
        (spans[0].attributes or {})["llm.usage.total_tokens"]
        == usage.total_token_count
    )
    assert (
        (spans[0].attributes or {})["llm.usage.total_tokens"]
        == cast(int, (spans[0].attributes or {})["gen_ai.usage.input_tokens"])
        + cast(int, (spans[0].attributes or {})["gen_ai.usage.output_tokens"])
    )
    assert json.loads(cast(str, (spans[0].attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {"text": response.parts[0].text, "thought": True},
                    {
                        "text": response.text,
                    },
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
    ]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_google_genai_reasoning_tokens_async(span_exporter: InMemorySpanExporter):
    client = Client(api_key="123")
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "text": "How many times does the letter 'r' appear in the word strawberry?"
                    },
                ],
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": "Think deep and thoroughly step by step."},
            thinking_config=types.ThinkingConfig(thinking_budget=512),
        ),
    )

    usage = response.usage_metadata
    assert usage is not None
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    assert (
        (spans[0].attributes or {})["gen_ai.usage.reasoning_tokens"]
        == usage.thoughts_token_count
    )
    assert (
        (spans[0].attributes or {})["gen_ai.usage.output_tokens"]
        == (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    )
    assert (
        (spans[0].attributes or {})["gen_ai.usage.input_tokens"]
        == usage.prompt_token_count
    )
    assert (
        (spans[0].attributes or {})["llm.usage.total_tokens"]
        == usage.total_token_count
    )
    assert (
        (spans[0].attributes or {})["llm.usage.total_tokens"]
        == cast(int, (spans[0].attributes or {})["gen_ai.usage.input_tokens"])
        + cast(int, (spans[0].attributes or {})["gen_ai.usage.output_tokens"])
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_google_genai_reasoning_tokens_with_include_thoughts_async(
    span_exporter: InMemorySpanExporter,
):
    client = Client(api_key="123")
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "text": "How many times does the letter 'r' appear in the word strawberry?"
                    },
                ],
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction={"text": "Think deep and thoroughly step by step."},
            thinking_config=types.ThinkingConfig(
                thinking_budget=512, include_thoughts=True
            ),
        ),
    )

    usage = response.usage_metadata
    assert usage is not None
    assert response.parts is not None
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    assert (
        (spans[0].attributes or {})["gen_ai.usage.reasoning_tokens"]
        == usage.thoughts_token_count
    )
    assert (
        (spans[0].attributes or {})["gen_ai.usage.output_tokens"]
        == (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    )
    assert (
        (spans[0].attributes or {})["gen_ai.usage.input_tokens"]
        == usage.prompt_token_count
    )
    assert (
        (spans[0].attributes or {})["llm.usage.total_tokens"]
        == usage.total_token_count
    )
    assert (
        (spans[0].attributes or {})["llm.usage.total_tokens"]
        == cast(int, (spans[0].attributes or {})["gen_ai.usage.input_tokens"])
        + cast(int, (spans[0].attributes or {})["gen_ai.usage.output_tokens"])
    )
    assert json.loads(cast(str, (spans[0].attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {"text": response.parts[0].text, "thought": True},
                    {
                        "text": response.text,
                    },
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
    ]


@pytest.mark.vcr
def test_google_genai_string_contents(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point. Use tools as much as possible."
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents="What is the capital of France?",
        config=types.GenerateContentConfig(
            system_instruction={"text": system_instruction},
        ),
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    messages = json.loads(cast(str, (spans[0].attributes or {})["gen_ai.input.messages"]))
    assert messages[0] == {"role": "system", "parts": [{"text": system_instruction}]}
    assert messages[1] == {
        "role": "user",
        "parts": [{"text": "What is the capital of France?"}],
    }
    assert json.loads(cast(str, (spans[0].attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": response.text,
                    }
                ],
            },
            "finish_reason": "STOP",
            "index": 0,
        }
    ]


def test_google_genai_error(span_exporter: InMemorySpanExporter):
    # Invalid key on purpose
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point. Use tools as much as possible."
    with pytest.raises(ClientError):
        _res = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": "What is the capital of France?"},
                    ],
                }
            ],
            config=types.GenerateContentConfig(
                system_instruction={"text": system_instruction},
            ),
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content"
    assert (
        (spans[0].attributes or {})["gen_ai.request.model"] == "gemini-2.5-flash-preview-05-20"
    )
    messages = json.loads(cast(str, (spans[0].attributes or {})["gen_ai.input.messages"]))
    assert messages[0] == {"role": "system", "parts": [{"text": system_instruction}]}
    assert messages[1] == {
        "role": "user",
        "parts": [{"text": "What is the capital of France?"}],
    }
    assert (spans[0].attributes or {})["error.type"] == "ClientError"

    assert spans[0].status.status_code == StatusCode.ERROR
    events = spans[0].events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert (event.attributes or {})["exception.type"] == "google.genai.errors.ClientError"
    assert cast(str, (event.attributes or {})["exception.message"]).startswith("400")
    assert (
        "Traceback (most recent call last):" in cast(str, (event.attributes or {})["exception.stacktrace"])
    )
    assert "google.genai.errors.ClientError" in cast(str, (event.attributes or {})["exception.stacktrace"])


@pytest.mark.vcr
def test_google_genai_streaming(span_exporter: InMemorySpanExporter):
    client = Client(api_key="123")

    stream = client.models.generate_content_stream(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "Write a short poem about cats"},
                ],
            }
        ],
    )
    final_response = ""
    chunk_count = 0
    for chunk in stream:
        final_response += chunk.text or ""
        chunk_count += 1

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "gemini.generate_content_stream"
    assert (
        final_response
        == """A silent stalk, a velvet paw,
A hunter's grace, without a flaw.
Emerald eyes, that softly gleam,
Lost in a dream, a furry dream.

A gentle purr, a rumbling sound,
When happy hearts are to be found.
They stretch and yawn, a lazy art,
And steal away a human heart."""
    )

    messages = json.loads(cast(str, (span.attributes or {})["gen_ai.input.messages"]))
    assert messages[0] == {
        "role": "user",
        "parts": [{"text": "Write a short poem about cats"}],
    }
    assert json.loads(cast(str, (spans[0].attributes or {})["gen_ai.output.messages"])) == [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": final_response,
                    }
                ],
            },
        }
    ]
    assert (span.attributes or {})["gen_ai.usage.input_tokens"] == 7
    assert (span.attributes or {})["gen_ai.usage.output_tokens"] == 166
    assert (span.attributes or {})["llm.usage.total_tokens"] == 175  # 173 + 2 (thinking tokens)
    assert len(span.events) == chunk_count
    assert all(event.name == "llm.content.completion.chunk" for event in span.events)


@pytest.mark.vcr
def test_google_genai_no_tokens(span_exporter: InMemorySpanExporter):
    client = Client(api_key="123")

    # The cassette is manually modified to set usage_metadata.total_token_count to None
    # (null). This is possible if the tool call fails.
    def get_weather(location: str) -> str:
        return f"The weather in {location} is sunny."

    stream = client.models.generate_content_stream(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "What is the weather in Paris?"},
                ],
            }
        ],
        config=types.GenerateContentConfig(
            tools=[get_weather],
        ),
    )
    full_response = ""
    for chunk in stream:
        # consume the stream
        full_response += chunk.text or ""

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content_stream"
    assert full_response == "The weather in Paris is sunny."


@pytest.mark.vcr
def test_google_genai_no_content(span_exporter: InMemorySpanExporter):
    client = Client(api_key="123")

    # The cassette is manually modified to set content to None (null) in some
    # parts of the response. This is possible if the model fails to generate
    # content.

    stream = client.models.generate_content_stream(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": "What is the capital of France? Tell me about the city."},
                ],
            }
        ],
    )
    full_response = ""
    expected_parts = [
        "The capital of France is **",
        None,  # "Paris**.\n\nParis is a city that truly needs no introduction. It's",
        " renowned worldwide for its **rich history, iconic landmarks, vibrant culture, and undeniable romantic ambiance**. Here's a glimpse into what makes Paris so special:\n\n**Iconic Landmarks & Attractions:**\n\n*   **Eiffel Tower:** Perhaps",
        " the most recognizable symbol of Paris, this wrought-iron lattice tower offers breathtaking panoramic views of the city.\n*   **Louvre Museum:** Home to an unparalleled collection of art, including Leonardo da Vinci's Mona Lisa, the Venus de Milo,",
        " and thousands of other masterpieces.\n*   **Notre-Dame Cathedral:** A magnificent Gothic cathedral, currently undergoing restoration after a devastating fire, it remains a powerful symbol of French heritage.\n*   **Arc de Triomphe:** Standing",
        " at the western end of the Champs-\xc9lys\xe9es, this monumental arch honors those who fought and died for France.\n*   **Champs-\xc9lys\xe9es:** A grand avenue lined with luxury boutiques, cafes, theaters, and cinemas",
        ", leading from the Place de la Concorde to the Arc de Triomphe.\n*   **Sacr\xe9-C\u0153ur Basilica:** Perched atop Montmartre hill, this stunning white basilica offers spectacular views and a charming atmosphere in",
        None,  # " its surrounding neighborhood.\n*   **Mus\xe9e d'Orsay:** Housed in a former Beaux-Arts railway station, this museum boasts an impressive collection of Impressionist and Post-Impressionist masterpieces.\n*   **S",
        "ainte-Chapelle:** Known for its exquisite stained-glass windows, this Gothic chapel is a marvel of medieval artistry.\n*   **Palace of Versailles:** A short train ride from Paris, this opulent former royal residence is a testament",
        " to the grandeur of French monarchy.\n\n**Culture and Lifestyle:**\n\n*   **Art and Fashion:** Paris is a global epicenter for art and fashion. Its museums, galleries, haute couture houses, and thriving street style scene are legendary.\n*   **Gast",
        "ronomy:** French cuisine is world-famous, and Paris is its ultimate expression. From Michelin-starred restaurants to charming bistros and bustling markets, the city offers an incredible culinary journey. Think croissants, macarons, escargots, and world",
        '-class wines.\n*   **Romance and Ambiance:** Paris is often called the "City of Love" for good reason. Its charming cobblestone streets, beautiful bridges over the Seine River, intimate cafes, and picturesque parks create an',
        " incredibly romantic atmosphere.\n*   **Intellectual and Artistic Hub:** Throughout history, Paris has been a magnet for intellectuals, artists, writers, and philosophers, fostering a vibrant and dynamic cultural scene.\n*   **Caf\xe9 Culture:** Parisians are",
        " known for their love of lingering in cafes, sipping coffee or wine, and watching the world go by. This caf\xe9 culture is an integral part of the city's social fabric.\n*   **Parks and Gardens:** Despite its urban density",
        None,  # ", Paris boasts beautiful green spaces like the Tuileries Garden, Luxembourg Gardens, and Bois de Boulogne, offering oases of tranquility.\n\n**Key Characteristics:**\n\n*   **River Seine:** The Seine River gracefully divides the city, with",
        " iconic bridges and embankments that are central to its charm.\n*   **Distinct Neighborhoods (Arrondissements):** Paris is divided into 20 arrondissements, each with its own unique character and atmosphere, from the bohemian Mont",
        "martre to the chic Saint-Germain-des-Pr\xe9s.\n*   **Lively and Bustling:** While known for its romance, Paris is also a dynamic and bustling metropolis with a constant flow of activity.\n\nIn essence, Paris is a city",
        " that captivates the senses and nourishes the soul. It's a place where history, art, food, and fashion converge to create an unforgettable experience.",
    ]
    for i, chunk in enumerate(stream):
        # consume the stream
        full_response += chunk.text or ""
        assert chunk.text == expected_parts[i]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gemini.generate_content_stream"
    assert full_response == "".join(part for part in expected_parts if part is not None)


# Tests for merge_text_parts function
def test_merge_text_parts_empty_list():
    """Test that empty list returns empty list"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )

    result = merge_text_parts([])
    assert result == []


def test_merge_text_parts_consecutive_strings():
    """Test merging consecutive string inputs"""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
        merge_text_parts,
    )

    parts = ["Hello ", "world", "!"]
    result = merge_text_parts(parts)

    assert len(result) == 1
    assert result[0].text == "Hello world!"


def test_merge_text_parts_consecutive_part_objects():
    """Test merging consecutive Part objects with text"""
    parts = [
        types.Part(text="abc"),
        types.Part(text="def"),
        types.Part(text="ghi"),
    ]
    result = merge_text_parts(parts)

    assert len(result) == 1
    assert result[0].text == "abcdefghi"


def test_merge_text_parts_consecutive_part_dicts():
    """Test merging consecutive PartDict (dict) inputs"""
    parts: list[types.PartDict] = [
        {"text": "First "},
        {"text": "second "},
        {"text": "third"},
    ]
    result = merge_text_parts(parts)

    assert len(result) == 1
    assert result[0].text == "First second third"


def test_merge_text_parts_mixed_types():
    """Test merging with mixed input types (str, Part, dict)"""
    parts: list[str | types.Part | types.PartDict] = [
        "Start ",
        types.Part(text="middle "),
        {"text": "end"},
    ]
    result = merge_text_parts(parts)

    assert len(result) == 1
    assert result[0].text == "Start middle end"


def test_merge_text_parts_with_non_text_part():
    """Test that non-text parts break the merge sequence"""
    # Create an inline_data part (e.g., image)
    inline_data_part = types.Part(
        inline_data=types.Blob(
            mime_type="image/png",
            data = b"fake_image_data"
        )
    )

    parts = [
        types.Part(text="abc"),
        types.Part(text="def"),
        inline_data_part,
        types.Part(text="xyz"),
    ]
    result = merge_text_parts(parts)

    # Should result in 3 parts: merged text "abcdef", inline_data, text "xyz"
    assert len(result) == 3
    assert result[0].text == "abcdef"
    assert result[1].inline_data is not None
    assert result[2].text == "xyz"


def test_merge_text_parts_with_function_call():
    """Test that function call parts break the merge sequence"""
    # Create a function call part
    function_call_part = types.Part(
        function_call=types.FunctionCall(
            name="get_weather",
            args={"location": "Tokyo"},
        )
    )

    parts = [
        types.Part(text="The weather is "),
        function_call_part,
        types.Part(text=" degrees."),
    ]
    result = merge_text_parts(parts)

    # Should result in 3 parts: text, function_call, text
    assert len(result) == 3
    assert result[0].text == "The weather is "
    assert result[1].function_call is not None
    assert result[2].text == " degrees."


def test_merge_text_parts_multiple_non_text_parts():
    """Test multiple non-text parts with text in between"""
    inline_data_part1 = types.Part(
        inline_data=types.Blob(
            mime_type="image/png",
            data=b"image1"
        )
    )
    inline_data_part2 = types.Part(
        inline_data=types.Blob(
            mime_type="image/png",
            data=b"image2",
        )
    )

    parts = [
        types.Part(text="Text1 "),
        types.Part(text="Text2"),
        inline_data_part1,
        types.Part(text="Text3"),
        inline_data_part2,
        types.Part(text="Text4 "),
        types.Part(text="Text5"),
    ]
    result = merge_text_parts(parts)

    # Should result in 5 parts:
    # merged "Text1 Text2", image1, "Text3", image2, merged "Text4 Text5"
    assert len(result) == 5
    assert result[0].text == "Text1 Text2"
    assert result[1].inline_data is not None
    assert result[2].text == "Text3"
    assert result[3].inline_data is not None
    assert result[4].text == "Text4 Text5"


def test_merge_text_parts_only_non_text_parts():
    """Test that only non-text parts are preserved as-is"""
    inline_data_part1 = types.Part(
        inline_data=types.Blob(
            mime_type="image/png",
            data=b"image1"
        )
    )
    inline_data_part2 = types.Part(
        inline_data=types.Blob(
            mime_type="image/png",
            data=b"image2",
        )
    )

    parts = [inline_data_part1, inline_data_part2]
    result = merge_text_parts(parts)

    # Should result in 2 parts unchanged
    assert len(result) == 2
    assert result[0].inline_data is not None
    assert result[1].inline_data is not None


def test_merge_text_parts_single_text_part():
    """Test that a single text part is returned as-is"""
    parts = [types.Part(text="Single text")]
    result = merge_text_parts(parts)

    assert len(result) == 1
    assert result[0].text == "Single text"


def test_merge_text_parts_with_file_object():
    """Test that File objects break the merge sequence"""
    # Create a File object
    file_obj = types.File(name="document.pdf", uri="gs://bucket/document.pdf")

    parts = [
        types.Part(text="Before file "),
        types.Part(text="part"),
        file_obj,
        types.Part(text="After "),
        types.Part(text="file"),
    ]
    result = merge_text_parts(parts)

    # Should result in 3 parts: merged text, file, merged text
    assert len(result) == 3
    assert result[0].text == "Before file part"
    assert isinstance(result[1], types.Part)
    assert result[1].file_data is not None
    assert cast(types.Part, result[1].file_data)  # pyright: ignore[reportInvalidCast]
    assert result[2].text == "After file"


def test_merge_text_parts_trailing_text_only():
    """Test parts ending with text (no non-text parts)"""
    parts = [
        types.Part(text="Part1 "),
        types.Part(text="Part2 "),
        types.Part(text="Part3"),
    ]
    result = merge_text_parts(parts)

    assert len(result) == 1
    assert result[0].text == "Part1 Part2 Part3"


def test_merge_text_parts_leading_non_text():
    """Test parts starting with non-text part"""
    inline_data_part = types.Part(
        inline_data=types.Blob(
            mime_type="image/png",
            data=b"image"
        )
    )

    parts = [
        inline_data_part,
        types.Part(text="Text1 "),
        types.Part(text="Text2"),
    ]
    result = merge_text_parts(parts)

    assert len(result) == 2
    assert result[0].inline_data is not None
    assert result[1].text == "Text1 Text2"

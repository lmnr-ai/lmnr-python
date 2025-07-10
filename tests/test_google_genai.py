import base64
import httpx
import json
import pytest

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from google.genai import Client, types


image_url = "https://upload.wikimedia.org/wikipedia/commons/8/8e/MuseumOfFineArtsBoston_CopleySquare_19thc.jpg"
image_media_type = "image/jpeg"
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
image_data_raw_bytes = base64.b64encode(httpx.get(image_url).content).decode()

get_weather_declaration = {
    "name": "get_weather",
    "description": "Gets the weather in a given city.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location to get the weather for.",
            },
        },
        "required": ["location"],
    },
}


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
        spans[0].attributes["gen_ai.request.model"] == "gemini-2.5-flash-preview-05-20"
    )
    assert (
        spans[0].attributes["gen_ai.response.model"]
        == "models/gemini-2.5-flash-preview-05-20"
    )
    assert spans[0].attributes["gen_ai.prompt.0.content"] == system_instruction
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "system"
    user_content = json.loads(spans[0].attributes["gen_ai.prompt.1.content"])
    assert user_content[0]["type"] == "text"
    assert user_content[0]["text"] == "What is the capital of France?"
    assert spans[0].attributes["gen_ai.prompt.1.role"] == "user"
    assert json.loads(spans[0].attributes["gen_ai.completion.0.content"])[0] == {
        "type": "text",
        "text": response.text,
    }
    assert spans[0].attributes["gen_ai.completion.0.role"] == "model"


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
            span.attributes["gen_ai.request.model"] == "gemini-2.5-flash-preview-05-20"
        )
        assert (
            span.attributes["gen_ai.response.model"]
            == "models/gemini-2.5-flash-preview-05-20"
        )
        assert span.attributes["gen_ai.prompt.0.content"] == system_instruction
        assert span.attributes["gen_ai.prompt.0.role"] == "system"

    spans = sorted(spans, key=lambda x: x.start_time)
    adjective_span = spans[0]
    haiku_span = spans[1]

    adjective_span_user_content = json.loads(
        adjective_span.attributes["gen_ai.prompt.1.content"]
    )
    assert adjective_span_user_content[0]["type"] == "text"
    assert (
        adjective_span_user_content[0]["text"]
        == "Come up with an adjective in English. Respond with only the adjective."
    )
    assert adjective_span.attributes["gen_ai.prompt.1.role"] == "user"
    assert json.loads(adjective_span.attributes["gen_ai.completion.0.content"])[0] == {
        "type": "text",
        "text": adjective_response.text,
    }
    assert adjective_span.attributes["gen_ai.completion.0.role"] == "model"

    haiku_span_user_content = json.loads(
        haiku_span.attributes["gen_ai.prompt.1.content"]
    )
    assert haiku_span_user_content[0]["type"] == "text"
    assert (
        haiku_span_user_content[0]["text"]
        == "Come up with an adjective in English. Respond with only the adjective."
    )
    assert haiku_span.attributes["gen_ai.prompt.1.role"] == "user"
    haiku_span_model_content = json.loads(
        haiku_span.attributes["gen_ai.prompt.2.content"]
    )
    assert haiku_span_model_content[0]["type"] == "text"
    assert haiku_span_model_content[0]["text"] == adjective_response.text
    assert haiku_span.attributes["gen_ai.prompt.2.role"] == "model"
    haiku_span_user_content2 = json.loads(
        haiku_span.attributes["gen_ai.prompt.3.content"]
    )
    assert haiku_span_user_content2[0]["type"] == "text"
    assert (
        haiku_span_user_content2[0]["text"]
        == "Now generate a haiku using this adjective."
    )

    assert haiku_span.attributes["gen_ai.prompt.3.role"] == "user"
    assert json.loads(haiku_span.attributes["gen_ai.completion.0.content"])[0] == {
        "type": "text",
        "text": haiku_response.text,
    }
    assert haiku_span.attributes["gen_ai.completion.0.role"] == "model"


@pytest.mark.vcr
def test_google_genai_tool_calls(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point. Use tools as much as possible."
    client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
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
    assert (
        spans[0].attributes["gen_ai.request.model"] == "gemini-2.5-flash-preview-05-20"
    )
    assert (
        spans[0].attributes["gen_ai.response.model"]
        == "models/gemini-2.5-flash-preview-05-20"
    )
    assert spans[0].attributes["gen_ai.prompt.0.content"] == system_instruction
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "system"
    user_content = json.loads(spans[0].attributes["gen_ai.prompt.1.content"])
    assert user_content[0]["type"] == "text"
    assert user_content[0]["text"] == "What is the weather in Tokyo?"
    assert spans[0].attributes["gen_ai.prompt.1.role"] == "user"
    assert spans[0].attributes["gen_ai.completion.0.tool_calls.0.name"] == "get_weather"
    assert json.loads(
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.arguments"]
    ) == {"location": "Tokyo"}
    assert spans[0].attributes["gen_ai.completion.0.role"] == "model"


@pytest.mark.vcr
def test_google_genai_multiple_tool_calls(span_exporter: InMemorySpanExporter):
    # The actual key was used during recording and the request/response was saved
    # to the VCR cassette.
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point. Use tools as much as possible."
    client.models.generate_content(
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
        spans[0].attributes["gen_ai.request.model"] == "gemini-2.5-flash-preview-05-20"
    )
    assert (
        spans[0].attributes["gen_ai.response.model"]
        == "models/gemini-2.5-flash-preview-05-20"
    )
    assert spans[0].attributes["gen_ai.prompt.0.content"] == system_instruction
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "system"
    user_content = json.loads(spans[0].attributes["gen_ai.prompt.1.content"])
    assert user_content[0]["type"] == "text"
    assert user_content[0]["text"] == "What is the weather in Tokyo and Paris?"
    assert spans[0].attributes["gen_ai.prompt.1.role"] == "user"
    assert spans[0].attributes["gen_ai.completion.0.tool_calls.0.name"] == "get_weather"
    assert json.loads(
        spans[0].attributes["gen_ai.completion.0.tool_calls.0.arguments"]
    ) == {"location": "Tokyo"}
    assert spans[0].attributes["gen_ai.completion.0.tool_calls.1.name"] == "get_weather"
    assert json.loads(
        spans[0].attributes["gen_ai.completion.0.tool_calls.1.arguments"]
    ) == {"location": "Paris"}
    assert spans[0].attributes["gen_ai.completion.0.role"] == "model"


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
        spans[0].attributes["gen_ai.request.model"] == "gemini-2.5-flash-preview-05-20"
    )
    assert (
        spans[0].attributes["gen_ai.response.model"]
        == "models/gemini-2.5-flash-preview-05-20"
    )
    assert spans[0].attributes["gen_ai.prompt.0.content"] == system_instruction
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "system"

    user_content = json.loads(spans[0].attributes["gen_ai.prompt.1.content"])

    assert user_content[0]["type"] == "text"
    assert user_content[0]["text"] == "Describe this image"
    assert user_content[1]["type"] == "image_url"
    assert (
        user_content[1]["image_url"]["url"]
        == f"data:{image_media_type};base64,{image_data}"
    )
    assert spans[0].attributes["gen_ai.prompt.1.role"] == "user"
    assert json.loads(spans[0].attributes["gen_ai.completion.0.content"])[0] == {
        "type": "text",
        "text": response.text,
    }
    assert spans[0].attributes["gen_ai.completion.0.role"] == "model"


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
        spans[0].attributes["gen_ai.request.model"] == "gemini-2.5-flash-preview-05-20"
    )
    assert (
        spans[0].attributes["gen_ai.response.model"]
        == "models/gemini-2.5-flash-preview-05-20"
    )
    assert spans[0].attributes["gen_ai.prompt.0.content"] == system_instruction
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "system"

    user_content = json.loads(spans[0].attributes["gen_ai.prompt.1.content"])

    assert user_content[0]["type"] == "text"
    assert user_content[0]["text"] == "Describe this image"
    assert user_content[1]["type"] == "image_url"
    assert (
        user_content[1]["image_url"]["url"]
        == f"data:{image_media_type};base64,{image_data}"
    )
    assert spans[0].attributes["gen_ai.prompt.1.role"] == "user"
    assert json.loads(spans[0].attributes["gen_ai.completion.0.content"])[0] == {
        "type": "text",
        "text": response.text,
    }
    assert spans[0].attributes["gen_ai.completion.0.role"] == "model"

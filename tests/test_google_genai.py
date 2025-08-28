import base64
import httpx
import json
import pytest
import pydantic

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from google.genai import Client, types
from google.genai.errors import ClientError


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
        spans[0].attributes["gen_ai.request.model"]
        == "gemini-2.5-flash-lite-preview-06-17"
    )
    assert (
        spans[0].attributes["gen_ai.response.model"]
        == "gemini-2.5-flash-lite-preview-06-17"
    )

    user_content = json.loads(spans[0].attributes["gen_ai.prompt.0.content"])
    assert user_content == [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert json.loads(spans[0].attributes["gen_ai.completion.0.content"])[0] == {
        "type": "text",
        "text": response.text,
    }
    assert spans[0].attributes["gen_ai.completion.0.role"] == "model"
    assert (
        json.loads(spans[0].attributes["gen_ai.request.structured_output_schema"])
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
        spans[0].attributes["gen_ai.request.model"]
        == "gemini-2.5-flash-lite-preview-06-17"
    )
    assert (
        spans[0].attributes["gen_ai.response.model"]
        == "gemini-2.5-flash-lite-preview-06-17"
    )

    user_content = json.loads(spans[0].attributes["gen_ai.prompt.0.content"])
    assert user_content == [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    assert spans[0].attributes["gen_ai.prompt.0.role"] == "user"
    assert json.loads(spans[0].attributes["gen_ai.completion.0.content"])[0] == {
        "type": "text",
        "text": response.text,
    }
    assert spans[0].attributes["gen_ai.completion.0.role"] == "model"
    assert (
        json.loads(spans[0].attributes["gen_ai.request.structured_output_schema"])
        == EXPECTED_SCHEMA
    )


def test_google_genai_error(span_exporter: InMemorySpanExporter):
    # Invalid key on purpose
    client = Client(api_key="123")
    system_instruction = "Be concise and to the point. Use tools as much as possible."
    with pytest.raises(ClientError):
        client.models.generate_content(
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
    assert spans[0].attributes["gen_ai.prompt.0.content"] == system_instruction
    assert spans[0].attributes["gen_ai.prompt.0.role"] == "system"
    user_content = json.loads(spans[0].attributes["gen_ai.prompt.1.content"])
    assert user_content[0]["type"] == "text"
    assert user_content[0]["text"] == "What is the capital of France?"
    assert spans[0].attributes["gen_ai.prompt.1.role"] == "user"
    assert spans[0].attributes["error.type"] == "ClientError"

    assert spans[0].status.status_code == StatusCode.ERROR
    events = spans[0].events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "google.genai.errors.ClientError"
    assert event.attributes["exception.message"].startswith("400")
    assert (
        "Traceback (most recent call last):" in event.attributes["exception.stacktrace"]
    )
    assert "google.genai.errors.ClientError" in event.attributes["exception.stacktrace"]


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

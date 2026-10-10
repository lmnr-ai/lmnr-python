import json
from typing import cast

import pydantic
import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import (
    OpenAIInstrumentor,
)


class Joke(pydantic.BaseModel):
    joke: str
    rating: int


@pytest.mark.vcr
def test_chat_response_format(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    response = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=ResponseFormatJSONSchema(
            type="json_schema",
            json_schema=JSONSchema(
                name="Joke",
                description="A joke and your self evaluation of it from 1 to 10",
                schema=Joke.model_json_schema(),
            ),
        ),
    )

    assert "joke" in json.loads(cast(str, response.choices[0].message.content))
    assert "rating" in json.loads(cast(str, response.choices[0].message.content))

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = span.attributes or {}
    attributes = span.attributes or {}
    assert span.name == "openai.chat"
    assert attributes.get("gen_ai.response.id") == "chatcmpl-Botx5sCVD3gs6JSNnNcmlvZdqXNZp"
    assert attributes.get("gen_ai.request.model") == "gpt-4.1-nano"
    assert attributes.get("gen_ai.response.model") == "gpt-4.1-nano-2025-04-14"
    assert json.loads(cast(str, attributes.get("gen_ai.request.structured_output_schema"))) == Joke.model_json_schema()

    input_messages = json.loads(cast(str, attributes.get("gen_ai.input.messages")))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert input_messages[0]["role"] == "user"
    output_messages = json.loads(cast(str, attributes.get("gen_ai.output.messages")))
    assert output_messages[0]["message"]["role"] == "assistant"
    assert output_messages[0]["message"]["content"] == response.choices[0].message.content


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chat_response_format(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        response_format=ResponseFormatJSONSchema(
            type="json_schema",
            json_schema=JSONSchema(
                name="Joke",
                description="A joke and your self evaluation of it from 1 to 10",
                schema=Joke.model_json_schema(),
            ),
        ),
    )

    assert "joke" in json.loads(cast(str, response.choices[0].message.content))
    assert "rating" in json.loads(cast(str, response.choices[0].message.content))

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = span.attributes or {}
    attributes = span.attributes or {}
    assert span.name == "openai.chat"
    assert attributes.get("gen_ai.response.id") == "chatcmpl-BouUvjFs4cpqtzGMx4TQZ92swSqz0"
    assert attributes.get("gen_ai.request.model") == "gpt-4.1-nano"
    assert attributes.get("gen_ai.response.model") == "gpt-4.1-nano-2025-04-14"
    assert json.loads(cast(str, attributes.get("gen_ai.request.structured_output_schema"))) == Joke.model_json_schema()

    input_messages = json.loads(cast(str, attributes.get("gen_ai.input.messages")))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert input_messages[0]["role"] == "user"
    output_messages = json.loads(cast(str, attributes.get("gen_ai.output.messages")))
    assert output_messages[0]["message"]["role"] == "assistant"
    assert output_messages[0]["message"]["content"] == response.choices[0].message.content

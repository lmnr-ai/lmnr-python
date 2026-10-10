import json
from typing import Any, cast

import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import ParsedTextBlock
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel


class Joke(BaseModel):
    setup: str
    punchline: str


@pytest.mark.vcr
def test_anthropic_message_parse_pydantic_model(
    instrumentor: Any,
    anthropic_client: Anthropic,
    span_exporter: InMemorySpanExporter,
):
    response = anthropic_client.messages.parse(
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about opentelemetry",
            }
        ],
        model="claude-sonnet-4-5",
        output_format=Joke,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["anthropic.chat"]
    anthropic_span = spans[0]
    attributes = anthropic_span.attributes or {}

    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"

    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    # Structured output is returned as a text block with JSON.
    text_block = next(
        block for block in output_messages[0]["content"] if block["type"] == "text"
    )
    assert text_block["text"] == cast(ParsedTextBlock, response.content[0]).text

    assert (
        json.loads(
            cast(str, attributes.get("gen_ai.request.structured_output_schema"))
        )
        == Joke.model_json_schema()
    )


@pytest.mark.vcr
def test_anthropic_message_create_with_output_config(
    instrumentor: Any,
    anthropic_client: Anthropic,
    span_exporter: InMemorySpanExporter,
):
    schema = {
        "type": "object",
        "properties": {
            "setup": {"type": "string"},
            "punchline": {"type": "string"},
        },
        "required": ["setup", "punchline"],
        "additionalProperties": False,
    }

    _res = anthropic_client.messages.create(
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about opentelemetry",
            }
        ],
        model="claude-sonnet-4-5",
        output_config=cast(Any, {"format": {"type": "json_schema", "schema": schema}}),
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["anthropic.chat"]
    anthropic_span = spans[0]

    assert (
        json.loads(
            cast(
                str,
                (anthropic_span.attributes or {}).get("gen_ai.request.structured_output_schema")
            )
        )
        == schema
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_anthropic_message_parse_pydantic_model(
    instrumentor: Any,
    async_anthropic_client: AsyncAnthropic,
    span_exporter: InMemorySpanExporter,
):
    response = await async_anthropic_client.messages.parse(
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about opentelemetry",
            }
        ],
        model="claude-sonnet-4-5",
        output_format=Joke,
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["anthropic.chat"]
    anthropic_span = spans[0]

    attributes = anthropic_span.attributes or {}
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    text_block = next(
        block for block in output_messages[0]["content"] if block["type"] == "text"
    )
    assert text_block["text"] == cast(ParsedTextBlock, response.content[0]).text

    assert (
        json.loads(
            cast(str, attributes.get("gen_ai.request.structured_output_schema"))
        )
        == Joke.model_json_schema()
    )

import json
from pathlib import Path
from typing import cast

import anyio
import pytest
from openai import AsyncOpenAI, OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import (
    OpenAIInstrumentor,
)


@pytest.mark.vcr
def test_openai_prompt_caching(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
):
    with open(Path(__file__).parent.parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_openai_prompt_caching <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + f.read()
        )
    client = OpenAI()

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        _ = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "openai.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_creation_attributes = cache_creation_span.attributes or {}
    cache_read_span = spans[1]
    cache_read_attributes = cache_read_span.attributes or {}

    creation_input = json.loads(cast(str, cache_creation_attributes["gen_ai.input.messages"]))
    assert creation_input[0]["role"] == "system"
    assert creation_input[0]["content"] == system_message
    read_input = json.loads(cast(str, cache_read_attributes["gen_ai.input.messages"]))
    assert read_input[0]["role"] == "system"
    assert read_input[0]["content"] == system_message

    assert creation_input[1]["role"] == "user"
    assert creation_input[1]["content"] == text
    assert read_input[1]["role"] == "user"
    assert read_input[1]["content"] == text

    assert (
        cache_creation_attributes.get("gen_ai.response.id")
        == "chatcmpl-BNi3xzj4EEAzo6vce1IwHwie9IRhH"
    )
    assert (
        cache_read_attributes.get("gen_ai.response.id")
        == "chatcmpl-BNi420iFNtIOHzy8Gq2fVS5utTus7"
    )

    creation_output = json.loads(cast(str, cache_creation_attributes["gen_ai.output.messages"]))
    assert creation_output[0]["message"]["role"] == "assistant"
    read_output = json.loads(cast(str, cache_read_attributes["gen_ai.output.messages"]))
    assert read_output[0]["message"]["role"] == "assistant"

    assert cache_creation_attributes["gen_ai.usage.input_tokens"] == 1149
    assert cache_creation_attributes["gen_ai.usage.output_tokens"] == 315
    assert cache_creation_attributes["gen_ai.usage.cache_read_input_tokens"] == 0

    assert cache_read_attributes["gen_ai.usage.input_tokens"] == 1149
    assert cache_read_attributes["gen_ai.usage.output_tokens"] == 353
    assert cache_read_attributes["gen_ai.usage.cache_read_input_tokens"] == 1024


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_prompt_caching_async(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
):
    async with await anyio.open_file(Path(__file__).parent.parent.joinpath("data/1024+tokens.txt"), "r") as f:
        # add the unique test name to the prompt to avoid caching leaking to other tests
        text = (
            "test_openai_prompt_caching_async <- IGNORE THIS. ARTICLES START ON THE NEXT LINE\n"
            + await f.read()
        )
    client = AsyncOpenAI()

    system_message = "You help generate concise summaries of news articles and blog posts that user sends you."

    for _ in range(2):
        _ = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

    spans = span_exporter.get_finished_spans()
    # verify overall shape
    assert all(span.name == "openai.chat" for span in spans)
    assert len(spans) == 2
    cache_creation_span = spans[0]
    cache_creation_attributes = cache_creation_span.attributes or {}
    cache_read_span = spans[1]
    cache_read_attributes = cache_read_span.attributes or {}

    creation_input = json.loads(cast(str, cache_creation_attributes["gen_ai.input.messages"]))
    assert creation_input[0]["role"] == "system"
    assert creation_input[0]["content"] == system_message
    read_input = json.loads(cast(str, cache_read_attributes["gen_ai.input.messages"]))
    assert read_input[0]["role"] == "system"
    assert read_input[0]["content"] == system_message

    assert creation_input[1]["role"] == "user"
    assert creation_input[1]["content"] == text
    assert read_input[1]["role"] == "user"
    assert read_input[1]["content"] == text
    assert (
        cache_creation_attributes.get("gen_ai.response.id")
        == "chatcmpl-BNhr79TlegaJvfSOAOH2jsPEpRHMd"
    )
    assert (
        cache_read_attributes.get("gen_ai.response.id")
        == "chatcmpl-BNhrEFvKSNY08Uphau5iA4InZH6jn"
    )

    creation_output = json.loads(cast(str, cache_creation_attributes["gen_ai.output.messages"]))
    assert creation_output[0]["message"]["role"] == "assistant"
    read_output = json.loads(cast(str, cache_read_attributes["gen_ai.output.messages"]))
    assert read_output[0]["message"]["role"] == "assistant"

    assert cache_creation_attributes["gen_ai.usage.input_tokens"] == 1150
    assert cache_creation_attributes["gen_ai.usage.output_tokens"] == 293
    assert cache_creation_attributes["gen_ai.usage.cache_read_input_tokens"] == 0

    assert cache_read_attributes["gen_ai.usage.input_tokens"] == 1150
    assert cache_read_attributes["gen_ai.usage.output_tokens"] == 307
    assert cache_read_attributes["gen_ai.usage.cache_read_input_tokens"] == 1024

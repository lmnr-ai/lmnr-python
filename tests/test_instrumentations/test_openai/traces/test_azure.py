import json
from typing import Any, cast

import pytest
from openai import AsyncAzureOpenAI, AzureOpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.utils import (
    is_reasoning_supported,
)

PROMPT_FILTER_KEY = "prompt_filter_results"
PROMPT_ERROR = "prompt_error"


@pytest.mark.vcr
def test_chat(
    instrumentor: Any,
    span_exporter: InMemorySpanExporter,
    azure_openai_client: AzureOpenAI,
):
    _res = azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    input_messages = json.loads(cast(str, (open_ai_span.attributes or {})["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    output_messages = json.loads(cast(str, (open_ai_span.attributes or {})["gen_ai.output.messages"]))
    assert output_messages[0]["message"]["content"]
    assert (
        (open_ai_span.attributes or {}).get("gen_ai.request.base_url")
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert (open_ai_span.attributes or {}).get("llm.is_streaming") is False
    assert (
        (open_ai_span.attributes or {}).get("gen_ai.response.id")
        == "chatcmpl-9HpbZPf84KZFiQG6fdY0KVtIwHyIa"
    )


@pytest.mark.vcr
def test_chat_content_filtering(
    instrumentor: Any,
    span_exporter: InMemorySpanExporter,
    azure_openai_client: AzureOpenAI,
):
    _res = azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    input_messages = json.loads(cast(str, (open_ai_span.attributes or {})["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    output_messages = json.loads(cast(str, (open_ai_span.attributes or {})["gen_ai.output.messages"]))
    assert output_messages[0]["finish_reason"] == "content_filter"
    content_filter_results = output_messages[0]["content_filter_results"]
    assert content_filter_results["hate"]["filtered"] is True
    assert content_filter_results["hate"]["severity"] == "high"
    assert content_filter_results["self_harm"]["filtered"] is False
    assert content_filter_results["self_harm"]["severity"] == "safe"
    assert (
        (open_ai_span.attributes or {}).get("gen_ai.request.base_url")
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert (open_ai_span.attributes or {}).get("llm.is_streaming") is False
    assert (
        (open_ai_span.attributes or {}).get("gen_ai.response.id")
        == "chatcmpl-9HpyGSWv1hoKdGaUaiFhfxzTEVlZo"
    )


@pytest.mark.vcr
def test_prompt_content_filtering(
    instrumentor: Any,
    span_exporter: InMemorySpanExporter,
    azure_openai_client: AzureOpenAI,
):
    _res = azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert isinstance((open_ai_span.attributes or {})[f"gen_ai.prompt.{PROMPT_ERROR}"], str)

    error = json.loads(cast(str, (open_ai_span.attributes or {})[f"gen_ai.prompt.{PROMPT_ERROR}"]))

    assert "innererror" in error

    assert "content_filter_result" in error["innererror"]

    assert error["innererror"]["code"] == "ResponsibleAIPolicyViolation"

    assert error["innererror"]["content_filter_result"]["hate"]["filtered"]

    assert error["innererror"]["content_filter_result"]["hate"]["severity"] == "high"

    assert error["innererror"]["content_filter_result"]["sexual"]["filtered"] is False

    assert error["innererror"]["content_filter_result"]["sexual"]["severity"] == "safe"


@pytest.mark.vcr
def test_chat_streaming(
    instrumentor: Any,
    span_exporter: InMemorySpanExporter,
    azure_openai_client: AzureOpenAI,
):
    response = azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    input_messages = json.loads(cast(str, (open_ai_span.attributes or {})["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    output_messages = json.loads(cast(str, (open_ai_span.attributes or {})["gen_ai.output.messages"]))
    assert output_messages[0]["message"]["content"]
    assert (
        (open_ai_span.attributes or {}).get("gen_ai.request.base_url")
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert (open_ai_span.attributes or {}).get("llm.is_streaming") is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    # prompt filter results
    prompt_filter_results = json.loads(
        cast(str, (open_ai_span.attributes or {}).get(f"gen_ai.prompt.{PROMPT_FILTER_KEY}"))
    )
    assert prompt_filter_results[0]["prompt_index"] == 0
    assert (
        prompt_filter_results[0]["content_filter_results"]["hate"]["severity"] == "safe"
    )
    assert (
        prompt_filter_results[0]["content_filter_results"]["self_harm"]["filtered"]
        is False
    )
    assert (
        (open_ai_span.attributes or {}).get("gen_ai.response.id")
        == "chatcmpl-9HpbaAXyt0cAnlWvI8kUAFpZt5jyQ"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_streaming(
    instrumentor: Any,
    span_exporter: InMemorySpanExporter,
    async_azure_openai_client: AsyncAzureOpenAI,
):
    response = await async_azure_openai_client.chat.completions.create(
        model="openllmetry-testing",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    async for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    input_messages = json.loads(cast(str, (open_ai_span.attributes or {})["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    output_messages = json.loads(cast(str, (open_ai_span.attributes or {})["gen_ai.output.messages"]))
    assert output_messages[0]["message"]["content"]
    assert (
        (open_ai_span.attributes or {}).get("gen_ai.request.base_url")
        == "https://traceloop-stg.openai.azure.com/openai/"
    )
    assert (open_ai_span.attributes or {}).get("llm.is_streaming") is True

    events = open_ai_span.events
    assert len(events) == chunk_count
    assert (
        (open_ai_span.attributes or {}).get("gen_ai.response.id")
        == "chatcmpl-9HpbbsSaH8U6amSDAwdA2WzMeDdLB"
    )


@pytest.mark.vcr
@pytest.mark.skipif(
    not is_reasoning_supported(),
    reason="Reasoning is not supported in older OpenAI library versions",
)
def test_chat_reasoning(
    instrumentor: Any,
    span_exporter: InMemorySpanExporter,
    azure_openai_client: AzureOpenAI,
):
    _res = azure_openai_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Count r's in strawberry"}],
        reasoning_effort="low",
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[-1]
    assert (span.attributes or {})["gen_ai.request.reasoning_effort"] == "low"
    assert cast(int, (span.attributes or {})["gen_ai.usage.reasoning_tokens"]) > 0

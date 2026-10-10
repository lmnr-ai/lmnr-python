import json
from typing import Any, cast
from unittest.mock import patch

import httpx
import openai
import pytest
from openai import AsyncOpenAI, AuthenticationError, OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import (
    OpenAIInstrumentor,
)

from .utils import (
    assert_request_contains_tracecontext,
    single_request_to_path,
    spy_decorator,
)


@pytest.mark.vcr
def test_embeddings(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    _ = openai_client.embeddings.create(
        input="Tell me a joke about opentelemetry",
        model="text-embedding-ada-002",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert attributes["gen_ai.request.model"] == "text-embedding-ada-002"
    assert attributes["gen_ai.usage.input_tokens"] == 8
    assert (
        attributes["gen_ai.request.base_url"]
        == "https://api.openai.com/v1/"
    )


@pytest.mark.vcr
def test_embeddings_with_raw_response(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    response = openai_client.embeddings.with_raw_response.create(
        input="Tell me a joke about opentelemetry",
        model="text-embedding-ada-002",
    )
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"

    assert attributes["gen_ai.request.model"] == "text-embedding-ada-002"
    assert attributes["gen_ai.usage.input_tokens"] == 8
    assert (
        attributes["gen_ai.request.base_url"]
        == "https://api.openai.com/v1/"
    )

    parsed_response = response.parse()
    assert parsed_response.data[0]


@pytest.mark.vcr
def test_azure_openai_embeddings(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
):
    api_key = "test-api-key"
    azure_resource = "test-resource"
    azure_deployment = "test-deployment"

    openai_client = openai.AzureOpenAI(
        api_key=api_key,
        azure_endpoint=f"https://{azure_resource}.openai.azure.com",
        azure_deployment=azure_deployment,
        api_version="2023-07-01-preview",
    )
    _ = openai_client.embeddings.create(
        input="Tell me a joke about opentelemetry",
        model="embedding",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert attributes["gen_ai.request.model"] == "embedding"
    assert attributes["gen_ai.usage.input_tokens"] == 8
    assert (
        attributes["gen_ai.request.base_url"]
        == f"https://{azure_resource}.openai.azure.com/openai/deployments/{azure_deployment}/"
    )
    assert attributes["gen_ai.openai.api_version"] == "2023-07-01-preview"


@pytest.mark.vcr
def test_embeddings_context_propagation(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    vllm_openai_client: OpenAI,
):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        _ = vllm_openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="intfloat/e5-mistral-7b-instruct",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    request = single_request_to_path(send_spy.mock, "/v1/embeddings")  # pyright: ignore[reportFunctionMemberAccess]

    assert_request_contains_tracecontext(request, cast(Any, open_ai_span))


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_embeddings_context_propagation(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_vllm_openai_client: AsyncOpenAI,
):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        _ = await async_vllm_openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="intfloat/e5-mistral-7b-instruct",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    request = single_request_to_path(send_spy.mock, "/v1/embeddings")  # pyright: ignore[reportFunctionMemberAccess]

    assert_request_contains_tracecontext(request, cast(Any, open_ai_span))


def test_embeddings_exception(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    openai_client.api_key = "invalid"
    with pytest.raises(AuthenticationError):
        _ = openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="text-embedding-ada-002",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.status.status_code == StatusCode.ERROR
    assert (open_ai_span.status.description or "").startswith("Error code: 401")
    events = open_ai_span.events
    assert len(events) == 1
    event = events[0]
    event_attributes = event.attributes or {}
    assert event.name == "exception"
    assert event_attributes["exception.type"] == "openai.AuthenticationError"
    assert cast(str, event_attributes["exception.message"]).startswith("Error code: 401")


@pytest.mark.asyncio
async def test_async_embeddings_exception(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    async_openai_client.api_key = "invalid"
    with pytest.raises(AuthenticationError):
        _ = await async_openai_client.embeddings.create(
            input="Tell me a joke about opentelemetry",
            model="text-embedding-ada-002",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.status.status_code == StatusCode.ERROR
    assert (open_ai_span.status.description or "").startswith("Error code: 401")
    events = open_ai_span.events
    assert len(events) == 1
    event = events[0]
    event_attributes = event.attributes or {}
    assert event.name == "exception"
    assert event_attributes["exception.type"] == "openai.AuthenticationError"
    assert cast(str, event_attributes["exception.message"]).startswith("Error code: 401")

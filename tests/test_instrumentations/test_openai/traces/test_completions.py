import json
import os
from typing import Any, cast
from unittest.mock import patch

import httpx
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
def test_completion(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    _ = openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert output_messages[0]["text"]
    assert (
        attributes.get("gen_ai.request.base_url")
        == "https://api.openai.com/v1/"
    )
    assert attributes.get("llm.is_streaming") is False
    assert (
        attributes.get("gen_ai.response.id")
        == "cmpl-8wq42D1Socatcl1rCmgYZOFX7dFZw"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    _ = await async_openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert output_messages[0]["text"]
    assert (
        attributes.get("gen_ai.response.id")
        == "cmpl-8wq43c8U5ZZCQBX5lrSpsANwcd3OF"
    )


@pytest.mark.vcr
def test_completion_langchain_style(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    _ = openai_client.completions.create(
        model="davinci-002",
        prompt=["Tell me a joke about opentelemetry"],
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert output_messages[0]["text"]
    assert (
        attributes.get("gen_ai.response.id")
        == "cmpl-8wq43QD6R2WqfxXLpYsRvSAIn9LB9"
    )


@pytest.mark.vcr
def test_completion_streaming(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    # set os env for token usage record in stream mode
    original_value = os.environ.get("TRACELOOP_STREAM_TOKEN_USAGE")
    os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = "true"

    try:
        response = openai_client.completions.create(
            model="davinci-002",
            prompt="Tell me a joke about opentelemetry",
            stream=True,
        )

        for _ in response:
            pass

        spans = span_exporter.get_finished_spans()
        assert [span.name for span in spans] == [
            "openai.completion",
        ]
        open_ai_span = spans[0]
        attributes = open_ai_span.attributes or {}
        input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
        assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
        output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
        assert output_messages[0]["text"]
        assert (
            attributes.get("gen_ai.request.base_url")
            == "https://api.openai.com/v1/"
        )

        assert (
            attributes.get("gen_ai.response.id")
            == "cmpl-8wq44ev1DvyhsBfm1hNwxfv6Dltco"
        )

    finally:
        # unset env
        if original_value is None:
            del os.environ["TRACELOOP_STREAM_TOKEN_USAGE"]
        else:
            os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = original_value


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion_streaming(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    response = await async_openai_client.completions.create(
        model="davinci-002",
        prompt="Tell me a joke about opentelemetry",
        stream=True,
    )

    async for _ in response:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    output_messages = json.loads(cast(str, attributes["gen_ai.output.messages"]))
    assert output_messages[0]["text"]
    assert (
        attributes.get("gen_ai.request.base_url")
        == "https://api.openai.com/v1/"
    )
    assert (
        attributes.get("gen_ai.response.id")
        == "cmpl-8wq44uFYuGm6kNe44ntRwluggKZFY"
    )


@pytest.mark.vcr
def test_completion_context_propagation(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    vllm_openai_client: OpenAI,
):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        _ = vllm_openai_client.completions.create(
            # model="davinci-002",
            model="meta-llama/Llama-3.2-1B-Instruct",
            prompt="Tell me a joke about opentelemetry",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    openai_span = spans[0]
    attributes = openai_span.attributes or {}

    request = single_request_to_path(send_spy.mock, "/v1/completions")  # pyright: ignore[reportFunctionMemberAccess]

    assert_request_contains_tracecontext(request, cast(Any, openai_span))
    assert (
        attributes.get("gen_ai.response.id")
        == "cmpl-2996bf68f7f142fa817bdd32af678df9"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_completion_context_propagation(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_vllm_openai_client: AsyncOpenAI,
):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        _ = await async_vllm_openai_client.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            prompt="Tell me a joke about opentelemetry",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    openai_span = spans[0]
    attributes = openai_span.attributes or {}

    request = single_request_to_path(send_spy.mock, "/v1/completions")  # pyright: ignore[reportFunctionMemberAccess]

    assert_request_contains_tracecontext(request, cast(Any, openai_span))
    assert (
        attributes.get("gen_ai.response.id")
        == "cmpl-4acc6171f6c34008af07ca8490da3b95"
    )


def test_completion_exception(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    openai_client: OpenAI,
):
    openai_client.api_key = "invalid"
    with pytest.raises(AuthenticationError):
        _ = openai_client.completions.create(
            model="gpt-3.5-turbo",
            prompt="Tell me a joke about opentelemetry",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert open_ai_span.status.status_code == StatusCode.ERROR
    assert (open_ai_span.status.description or "").startswith("Error code: 401")
    events = open_ai_span.events
    assert len(events) == 1
    event = events[0]
    event_attributes = event.attributes or {}
    assert event.name == "exception"
    assert event_attributes["exception.type"] == "openai.AuthenticationError"
    assert cast(str, event_attributes["exception.message"]).startswith("Error code: 401")
    assert (
        "Traceback (most recent call last):" in cast(str, event_attributes["exception.stacktrace"])
    )
    assert "openai.AuthenticationError" in cast(str, event_attributes["exception.stacktrace"])
    assert "invalid_api_key" in cast(str, event_attributes["exception.stacktrace"])
    assert attributes.get("error.type") == "AuthenticationError"


@pytest.mark.asyncio
async def test_async_completion_exception(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    async_openai_client: AsyncOpenAI,
):
    async_openai_client.api_key = "invalid"
    with pytest.raises(AuthenticationError):
        _ = await async_openai_client.completions.create(
            model="gpt-3.5-turbo",
            prompt="Tell me a joke about opentelemetry",
        )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.completion",
    ]
    open_ai_span = spans[0]
    attributes = open_ai_span.attributes or {}
    input_messages = json.loads(cast(str, attributes["gen_ai.input.messages"]))
    assert input_messages[0]["content"] == "Tell me a joke about opentelemetry"
    assert open_ai_span.status.status_code == StatusCode.ERROR
    assert (open_ai_span.status.description or "").startswith("Error code: 401")
    events = open_ai_span.events
    assert len(events) == 1
    event = events[0]
    event_attributes = event.attributes or {}
    assert event.name == "exception"
    assert event_attributes["exception.type"] == "openai.AuthenticationError"
    assert cast(str, event_attributes["exception.message"]).startswith("Error code: 401")
    assert (
        "Traceback (most recent call last):" in cast(str, event_attributes["exception.stacktrace"])
    )
    assert "openai.AuthenticationError" in cast(str, event_attributes["exception.stacktrace"])
    assert "invalid_api_key" in cast(str, event_attributes["exception.stacktrace"])
    assert attributes.get("error.type") == "AuthenticationError"

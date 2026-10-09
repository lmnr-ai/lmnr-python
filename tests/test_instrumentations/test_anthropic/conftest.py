"""Unit tests configuration module."""

import os
from collections.abc import Generator

import pytest
from anthropic import Anthropic, AsyncAnthropic
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic import (
    AnthropicInstrumentor,
)


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture
def anthropic_client() -> Anthropic:
    return Anthropic()


@pytest.fixture
def async_anthropic_client() -> AsyncAnthropic:
    return AsyncAnthropic()


@pytest.fixture(scope="function")
def instrumentor(tracer_provider: TracerProvider) -> Generator[AnthropicInstrumentor]:
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(autouse=True)
def environment():
    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = "test_api_key"


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, list[str]]:
    return {"filter_headers": ["x-api-key"]}

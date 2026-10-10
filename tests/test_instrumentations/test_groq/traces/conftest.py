"""Unit tests configuration module."""

import os
from collections.abc import Generator

import pytest
from groq import AsyncGroq, Groq
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.groq import GroqInstrumentor


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter: InMemorySpanExporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture
def groq_client():
    return Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )


@pytest.fixture
def async_groq_client():
    return AsyncGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )


@pytest.fixture(scope="function")
def instrument_legacy(tracer_provider: TracerProvider) -> Generator[GroqInstrumentor]:
    instrumentor = GroqInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(autouse=True)
def environment():
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = "api-key"


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "api-key"]}

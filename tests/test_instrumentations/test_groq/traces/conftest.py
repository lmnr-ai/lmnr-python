"""Unit tests configuration module."""

import os

import pytest
from groq import AsyncGroq, Groq
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.groq import GroqInstrumentor

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
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
def instrument_legacy(tracer_provider):
    instrumentor = GroqInstrumentor(enrich_token_usage=True)
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

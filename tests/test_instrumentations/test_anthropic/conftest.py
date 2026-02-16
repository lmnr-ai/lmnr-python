"""Unit tests configuration module."""

import os

import pytest
from anthropic import Anthropic, AsyncAnthropic
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic import (
    AnthropicInstrumentor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture
def anthropic_client():
    return Anthropic()


@pytest.fixture
def async_anthropic_client():
    return AsyncAnthropic()


@pytest.fixture(scope="function")
def instrument_legacy(tracer_provider):
    async def upload_base64_image(*args):
        return "/some/url"

    instrumentor = AnthropicInstrumentor(
        enrich_token_usage=True,
        upload_base64_image=upload_base64_image,
    )
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
def vcr_config():
    return {"filter_headers": ["x-api-key"]}

"""Unit tests configuration module."""

import os

import pytest
from anthropic import Anthropic, AsyncAnthropic
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic import (
    AnthropicInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic.utils import (
    LMNR_TRACE_CONTENT,
)
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="event_logger_provider")
def fixture_event_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    event_logger_provider = EventLoggerProvider(provider)

    return event_logger_provider


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


@pytest.fixture(scope="function")
def instrument_with_content(tracer_provider, event_logger_provider):
    os.environ.update({LMNR_TRACE_CONTENT: "True"})

    async def upload_base64_image(*args):
        return "/some/url"

    instrumentor = AnthropicInstrumentor(
        use_legacy_attributes=False,
        enrich_token_usage=True,
        upload_base64_image=upload_base64_image,
    )
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        event_logger_provider=event_logger_provider,
    )

    yield instrumentor

    os.environ.pop(LMNR_TRACE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_no_content(tracer_provider, event_logger_provider):
    os.environ.update({LMNR_TRACE_CONTENT: "False"})

    async def upload_base64_image(*args):
        return "/some/url"

    instrumentor = AnthropicInstrumentor(
        use_legacy_attributes=False,
        enrich_token_usage=True,
        upload_base64_image=upload_base64_image,
    )
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        event_logger_provider=event_logger_provider,
    )

    yield instrumentor

    os.environ.pop(LMNR_TRACE_CONTENT, None)
    instrumentor.uninstrument()


@pytest.fixture(autouse=True)
def environment():
    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = "test_api_key"


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["x-api-key"]}

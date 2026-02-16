"""Unit tests configuration module."""

import os

import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from opentelemetry._events import get_event_logger
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import (
    OpenAIInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.shared.config import (
    Config,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.utils import (
    LMNR_TRACE_CONTENT,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.version import (
    __version__,
)
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    InMemoryMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


@pytest.fixture(autouse=True)
def environment():
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "test_api_key"
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        os.environ["AZURE_OPENAI_API_KEY"] = "test_azure_api_key"
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://traceloop-stg.openai.azure.com/"


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.fixture
def vllm_openai_client():
    return OpenAI(base_url="http://localhost:8000/v1")


@pytest.fixture
def azure_openai_client():
    return AzureOpenAI(
        api_version="2024-02-01",
    )


@pytest.fixture
def async_azure_openai_client():
    return AsyncAzureOpenAI(
        api_version="2024-02-01",
    )


@pytest.fixture
def async_openai_client():
    return AsyncOpenAI()


@pytest.fixture
def async_vllm_openai_client():
    return AsyncOpenAI(base_url="http://localhost:8000/v1")


@pytest.fixture(scope="session", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function")
def instrument_legacy(tracer_provider):
    async def upload_base64_image(*args):
        return "/some/url"

    instrumentor = OpenAIInstrumentor(
        enrich_assistant=True,
        enrich_token_usage=True,
        upload_base64_image=upload_base64_image,
    )
    was_already_instrumented = instrumentor.is_instrumented_by_opentelemetry
    if not was_already_instrumented:
        instrumentor.instrument(
            tracer_provider=tracer_provider,
        )

    yield instrumentor

    # Only uninstrument if we instrumented it ourselves
    if not was_already_instrumented and instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "api-key"]}

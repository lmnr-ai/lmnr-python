from typing import Generator
import pytest
from unittest.mock import patch
from lmnr import Laminar
from lmnr.opentelemetry_lib import TracerManager
from lmnr.opentelemetry_lib.tracing import TracerWrapper
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry import context as context_api
from opentelemetry.context import Context

from lmnr.opentelemetry_lib.litellm import LaminarLiteLLMCallback

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def span_exporter() -> SpanExporter:
    exporter = InMemorySpanExporter()

    # Set up a partial mock of TracerManager.init to inject our exporter
    orig_tracermanager_init = TracerManager.init

    def mock_tracermanager_init(*args, **kwargs):
        new_kwargs = kwargs.copy()
        new_kwargs["exporter"] = exporter
        orig_tracermanager_init(*args, **new_kwargs)

    with patch(
        "lmnr.opentelemetry_lib.TracerManager.init",
        side_effect=mock_tracermanager_init,
    ):
        Laminar.initialize(
            project_api_key="test_key",
            disable_batch=True,
        )

    return exporter


@pytest.fixture(scope="function")
def litellm_callback() -> Generator[LaminarLiteLLMCallback, None, None]:
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import (
        OpenAIInstrumentor,
    )

    # Check if OpenAI was instrumented before we create the LiteLLM callback
    instrumentor = OpenAIInstrumentor()
    was_instrumented = instrumentor.is_instrumented_by_opentelemetry

    # Create the callback (this will uninstrument OpenAI if it was instrumented)
    callback = LaminarLiteLLMCallback()

    yield callback

    # Re-instrument OpenAI if it was originally instrumented
    if was_instrumented and not instrumentor.is_instrumented_by_opentelemetry:
        # Re-instrument with the same settings as the global initialization
        from lmnr.opentelemetry_lib.tracing import TracerWrapper

        if hasattr(TracerWrapper, "instance") and TracerWrapper.instance is not None:
            instrumentor.instrument(
                tracer_provider=TracerWrapper.instance._tracer_provider
            )


@pytest.fixture(scope="function", autouse=True)
def clear_span_exporter(span_exporter: InMemorySpanExporter):
    # Clear before test
    span_exporter.clear()
    TracerWrapper.clear()

    # Clear OpenTelemetry context to ensure clean state between tests
    # This prevents spans from one test from becoming parents of spans in another test
    # Create a fresh empty context and attach it
    fresh_context = Context()
    token = context_api.attach(fresh_context)

    yield

    # Clear after test as well for good measure
    span_exporter.clear()
    TracerWrapper.clear()

    # Restore and create fresh context again
    try:
        context_api.detach(token)
    except Exception:
        pass
    fresh_context = Context()
    context_api.attach(fresh_context)


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            "authorization",
            "api-key",
            "x-goog-api-key",
            "x-api-key",
        ]
    }

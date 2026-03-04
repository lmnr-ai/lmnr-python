"""Conftest for OpenAI Agents SDK integration tests."""

import os

import pytest
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents import (
    OpenAIAgentsInstrumentor,
)


@pytest.fixture(autouse=True)
def environment():
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "test_api_key"


@pytest.fixture(scope="function")
def instrument_openai_agents(span_exporter):
    instrumentor = OpenAIAgentsInstrumentor()
    was_already_instrumented = instrumentor.is_instrumented_by_opentelemetry
    if not was_already_instrumented:
        instrumentor.instrument()

    yield instrumentor

    if not was_already_instrumented and instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "api-key", "openai-organization"]}

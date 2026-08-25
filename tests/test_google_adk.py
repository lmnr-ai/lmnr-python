"""Tests for the Google ADK instrumentation.

The end-to-end tests drive a real `InMemoryRunner` against gemini-3.5-flash-lite.
The actual key was used during recording and the requests/responses were
saved to the VCR cassettes, so the asserted spans are the ones ADK produces
for real model turns.
"""

import asyncio
import json
import os

import pytest

pytest.importorskip("google.adk")

from google.adk.agents.llm_agent import Agent  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402


def get_weather(city: str) -> dict:
    """Returns the weather for a city."""
    return {"city": city, "forecast": "sunny"}


def get_time(city: str) -> dict:
    """Returns the local time for a city."""
    return {"city": city, "time": "12:00"}


def get_default_city() -> dict:
    """Returns the default city."""
    return {"city": "Almaty"}


@pytest.fixture(scope="module")
def vcr_config():
    # Same filters as the shared config in conftest, plus decompression:
    # ADK talks to Gemini through aiohttp, and vcrpy records that path with
    # a decoded body while keeping the Content-Encoding: gzip header, which
    # breaks replay through httpx. Decoding at record time stores plain
    # bodies with matching headers.
    return {
        "filter_headers": [
            "authorization",
            "api-key",
            "x-goog-api-key",
            "x-api-key",
        ],
        "filter_query_parameters": ["key"],
        "decode_compressed_response": True,
    }


@pytest.fixture(autouse=True)
def gemini_env(monkeypatch):
    # Real key from the environment while recording; the placeholder is
    # enough for replay because vcr_config filters the key out of matches.
    monkeypatch.setenv(
        "GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", "123")
    )


def run_agent(
    user_id: str = "test-user",
    instruction: str = (
        "Call get_weather for the city the user asks about, then answer "
        "in one short sentence."
    ),
    prompt: str = "Weather in Almaty?",
):
    agent = Agent(
        name="weather_agent",
        model="gemini-3.5-flash-lite",
        instruction=instruction,
        tools=[get_weather, get_time, get_default_city],
        generate_content_config=types.GenerateContentConfig(temperature=0),
    )
    runner = InMemoryRunner(agent=agent, app_name="test-app")
    session = asyncio.run(
        runner.session_service.create_session(
            app_name="test-app", user_id=user_id
        )
    )
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    for _ in runner.run(
        user_id=session.user_id, session_id=session.id, new_message=message
    ):
        pass
    return session


def spans_by_name(span_exporter, name):
    return [s for s in span_exporter.get_finished_spans() if s.name == name]


@pytest.mark.vcr
def test_tool_span_typed_with_input_and_output(span_exporter):
    run_agent()

    (tool_span,) = spans_by_name(span_exporter, "execute_tool get_weather")
    assert tool_span.attributes["lmnr.span.type"] == "TOOL"
    assert json.loads(tool_span.attributes["lmnr.span.input"]) == {
        "city": "Almaty"
    }
    assert json.loads(tool_span.attributes["lmnr.span.output"]) == {
        "city": "Almaty",
        "forecast": "sunny",
    }


@pytest.mark.vcr
def test_llm_call_spanned_once(span_exporter):
    # The point of the integration: with Laminar's genai instrumentation
    # active, ADK skips its own model span, so each model turn produces
    # exactly one gemini.generate_content span (two turns here: the tool
    # call and the final answer).
    run_agent()

    llm_spans = spans_by_name(span_exporter, "gemini.generate_content")
    assert len(llm_spans) == 2
    for span in llm_spans:
        assert (
            span.attributes["gen_ai.request.model"] == "gemini-3.5-flash-lite"
        )
    # ADK's native span is named "generate_content <model>"; if the
    # detection patch stops working it comes back alongside the Laminar
    # one, so its absence is the actual regression check.
    native = [
        s
        for s in span_exporter.get_finished_spans()
        if s.name.startswith("generate_content ")
    ]
    assert native == []


@pytest.mark.vcr
def test_agent_span_carries_session_association(span_exporter):
    session = run_agent(user_id="user-42")

    (agent_span,) = spans_by_name(span_exporter, "invoke_agent weather_agent")
    attributes = agent_span.attributes
    assert (
        attributes["lmnr.association.properties.session_id"] == session.id
    )
    assert attributes["lmnr.association.properties.user_id"] == "user-42"


@pytest.mark.vcr
def test_tool_content_respects_adk_content_toggle(span_exporter, monkeypatch):
    # With the content knob off, ADK stamps "{}" for tool args/response;
    # that must not leak into the Laminar input/output attributes.
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "0")
    run_agent()

    (tool_span,) = spans_by_name(span_exporter, "execute_tool get_weather")
    assert tool_span.attributes["lmnr.span.type"] == "TOOL"
    assert "lmnr.span.input" not in tool_span.attributes
    assert "lmnr.span.output" not in tool_span.attributes


def test_adk_recognizes_laminar_genai_instrumentation(span_exporter):
    # ADK's detector only knows the otel-contrib package by filename; the
    # instrumentor teaches it about Laminar's wrapper, active in this
    # session.
    from google.adk.telemetry import tracing

    detected = (
        tracing._instrumented_with_opentelemetry_instrumentation_google_genai()
    )
    assert detected is True


@pytest.mark.vcr
def test_merged_parallel_tool_span_typed(span_exporter):
    # Two function calls in one turn produce an `execute_tool (merged)`
    # span via trace_merged_tool_calls, which flows.llm_flows.functions
    # binds by name at import time. This module imports ADK at collection,
    # before Laminar.initialize() runs, so this fails unless the
    # instrumentor patches the flow module's binding too.
    run_agent(
        instruction=(
            "Call get_weather and get_time together in the same turn, "
            "then answer in one short sentence."
        ),
        prompt="Weather and local time in Almaty?",
    )

    (merged_span,) = spans_by_name(span_exporter, "execute_tool (merged)")
    assert merged_span.attributes["lmnr.span.type"] == "TOOL"
    assert "lmnr.span.output" in merged_span.attributes


@pytest.mark.vcr
def test_empty_args_tool_is_not_mistaken_for_redaction(span_exporter):
    # A niladic tool's args serialize to "{}", same as ADK's redaction
    # sentinel; the (empty) input must still be recorded.
    run_agent(
        instruction=(
            "Call get_default_city, then answer with just the city name."
        ),
        prompt="Which city am I in?",
    )

    (tool_span,) = spans_by_name(
        span_exporter, "execute_tool get_default_city"
    )
    assert tool_span.attributes["lmnr.span.type"] == "TOOL"
    assert tool_span.attributes["lmnr.span.input"] == "{}"
    assert json.loads(tool_span.attributes["lmnr.span.output"]) == {
        "city": "Almaty"
    }


def test_wrap_tracking_survives_reconstruction(span_exporter):
    # BaseInstrumentor is a singleton whose __init__ reruns on every
    # construction; the wrap bookkeeping must not live there, or a second
    # GoogleAdkInstrumentor() would clear it and uninstrument would leave
    # the wraps behind.
    from google.adk.telemetry import tracing
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation import (
        google_adk,
    )

    instrumentor = google_adk.GoogleAdkInstrumentor()
    assert instrumentor.is_instrumented_by_opentelemetry
    assert hasattr(tracing.trace_tool_call, "__wrapped__")
    assert instrumentor._wrapped_functions

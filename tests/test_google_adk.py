"""Tests for the Google ADK instrumentation.

The end-to-end tests drive a real `InMemoryRunner` against
gemini-3.5-flash-lite. The actual key was used during recording and the
requests/responses were saved to the VCR cassettes, so the asserted spans
are the ones ADK produces for real model turns.
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


@pytest.fixture(scope="module", autouse=True)
def adk_instrumentation(span_exporter):
    # conftest.py's session fixture blocks GOOGLE_ADK: with `google-adk` a
    # pinned dev dependency, leaving it enabled would auto-remove
    # GOOGLE_GENAI from the session-wide default set (see
    # _GOOGLE_ADK_GENAI_CONFLICTS in tracing/instruments.py) and break every
    # other module's raw google_genai tests. This module instruments ADK and
    # uninstruments google_genai for its own tests, mirroring what a real
    # ADK-only application gets by default in production.
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_adk import (
        GoogleAdkInstrumentor,
    )
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai import (
        GoogleGenAiSdkInstrumentor,
    )

    adk_instrumentor = GoogleAdkInstrumentor()
    genai_instrumentor = GoogleGenAiSdkInstrumentor()
    adk_was_instrumented = adk_instrumentor.is_instrumented_by_opentelemetry
    genai_was_instrumented = genai_instrumentor.is_instrumented_by_opentelemetry

    if genai_was_instrumented:
        genai_instrumentor.uninstrument()
    if not adk_was_instrumented:
        adk_instrumentor.instrument()

    yield

    if not adk_was_instrumented:
        adk_instrumentor.uninstrument()
    if genai_was_instrumented:
        genai_instrumentor.instrument()


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
def test_call_llm_span_carries_gen_ai_attributes(span_exporter):
    # With GOOGLE_GENAI excluded by default (google-adk installed), ADK's own
    # `call_llm` span is the sole LLM span per turn (two turns here: the
    # tool call and the final answer), enriched directly from the real
    # LlmRequest/LlmResponse objects instead of relying on a separate
    # google_genai span or ADK's raw gcp.vertex.agent.llm_request/response
    # JSON blobs.
    run_agent()

    call_llm_spans = spans_by_name(span_exporter, "call_llm")
    assert len(call_llm_spans) == 2
    for span in call_llm_spans:
        assert (
            span.attributes["gen_ai.request.model"] == "gemini-3.5-flash-lite"
        )
        assert span.attributes["gen_ai.response.model"]
        assert json.loads(span.attributes["gen_ai.input.messages"])
        assert json.loads(span.attributes["gen_ai.output.messages"])
    assert json.loads(call_llm_spans[0].attributes["gen_ai.tool.definitions"])

    # Neither a separate Laminar google_genai span nor ADK's own native
    # "generate_content <model>" span should exist.
    assert spans_by_name(span_exporter, "gemini.generate_content") == []
    native = [
        s
        for s in span_exporter.get_finished_spans()
        if s.name.startswith("generate_content ")
    ]
    assert native == []


@pytest.mark.vcr
def test_tool_span_is_sibling_of_call_llm_not_child(span_exporter):
    # Regression: ADK's own `call_llm` span stays the ambient "current span"
    # (via its still-open start_as_current_span block) through the tool
    # postprocessing that immediately follows, unless the call_llm wrap
    # detaches it first. Confirm the tool span shares call_llm's parent
    # instead of nesting under call_llm itself.
    run_agent()

    call_llm_spans = spans_by_name(span_exporter, "call_llm")
    (tool_span,) = spans_by_name(span_exporter, "execute_tool get_weather")

    call_llm_span_ids = {s.context.span_id for s in call_llm_spans}
    assert tool_span.parent is not None
    assert tool_span.parent.span_id not in call_llm_span_ids
    assert tool_span.parent.span_id in {
        s.parent.span_id for s in call_llm_spans if s.parent is not None
    }


@pytest.mark.vcr
def test_call_llm_span_ends_before_tool_execution(span_exporter):
    # Regression: call_llm's recorded duration must not stretch across the
    # tool-execution/callback postprocessing that follows inside ADK's
    # still-open start_as_current_span block — it should end as soon as the
    # model's last token arrives, not whenever ADK's own `with` block
    # eventually unwinds.
    run_agent()

    call_llm_spans = spans_by_name(span_exporter, "call_llm")
    (tool_span,) = spans_by_name(span_exporter, "execute_tool get_weather")

    # Both call_llm spans (tool-call turn, final-answer turn) share the same
    # invoke_agent parent, so pick the one that produced the function call by
    # timing: it's the turn that ran first.
    call_llm_span = min(call_llm_spans, key=lambda s: s.start_time)
    assert call_llm_span.end_time is not None
    assert call_llm_span.end_time <= tool_span.start_time


@pytest.mark.vcr
def test_call_llm_content_respects_adk_content_toggle(
    span_exporter, monkeypatch
):
    # With the content knob off, ADK stamps "{}" for the legacy
    # gcp.vertex.agent.llm_request/response attributes; the new gen_ai.*
    # attributes must not leak content through a side door.
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "0")
    run_agent()

    for span in spans_by_name(span_exporter, "call_llm"):
        assert "gen_ai.input.messages" not in span.attributes
        assert "gen_ai.tool.definitions" not in span.attributes
        assert "gen_ai.output.messages" not in span.attributes


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
    # ADK's own native inner LLM span is always redundant while this
    # instrumentor is active — call_llm enrichment (or, in the explicit
    # opt-in case, Laminar's own google_genai span) already covers the
    # call — so the detection patch unconditionally reports an external
    # genai instrumentation, regardless of GOOGLE_GENAI's own state.
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


def test_agent_enrichment_keeps_explicit_session_id():
    # An id already on the span was set explicitly through Laminar (the
    # processor stamps context values at span start); the derived ADK id
    # must not overwrite it.
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation import (
        google_adk,
    )

    class FakeSession:
        id = "adk-session-uuid"
        user_id = "adk-user"

    class FakeCtx:
        session = FakeSession()

    class FakeSpan:
        def __init__(self, attributes):
            self.attributes = dict(attributes)

        def is_recording(self):
            return True

        def set_attribute(self, key, value):
            self.attributes[key] = value

    span = FakeSpan({"lmnr.association.properties.session_id": "checkout-42"})
    google_adk._wrap_trace_agent_invocation(
        lambda *a, **k: None, None, (span, object(), FakeCtx()), {}
    )
    assert (
        span.attributes["lmnr.association.properties.session_id"]
        == "checkout-42"
    )
    assert (
        span.attributes["lmnr.association.properties.user_id"] == "adk-user"
    )




def test_uninstrument_unwraps_lazily_imported_binding():
    # initialize() before any ADK import means the flow module is first
    # imported by the instrumentor itself, mid wrap-loop; its by-name
    # binding of trace_merged_tool_calls must come out single-wrapped, and
    # uninstrument must leave no live layer on any binding.
    import sys

    import wrapt
    from google.adk.flows import llm_flows
    from google.adk.telemetry import tracing
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation import (
        google_adk,
    )

    module_name = "google.adk.flows.llm_flows.functions"
    original_module = sys.modules[module_name]
    instrumentor = google_adk.GoogleAdkInstrumentor()
    instrumentor.uninstrument()
    sys.modules.pop(module_name)
    try:
        instrumentor.instrument()
        fresh = sys.modules[module_name]
        assert isinstance(fresh.trace_merged_tool_calls, wrapt.ObjectProxy)
        assert not isinstance(
            fresh.trace_merged_tool_calls.__wrapped__, wrapt.ObjectProxy
        )
        instrumentor.uninstrument()
        assert not isinstance(
            fresh.trace_merged_tool_calls, wrapt.ObjectProxy
        )
        assert not isinstance(
            tracing.trace_merged_tool_calls, wrapt.ObjectProxy
        )
    finally:
        sys.modules[module_name] = original_module
        llm_flows.functions = original_module
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()


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

"""Tests for the Google ADK instrumentation.

ADK emits its spans itself through the global tracer provider, so the
instrumentor's job is enrichment: type the tool spans, copy their
content-gated input/output, map the ADK session/user onto Laminar
association properties, and make ADK's external-genai detection recognize
Laminar's google-genai instrumentation (so LLM calls aren't double-spanned).

The end-to-end tests drive a real `InMemoryRunner` with a stub model that
requests one tool call and then answers, so the spans under test are the ones
ADK actually produces, not hand-built ones.
"""

import asyncio
import json

import pytest

pytest.importorskip("google.adk")

from google.adk.agents.llm_agent import Agent  # noqa: E402
from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
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


class StubLlm(BaseLlm):
    """Requests the configured tool calls, then answers once results are back.

    Stateless on purpose: the branch is decided from the request contents, so
    the same instance can be reused across runs and tests.
    """

    model: str = "stub-model"
    calls: list[tuple[str, dict]] = [("get_weather", {"city": "Almaty"})]

    async def generate_content_async(self, llm_request, stream=False):
        has_tool_result = any(
            part.function_response is not None
            for content in llm_request.contents or []
            for part in content.parts or []
        )
        if has_tool_result:
            parts = [types.Part(text="It is sunny in Almaty.")]
        else:
            parts = [
                types.Part(
                    function_call=types.FunctionCall(name=name, args=args)
                )
                for name, args in self.calls
            ]
        yield LlmResponse(content=types.Content(role="model", parts=parts))


def run_stub_agent(user_id: str = "test-user", calls=None):
    model = StubLlm(calls=calls) if calls else StubLlm()
    agent = Agent(
        name="weather_agent",
        model=model,
        instruction="Answer with the tools.",
        tools=[get_weather, get_time, get_default_city],
    )
    runner = InMemoryRunner(agent=agent, app_name="test-app")
    session = asyncio.run(
        runner.session_service.create_session(
            app_name="test-app", user_id=user_id
        )
    )
    message = types.Content(
        role="user", parts=[types.Part(text="Weather in Almaty?")]
    )
    for _ in runner.run(
        user_id=session.user_id, session_id=session.id, new_message=message
    ):
        pass
    return session


def spans_by_name(span_exporter, name):
    return [s for s in span_exporter.get_finished_spans() if s.name == name]


def test_tool_span_typed_with_input_and_output(span_exporter):
    run_stub_agent()

    (tool_span,) = spans_by_name(span_exporter, "execute_tool get_weather")
    assert tool_span.attributes["lmnr.span.type"] == "TOOL"
    assert json.loads(tool_span.attributes["lmnr.span.input"]) == {
        "city": "Almaty"
    }
    assert json.loads(tool_span.attributes["lmnr.span.output"]) == {
        "city": "Almaty",
        "forecast": "sunny",
    }


def test_agent_span_carries_session_association(span_exporter):
    session = run_stub_agent(user_id="user-42")

    (agent_span,) = spans_by_name(span_exporter, "invoke_agent weather_agent")
    attributes = agent_span.attributes
    assert (
        attributes["lmnr.association.properties.session_id"] == session.id
    )
    assert attributes["lmnr.association.properties.user_id"] == "user-42"


def test_tool_content_respects_adk_content_toggle(span_exporter, monkeypatch):
    # ADK's legacy content knob defaults on; turning it off makes ADK stamp
    # "{}" for tool args/response, and the instrumentor must not copy that
    # onto the Laminar input/output attributes.
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "0")
    run_stub_agent()

    (tool_span,) = spans_by_name(span_exporter, "execute_tool get_weather")
    assert tool_span.attributes["lmnr.span.type"] == "TOOL"
    assert "lmnr.span.input" not in tool_span.attributes
    assert "lmnr.span.output" not in tool_span.attributes


def test_adk_recognizes_laminar_genai_instrumentation(span_exporter):
    # ADK skips its own `generate_content` span when it detects an external
    # google-genai instrumentation. Its detector only knows the otel-contrib
    # package; the instrumentor extends it to report Laminar's, which is
    # active in this test session.
    from google.adk.telemetry import tracing

    detected = (
        tracing._instrumented_with_opentelemetry_instrumentation_google_genai()
    )
    assert detected is True


def test_merged_parallel_tool_span_typed(span_exporter):
    # Two function calls in one model turn make ADK create an
    # `execute_tool (merged)` span via `trace_merged_tool_calls`. That
    # function is bound by name in `flows.llm_flows.functions` at import
    # time — and this module imports ADK at collection, before
    # `Laminar.initialize()` runs — so this test fails unless the
    # instrumentor patches the flow module's binding, not just the
    # `telemetry.tracing` attribute.
    run_stub_agent(
        calls=[
            ("get_weather", {"city": "Almaty"}),
            ("get_time", {"city": "Almaty"}),
        ]
    )

    (merged_span,) = spans_by_name(span_exporter, "execute_tool (merged)")
    assert merged_span.attributes["lmnr.span.type"] == "TOOL"
    assert "lmnr.span.output" in merged_span.attributes


def test_empty_args_tool_is_not_mistaken_for_redaction(span_exporter):
    # ADK stamps "{}" both for redacted content and for a genuinely empty
    # args dict; with the content toggle on, a niladic tool must still get
    # its (empty) input recorded.
    run_stub_agent(calls=[("get_default_city", {})])

    (tool_span,) = spans_by_name(
        span_exporter, "execute_tool get_default_city"
    )
    assert tool_span.attributes["lmnr.span.type"] == "TOOL"
    assert tool_span.attributes["lmnr.span.input"] == "{}"
    assert json.loads(tool_span.attributes["lmnr.span.output"]) == {
        "city": "Almaty"
    }

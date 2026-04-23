"""Integration tests for OpenAI Agents SDK instrumentation.

These tests verify the full pipeline: Agent SDK -> Trace Processor -> Laminar spans.
They use VCR to record/replay actual OpenAI API calls and verify the resulting
spans via InMemorySpanExporter.
"""

import json
import time

import pytest
from agents import Agent, Runner, function_tool

from lmnr.opentelemetry_lib.tracing.attributes import Attributes


def _get_spans(span_exporter):
    """Helper to get spans with a small delay for flush."""
    time.sleep(0.1)
    return span_exporter.get_finished_spans()


def _attrs(spans):
    """Build a list of {name, attributes} dicts from spans for easy inspection."""
    return [{"name": s.name, "attributes": dict(s.attributes or {})} for s in spans]


def _find_spans_with_model(span_data):
    """Find spans that have the model attribute set (response/generation spans)."""
    return [s for s in span_data if s["attributes"].get(Attributes.REQUEST_MODEL.value)]


def _find_spans_by_attr(span_data, key, value=None):
    """Find spans that have a specific attribute, optionally matching a value."""
    if value is None:
        return [s for s in span_data if key in s["attributes"]]
    return [s for s in span_data if s["attributes"].get(key) == value]


@pytest.mark.vcr
def test_simple_agent(instrument_openai_agents, span_exporter):
    """Run a simple agent and verify span structure and attributes."""
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model="gpt-4o-mini",
    )

    result = Runner.run_sync(agent, "What is 2+2?")
    assert "4" in result.final_output

    spans = _get_spans(span_exporter)
    assert len(spans) > 0

    span_data = _attrs(spans)

    # Should have a response span with LLM attributes
    response_spans = _find_spans_with_model(span_data)
    assert len(response_spans) >= 1

    resp = response_spans[0]
    assert resp["attributes"][Attributes.REQUEST_MODEL.value] == "gpt-4o-mini"
    assert resp["attributes"][Attributes.PROVIDER.value] == "openai"

    # Verify token usage was recorded
    assert resp["attributes"][Attributes.INPUT_TOKEN_COUNT.value] == 25
    assert resp["attributes"][Attributes.OUTPUT_TOKEN_COUNT.value] == 8
    assert resp["attributes"][Attributes.TOTAL_TOKEN_COUNT.value] == 33

    # Verify gen_ai.output.messages contains the response text
    output_messages = json.loads(resp["attributes"]["gen_ai.output.messages"])
    assert any("4" in str(m) for m in output_messages["output"])

    # Verify gen_ai.input.messages contains the user message
    input_messages = json.loads(resp["attributes"]["gen_ai.input.messages"])
    assert any("2+2" in str(m) or "2 + 2" in str(m) for m in input_messages)

    # The system instructions from the agent must be prepended as the first
    # message so the full prompt is visible in the trace.
    assert input_messages[0] == {
        "role": "system",
        "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
    }

    # Verify response ID was recorded
    assert resp["attributes"].get(Attributes.RESPONSE_ID.value)


@pytest.mark.vcr
def test_agent_with_tool(instrument_openai_agents, span_exporter):
    """Run an agent that calls a tool and verify tool + LLM spans."""

    @function_tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return "72°F and sunny"

    agent = Agent(
        name="WeatherBot",
        instructions="You are a helpful assistant. Use the get_weather tool to answer weather questions.",
        model="gpt-4.1-nano",
        tools=[get_weather],
    )

    result = Runner.run_sync(agent, "What is the weather in NYC?")
    assert "72" in result.final_output

    spans = _get_spans(span_exporter)
    assert len(spans) > 0

    span_data = _attrs(spans)

    # Should have response spans with model info
    model_spans = _find_spans_with_model(span_data)
    assert len(model_spans) >= 1

    # Verify at least one span recorded the tool call output containing the answer
    all_output = []
    for s in span_data:
        if "gen_ai.output.messages" in s["attributes"]:
            msgs = json.loads(s["attributes"]["gen_ai.output.messages"])
            all_output.extend(msgs["output"])
    assert any("72" in str(m) for m in all_output)

    # Should have a function/tool span for get_weather
    tool_spans = _find_spans_by_attr(span_data, "lmnr.span.type", "TOOL")
    assert len(tool_spans) >= 1

    # Verify token usage on at least one model span
    assert any(
        s["attributes"].get(Attributes.INPUT_TOKEN_COUNT.value) is not None
        for s in model_spans
    )

    # Every span that records input messages should carry the agent's system
    # instructions as the first entry.
    expected_system = {
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "You are a helpful assistant. Use the get_weather tool "
                    "to answer weather questions."
                ),
            }
        ],
    }
    input_spans = [s for s in span_data if "gen_ai.input.messages" in s["attributes"]]
    assert input_spans, "expected at least one span with gen_ai.input.messages"
    for s in input_spans:
        msgs = json.loads(s["attributes"]["gen_ai.input.messages"])
        assert msgs[0] == expected_system

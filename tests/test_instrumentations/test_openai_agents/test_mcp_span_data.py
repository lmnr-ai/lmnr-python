"""Attribute mapping for the Agents SDK's MCP list-tools spans."""

import json

from agents.tracing.span_data import MCPListToolsSpanData

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.span_data import (
    apply_span_data,
)


class _RecordingSpan:
    """Minimal stand-in for LaminarSpan that just collects attributes."""

    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def test_mcp_list_tools_span_gets_io():
    span = _RecordingSpan()
    tools = ["chart.search_patients", "cases.get_case"]

    apply_span_data(span, MCPListToolsSpanData(server="chi_bench", result=tools))

    # The custom attributes are what the SDK hands us...
    assert span.attributes["openai.agents.mcp.server"] == "chi_bench"
    assert json.loads(span.attributes["openai.agents.mcp.result"]) == tools
    # ...but only span I/O is rendered in the trace view.
    assert json.loads(span.attributes["lmnr.span.input"]) == {"server": "chi_bench"}
    assert json.loads(span.attributes["lmnr.span.output"]) == tools


def test_mcp_list_tools_span_without_result_sets_no_output():
    span = _RecordingSpan()

    apply_span_data(span, MCPListToolsSpanData(server="chi_bench"))

    assert json.loads(span.attributes["lmnr.span.input"]) == {"server": "chi_bench"}
    assert "lmnr.span.output" not in span.attributes

"""Unit tests for the OpenAI Agents span-data handlers.

These are hermetic: `apply_span_data` only ever calls `set_attribute` on the
span it is handed, so a recording double is enough and no Agents SDK / OTEL
provider is needed.
"""

import json

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.span_data import (
    apply_span_data,
)


class _RecordingSpan:
    def __init__(self):
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class _McpListToolsSpanData:
    """Mirrors `agents.tracing.span_data.MCPListToolsSpanData`."""

    type = "mcp_tools"

    def __init__(self, server: str | None, result: list[str] | None):
        self.server = server
        self.result = result

    def export(self) -> dict[str, object]:
        return {"type": self.type, "server": self.server, "result": self.result}


def test_mcp_span_sets_io_so_the_trace_view_renders_it():
    span = _RecordingSpan()
    tools = ["chart.search_patients", "cases.get_case"]

    apply_span_data(span, _McpListToolsSpanData("chi_bench", tools))

    assert span.attributes["openai.agents.mcp.server"] == "chi_bench"
    assert json.loads(span.attributes["openai.agents.mcp.result"]) == tools
    # The I/O attributes are what the trace view reads; without them the span
    # renders empty even though the two attributes above are populated.
    assert json.loads(span.attributes["lmnr.span.input"]) == {"server": "chi_bench"}
    assert json.loads(span.attributes["lmnr.span.output"]) == tools


def test_mcp_span_without_a_server_still_records_the_tool_list():
    span = _RecordingSpan()

    apply_span_data(span, _McpListToolsSpanData(None, []))

    assert "openai.agents.mcp.server" not in span.attributes
    assert "lmnr.span.input" not in span.attributes
    assert json.loads(span.attributes["lmnr.span.output"]) == []


def test_mcp_span_with_no_result_records_no_output():
    span = _RecordingSpan()

    apply_span_data(span, _McpListToolsSpanData("chi_bench", None))

    assert span.attributes["openai.agents.mcp.server"] == "chi_bench"
    assert "openai.agents.mcp.result" not in span.attributes
    assert "lmnr.span.output" not in span.attributes
    assert json.loads(span.attributes["lmnr.span.input"]) == {"server": "chi_bench"}

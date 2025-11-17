import pytest

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from claude_agent_sdk import ClaudeAgentOptions, query as claude_query

from mock_transport import MockClaudeTransport


@pytest.mark.asyncio
async def test_claude_agent_query(span_exporter: InMemorySpanExporter):
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are an expert software engineer.",
        permission_mode="acceptEdits",
    )

    async for message in claude_query(
        prompt="What is the capital of France?",
        options=options,
        transport=MockClaudeTransport(
            auto_respond_on_connect=True, close_after_responses=True
        ),
    ):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "query"
    assert spans[0].attributes["lmnr.span.path"] == ("query",)
    assert "Paris" in spans[0].attributes["lmnr.span.output"]

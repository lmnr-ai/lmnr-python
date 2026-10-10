from typing import cast

import claude_agent_sdk
import pytest
from claude_agent_sdk import ClaudeAgentOptions
from mock_transport import (  # pyright: ignore[reportImplicitRelativeImport]
    MockClaudeTransport,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.asyncio
async def test_claude_agent_query(span_exporter: InMemorySpanExporter):
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are an expert software engineer.",
        permission_mode="acceptEdits",
    )

    async for _message in claude_agent_sdk.query(
        prompt="What is the capital of France?",
        options=options,
        transport=MockClaudeTransport(close_after_responses=True),
    ):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "query"
    assert (spans[0].attributes or {})["lmnr.span.path"] == ("query",)
    assert "Paris" in cast(str, (spans[0].attributes or {})["lmnr.span.output"])

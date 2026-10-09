from typing import cast

import pytest
from claude_agent_sdk import ClaudeSDKClient
from mock_transport import (  # pyright: ignore[reportImplicitRelativeImport]
    MockClaudeTransport,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.asyncio
async def test_claude_agent_query(span_exporter: InMemorySpanExporter):
    async with ClaudeSDKClient(transport=MockClaudeTransport()) as client:
        await client.query("What's the capital of France?")
        async for _ in client.receive_response():
            pass

        await client.query("What's the population of that city?")
        async for _ in client.receive_response():
            pass

    spans_tuple = span_exporter.get_finished_spans()
    spans = sorted(spans_tuple, key=lambda x: x.start_time or 0)

    assert len(spans) == 8
    assert spans[0].name == "ClaudeSDKClient.connect"
    assert spans[1].name == "ClaudeSDKClient.query"
    assert spans[2].name == "ClaudeSDKClient.receive_response"
    assert spans[3].name == "ClaudeSDKClient.receive_messages"
    assert spans[4].name == "ClaudeSDKClient.query"
    assert spans[5].name == "ClaudeSDKClient.receive_response"
    assert spans[6].name == "ClaudeSDKClient.receive_messages"
    assert spans[7].name == "ClaudeSDKClient.disconnect"

    assert (spans[1].attributes or {})["lmnr.span.path"] == ("ClaudeSDKClient.query",)
    assert (
        (spans[1].attributes or {})["lmnr.span.input"]
        == '{"prompt":"What\'s the capital of France?"}'
    )
    assert spans[3].parent is not None
    assert spans[2].context is not None
    assert spans[3].parent.trace_id == spans[2].context.trace_id
    assert spans[3].parent.span_id == spans[2].context.span_id
    assert "million" in cast(str, (spans[6].attributes or {})["lmnr.span.output"])

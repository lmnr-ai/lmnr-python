import asyncio, pytest
from dotenv import load_dotenv

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from claude_agent_sdk import ClaudeAgentOptions
import claude_agent_sdk

# Note: can not use alias "query" aka "from claude_agent_sdk import query" because it is not wrapped by Laminar

# TODO: remove this once cassetes are recorded
load_dotenv()

@pytest.mark.vcr(record_mode='once')
def test_claude_agent_query(span_exporter: InMemorySpanExporter):
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are an expert software engineer.",
        permission_mode='acceptEdits',
    )

    async def _collect_messages():
        messages = []
        async for message in claude_agent_sdk.query(
            # prompt="Create a readme doc for the test_claude_agent package. Then delete it. Return task status.",
            prompt="What is the capital of France?",
            options=options
        ):
            messages.append(message)
        return messages

    for _ in asyncio.run(_collect_messages()):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "query"
    assert spans[0].attributes["lmnr.span.path"] == ("query",)
    assert "Paris" in spans[0].attributes["lmnr.span.output"]

import asyncio, os
from dotenv import load_dotenv

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from claude_agent_sdk import query, ClaudeAgentOptions

# TODO: ok to use .env for tests?
load_dotenv()

# TODO: mock claude sdk call to not spend tokens for each test call
def test_claude_agent_query(span_exporter: InMemorySpanExporter):
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are an expert software engineer developer",
        permission_mode='acceptEdits',
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    async def _collect_messages():
        messages = []
        async for message in query(
            # prompt="Create a readme doc for the test_claude_agent package. Then delete it. Return task status.",
            prompt="What is Laminar?",
            options=options
        ):
            messages.append(message)
        return messages

    for _ in asyncio.run(_collect_messages()):
        pass

    spans = span_exporter.get_finished_spans()
    # TODO: fix, not working as expected
    assert len(spans) == 1
    # assert spans[0].name == "claude_agent.query"
    # assert spans[0].attributes["claude_agent.request.model"] == "claude-sonnet-4-5"
    # assert spans[0].attributes["lmnr.span.path"] == ("claude_agent.query",)

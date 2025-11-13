import asyncio, pytest
from dotenv import load_dotenv

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from claude_agent_sdk import query, ClaudeAgentOptions

# TODO: remove this once cassetes are recorded
load_dotenv()

@pytest.mark.vcr(record_mode='once')
def test_claude_agent_query(span_exporter: InMemorySpanExporter):
    # TODO: fix error saying "clear_thinking_20251015 strategy requires `thinking` to be enabled"
    # setting "max_thinking_tokens" to 1024 worked for claudeSDKClient test but not for this test
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are an expert software engineer.",
        permission_mode='acceptEdits',
        # TODO: remove temporary debug print
        stderr=lambda line: print(line),
        extra_args={"debug-to-stderr": None},
        max_thinking_tokens=1024,
    )

    async def _collect_messages():
        messages = []
        async for message in query(
            # prompt="Create a readme doc for the test_claude_agent package. Then delete it. Return task status.",
            prompt="What is the capital of France?",
            options=options
        ):
            messages.append(message)
        return messages

    for _ in asyncio.run(_collect_messages()):
        pass

    spans = span_exporter.get_finished_spans()
    # TODO: fix, not working as expected
    assert len(spans) == 1
    assert spans[0].name == "claude_agent.query"
    assert spans[0].attributes["claude_agent.request.model"] == "claude-sonnet-4-5"
    assert spans[0].attributes["lmnr.span.path"] == ("claude_agent.query",)
    assert "Paris" in spans[0].attributes["lmnr.span.output"]

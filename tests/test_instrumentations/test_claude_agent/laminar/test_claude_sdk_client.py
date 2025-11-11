import asyncio
from dotenv import load_dotenv

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from claude_agent_sdk import ClaudeSDKClient, AssistantMessage, TextBlock

# TODO: ok to use .env for tests?
load_dotenv()

# TODO: mock claude sdk call to not spend tokens for each test call
def test_claude_agent_query(span_exporter: InMemorySpanExporter):
    async def _collect_messages():
        async with ClaudeSDKClient() as client:
            await client.query("What's the capital of France?")
            async for _ in client.receive_response():
                pass

            await client.query("What's the population of that city?")
            async for _ in client.receive_response():
                pass

    asyncio.run(_collect_messages())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].name == "ClaudeSDKClient.query"
    assert spans[0].attributes["lmnr.span.path"] == ("ClaudeSDKClient.query",)
    assert spans[0].attributes["lmnr.span.input"] == '{"prompt":"What\'s the capital of France?"}'
    # TODO: validate spans have same trace id [?]

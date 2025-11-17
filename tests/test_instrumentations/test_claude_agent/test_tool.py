import pytest
from typing import Any
from claude_agent_sdk import (
    tool,
    create_sdk_mcp_server,
    ClaudeAgentOptions,
    ClaudeSDKClient,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mock_transport import MockClaudeTransport


@tool("calculate", "Perform mathematical calculations", {"expression": str})
async def calculate(args: dict[str, Any]) -> dict[str, Any]:
    try:
        result = eval(args["expression"], {"__builtins__": {}})
        return {"content": [{"type": "text", "text": f"Result: {result}"}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error: {str(e)}"}],
            "is_error": True,
        }


@tool("get_time", "Get current time", {})
async def get_time(args: dict[str, Any]) -> dict[str, Any]:
    from datetime import datetime

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"content": [{"type": "text", "text": f"Current time: {current_time}"}]}


@pytest.mark.asyncio
async def test_claude_agent_tool(span_exporter: InMemorySpanExporter):
    my_server = create_sdk_mcp_server(
        name="utilities", version="1.0.0", tools=[calculate, get_time]
    )

    options = ClaudeAgentOptions(
        mcp_servers={"utils": my_server},
        allowed_tools=["mcp__utils__calculate", "mcp__utils__get_time"],
    )

    async with ClaudeSDKClient(
        options=options, transport=MockClaudeTransport()
    ) as client:
        await client.query("What's 123 * 456?")
        async for _ in client.receive_response():
            pass

        await client.query("What time is it now?")
        async for _ in client.receive_response():
            pass

    spans_tuple = span_exporter.get_finished_spans()
    spans = sorted(list(spans_tuple), key=lambda x: x.start_time)

    assert len(spans) == 9
    assert spans[0].name == "create_sdk_mcp_server"
    assert spans[1].name == "ClaudeSDKClient.connect"
    assert spans[2].name == "ClaudeSDKClient.query"
    assert spans[3].name == "ClaudeSDKClient.receive_response"
    assert spans[4].name == "ClaudeSDKClient.receive_messages"
    assert spans[5].name == "ClaudeSDKClient.query"
    assert spans[6].name == "ClaudeSDKClient.receive_response"
    assert spans[7].name == "ClaudeSDKClient.receive_messages"
    assert spans[8].name == "ClaudeSDKClient.disconnect"

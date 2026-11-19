"""`wrap_mcp_server` on mcp 1.x (`FastMCP`, low-level `Server`) and FastMCP 3.

Client and server talk over mcp's in-memory transport. The session-wide
`McpInstrumentor` is on, so every test also checks that a wrapped server's
request gets exactly one SERVER span. The FastMCP 3 tests run only with
`fastmcp` 3.x installed:
`uv run --with mcp==1.30.0 --with fastmcp==3.4.9 pytest <this file>`.
"""

# The dev dependency pins mcp 2.x, which basedpyright checks this module
# against; it only runs under CI's mcp 1.x overlay (see conftest.py).
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUntypedFunctionDecorator=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false

from __future__ import annotations

import json
from collections.abc import Iterator
from importlib.metadata import version
from typing import Any

import pytest
from mcp import types
from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel.server import Server
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from pydantic import AnyUrl

from lmnr import Laminar, wrap_mcp_server
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.observed import (
    OBSERVED_SERVER_ATTR,
)
from lmnr.sdk.mcp_server.observe import CONTEXT_DESCRIPTION
from lmnr.sdk.mcp_server.turns import turn_trace_id
from lmnr.sdk.mcp_server.v1 import ObservedHandlers

MARKER = "content-marker-51d0"
AI_INVOCATION = "io.modelcontextprotocol/aiInvocation"
CLIENT_INFO = types.Implementation(name="cursor-vscode", version="1.4.0")


def _fastmcp() -> FastMCP:
    server = FastMCP("docs")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        with Laminar.start_as_current_span("inside_tool"):
            return a + b

    @server.tool()
    def fail(reason: str) -> str:
        """Always fails."""
        raise ValueError(f"boom: {reason}")

    @server.tool()
    def note(context: str, text: str) -> str:
        """A tool with its own `context` argument."""
        return f"{context}|{text}"

    @server.prompt()
    def summarize(topic: str) -> str:
        """A prompt."""
        return f"Summarize {topic}"

    @server.resource("docs://{slug}")
    def doc(slug: str) -> str:
        """A resource."""
        return f"doc {slug}"

    return server


def _wrapped_lowlevel(received: list[dict[str, Any]]) -> Server:
    """A low-level server wrapped BEFORE its handlers are registered."""
    server: Server = wrap_mcp_server(Server("strict"))

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            types.Tool(name="explode", inputSchema={"type": "object"}),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        received.append(dict(arguments))
        if name == "explode":
            raise McpError(types.ErrorData(code=-32603, message="kaboom " + MARKER))
        return [types.TextContent(type="text", text=f"found {arguments['query']}")]

    return server


@pytest.fixture
def no_client_spans() -> Iterator[None]:
    """Client calls without the instrumentor's CLIENT span and `traceparent`."""
    token = context_api.attach(
        context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    yield
    context_api.detach(token)


def _server_spans(spans: tuple[ReadableSpan, ...]) -> list[ReadableSpan]:
    return [s for s in spans if s.kind == SpanKind.SERVER]


def _one_server_span(spans: tuple[ReadableSpan, ...], name: str) -> ReadableSpan:
    matching = [s for s in _server_spans(spans) if s.name == name]
    assert len(matching) == 1, [(s.name, s.kind) for s in spans]
    return matching[0]


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _trace_id(span: ReadableSpan) -> int:
    assert span.context is not None
    return span.context.trace_id


def _exception_events(span: ReadableSpan) -> list[dict[str, Any]]:
    return [dict(e.attributes or {}) for e in span.events if e.name == "exception"]


@pytest.mark.asyncio
async def test_tool_call_span(span_exporter: InMemorySpanExporter):
    server = wrap_mcp_server(_fastmcp())
    async with create_connected_server_and_client_session(
        server._mcp_server, client_info=CLIENT_INFO
    ) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        result = await client.call_tool("add", {"a": 2, "b": 3, "context": "sum two numbers"})
    assert result.isError is False
    assert result.content[0].text == "5"
    assert tools["add"].inputSchema["properties"]["context"] == {
        "type": "string",
        "description": CONTEXT_DESCRIPTION,
    }
    assert "context" not in tools["add"].inputSchema.get("required", [])
    # the tool's own `context` is left alone
    assert tools["note"].inputSchema == _fastmcp()._tool_manager.get_tool("note").parameters

    spans = span_exporter.get_finished_spans()
    # the instrumentor stands aside for a wrapped server: one SERVER span
    assert [s.name for s in _server_spans(spans)] == ["tools/call add"]
    span = _one_server_span(spans, "tools/call add")
    assert span.parent is None
    attrs = _attrs(span)
    assert attrs["mcp.method.name"] == "tools/call"
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["gen_ai.tool.name"] == "add"
    assert attrs["lmnr.span.type"] == "TOOL"
    assert attrs["jsonrpc.request.id"]
    assert attrs["mcp.protocol.version"]
    assert attrs["lmnr.mcp.client.name"] == "cursor-vscode"
    assert attrs["lmnr.mcp.client.version"] == "1.4.0"
    assert attrs["lmnr.mcp.context"] == "sum two numbers"
    assert json.loads(attrs["gen_ai.tool.call.arguments"]) == {
        "a": 2,
        "b": 3,
        "context": "sum two numbers",
    }
    assert json.loads(attrs["gen_ai.tool.call.result"])["content"][0]["text"] == "5"
    assert "lmnr.span.input" not in attrs

    inner = next(s for s in spans if s.name == "inside_tool")
    assert span.context is not None and inner.parent is not None
    assert inner.parent.span_id == span.context.span_id


@pytest.mark.asyncio
async def test_own_context_untouched(span_exporter: InMemorySpanExporter):
    server = wrap_mcp_server(_fastmcp())
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        result = await client.call_tool("note", {"context": "mine", "text": "t"})
    assert result.content[0].text == "mine|t"
    attrs = _attrs(_one_server_span(span_exporter.get_finished_spans(), "tools/call note"))
    assert "lmnr.mcp.context" not in attrs


@pytest.mark.asyncio
async def test_lowlevel_strict_schema(span_exporter: InMemorySpanExporter):
    received: list[dict[str, Any]] = []
    server = _wrapped_lowlevel(received)
    assert isinstance(server.request_handlers, ObservedHandlers)
    async with create_connected_server_and_client_session(server) as client:
        listed = (await client.list_tools()).tools[0].inputSchema
        # validated against the original schema (`additionalProperties: false`)
        ok = await client.call_tool("search", {"query": "q", "context": "goal"})
        bad = await client.call_tool("search", {"context": "goal"})
        # 1.x's `call_tool` decorator folds every handler error into `isError`
        exploded = await client.call_tool("explode", {})
    assert "context" in listed["properties"]
    assert ok.isError is False
    assert bad.isError is True
    assert exploded.isError is True
    assert received == [{"query": "q"}, {}]

    spans = span_exporter.get_finished_spans()
    search = [s for s in _server_spans(spans) if s.name == "tools/call search"]
    assert len(search) == 2
    assert _attrs(search[0])["lmnr.mcp.context"] == "goal"
    assert "error.type" not in _attrs(search[0])
    # "Input validation error: ..." folded into an `isError` result
    assert _attrs(search[1])["error.type"] == "-32602"
    assert "rpc.response.status_code" not in _attrs(search[1])

    explode = _one_server_span(spans, "tools/call explode")
    assert _attrs(explode)["error.type"] == "tool_error"
    assert explode.status.status_code == StatusCode.ERROR
    assert MARKER in _exception_events(explode)[0]["exception.message"]


@pytest.mark.asyncio
async def test_errors(span_exporter: InMemorySpanExporter):
    server = wrap_mcp_server(_fastmcp())
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        fail = await client.call_tool("fail", {"reason": "r"})
        unknown = await client.call_tool("nope", {})
        invalid = await client.call_tool("add", {"a": "x"})
    assert fail.isError and unknown.isError and invalid.isError

    spans = span_exporter.get_finished_spans()
    assert _attrs(_one_server_span(spans, "tools/call fail"))["error.type"] == "tool_error"
    for name in ("tools/call nope", "tools/call add"):
        assert _attrs(_one_server_span(spans, name))["error.type"] == "-32602"


@pytest.mark.asyncio
async def test_raising_handler_class_name(span_exporter: InMemorySpanExporter):
    server: Server = wrap_mcp_server(Server("raw"))

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        raise KeyError(name)

    async with create_connected_server_and_client_session(server) as client:
        with pytest.raises(McpError):
            _ = await client.get_prompt("missing")
        # no resources handler: "Method not found", still traced
        with pytest.raises(McpError):
            _ = await client.read_resource(AnyUrl("docs://x"))

    spans = span_exporter.get_finished_spans()
    prompt = _one_server_span(spans, "prompts/get missing")
    assert _attrs(prompt)["error.type"] == "KeyError"
    assert "rpc.response.status_code" not in _attrs(prompt)
    resource = _one_server_span(spans, "resources/read")
    assert _attrs(resource)["error.type"] == "-32601"
    assert _attrs(resource)["rpc.response.status_code"] == "-32601"


@pytest.mark.asyncio
async def test_prompt_and_resource(span_exporter: InMemorySpanExporter):
    server = wrap_mcp_server(_fastmcp())
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        _ = await client.get_prompt("summarize", {"topic": "otel"})
        _ = await client.read_resource(AnyUrl("docs://intro"))

    spans = span_exporter.get_finished_spans()
    prompt = _attrs(_one_server_span(spans, "prompts/get summarize"))
    assert prompt["gen_ai.prompt.name"] == "summarize"
    assert json.loads(prompt["lmnr.span.input"]) == {
        "name": "summarize",
        "arguments": {"topic": "otel"},
    }
    assert "Summarize otel" in prompt["lmnr.span.output"]
    resource = _attrs(_one_server_span(spans, "resources/read"))
    assert resource["mcp.resource.uri"] == "docs://intro"


@pytest.mark.asyncio
async def test_record_payloads_off(span_exporter: InMemorySpanExporter):
    server = wrap_mcp_server(_fastmcp(), record_payloads=False)
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        listed = {t.name: t for t in (await client.list_tools()).tools}
        _ = await client.call_tool("add", {"a": 1, "b": 2})
        _ = await client.call_tool("fail", {"reason": MARKER})
    assert "context" not in listed["add"].inputSchema["properties"]
    spans = _server_spans(span_exporter.get_finished_spans())
    assert len(spans) == 2
    for span in spans:
        attrs = _attrs(span)
        assert "gen_ai.tool.call.arguments" not in attrs
        assert "gen_ai.tool.call.result" not in attrs
        assert MARKER not in json.dumps(attrs) + json.dumps(_exception_events(span))


@pytest.mark.asyncio
async def test_turns(span_exporter: InMemorySpanExporter, no_client_spans: None):
    server = wrap_mcp_server(_fastmcp())
    now = [0.0]
    getattr(server, OBSERVED_SERVER_ATTR).turns._clock = lambda: now[0]
    session = {"openai/session": "chat-9"}
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        _ = await client.call_tool("add", {"a": 1, "b": 1}, meta=session)
        now[0] += 10
        _ = await client.call_tool("add", {"a": 1, "b": 2}, meta=session)
        now[0] += 30
        _ = await client.call_tool("add", {"a": 1, "b": 3}, meta=session)
        _ = await client.call_tool(
            "add", {"a": 1, "b": 4}, meta={AI_INVOCATION: {"turnId": "turn-7"}}
        )

    ids = [_trace_id(s) for s in _server_spans(span_exporter.get_finished_spans())]
    assert ids[0] == ids[1] != ids[2]
    assert ids[3] == turn_trace_id(None, "turn-7")


@pytest.mark.asyncio
async def test_client_traceparent_is_linked(span_exporter: InMemorySpanExporter):
    server = wrap_mcp_server(_fastmcp())
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        _ = await client.call_tool("add", {"a": 1, "b": 2})
    spans = span_exporter.get_finished_spans()
    server_span = _one_server_span(spans, "tools/call add")
    client_span = next(s for s in spans if s.kind == SpanKind.CLIENT and s.name == "add.tool")
    assert server_span.parent is None
    assert _trace_id(server_span) == _trace_id(client_span)
    assert client_span.context is not None
    assert server_span.links[0].context.span_id == client_span.context.span_id


def test_wrapping_twice_is_a_no_op():
    server = _fastmcp()
    wrap_mcp_server(server)
    handlers = server._mcp_server.request_handlers
    assert wrap_mcp_server(server) is server
    assert server._mcp_server.request_handlers is handlers
    assert type(dict.__getitem__(handlers, types.CallToolRequest)).__name__ == "function"


# --- FastMCP 3 (standalone `fastmcp`, on mcp 1.x) -----------------------------------


def _fastmcp3() -> Any:
    try:
        if not version("fastmcp").startswith("3."):
            pytest.skip("needs fastmcp 3.x")
        from fastmcp import FastMCP as FastMCP3
    except Exception:
        pytest.skip("needs fastmcp 3.x")

    server = FastMCP3("fm3")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        with Laminar.start_as_current_span("inside_tool"):
            return a + b

    return server


@pytest.mark.asyncio
async def test_fastmcp3(span_exporter: InMemorySpanExporter):
    server = wrap_mcp_server(_fastmcp3())
    from fastmcp import Client as FastMCP3Client

    async with FastMCP3Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
        result = await client.call_tool("add", {"a": 2, "b": 3, "context": "sum"})
        unknown = await client.call_tool("nope", {}, raise_on_error=False)
    assert "context" in tools["add"].inputSchema["properties"]
    assert result.data == 5
    assert unknown.is_error

    spans = span_exporter.get_finished_spans()
    assert not [
        s
        for s in spans
        if s.kind == SpanKind.SERVER
        and s.instrumentation_scope is not None
        and s.instrumentation_scope.name == "fastmcp"
    ]
    assert sorted(s.name for s in _server_spans(spans)) == ["tools/call add", "tools/call nope"]
    span = _one_server_span(spans, "tools/call add")
    assert _attrs(span)["lmnr.mcp.context"] == "sum"
    inner = next(s for s in spans if s.name == "inside_tool")
    assert span.context is not None and inner.parent is not None
    assert inner.parent.span_id == span.context.span_id
    assert _attrs(_one_server_span(spans, "tools/call nope"))["error.type"] == "-32602"

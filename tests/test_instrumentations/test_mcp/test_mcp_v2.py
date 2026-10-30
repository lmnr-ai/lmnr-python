"""Tests for the MCP instrumentation on mcp 2.x.

Every test runs over both client paths: `Client(server)` dispatches in-process
(`DirectDispatcher`, no native spans) and `Client(InMemoryTransport(server))`
goes through the JSON-RPC dispatcher, whose native request span our client
span stands in for. Both sides run in this process through the real SDK, with
no network and no cassettes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from _wrapt_compat import is_wrapt_proxy
from mcp import Client, types
from mcp.client._memory import InMemoryTransport
from mcp.client.session import ClientSession
from mcp.server import _otel
from mcp.server.lowlevel.server import Server
from mcp.server.mcpserver import MCPServer
from mcp.server.session import ServerSession
from mcp.shared import jsonrpc_dispatcher
from mcp.shared.exceptions import MCPError
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp import McpInstrumentor

MARKER = "content-marker-9f3a"
# the tracer mcp 2.x opens its own spans with
NATIVE_SCOPE = "mcp-python-sdk"

Target = Callable[[Any], Any]


def _server() -> MCPServer:
    server = MCPServer("test-server")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        with Laminar.start_as_current_span("inside_tool"):
            return a + b

    @server.tool()
    def echo(text: str) -> str:
        """Echo the text back."""
        return f"echo: {text}"

    @server.tool()
    def fail(reason: str) -> str:
        """Always fails."""
        raise ValueError(f"boom: {reason}")

    return server


@pytest.fixture(params=["direct", "jsonrpc"])
def target(request: pytest.FixtureRequest) -> Target:
    if request.param == "direct":
        return lambda server: server
    return InMemoryTransport


def _by_name(spans: tuple[ReadableSpan, ...], name: str) -> list[ReadableSpan]:
    return [s for s in spans if s.name == name]


def _one(spans: tuple[ReadableSpan, ...], name: str, kind: SpanKind) -> ReadableSpan:
    matching = [s for s in _by_name(spans, name) if s.kind == kind]
    assert len(matching) == 1, [(s.name, s.kind) for s in spans]
    return matching[0]


def _ours(spans: tuple[ReadableSpan, ...]) -> list[ReadableSpan]:
    return [s for s in spans if s.name.endswith((".tool", ".mcp"))]


def _native(spans: tuple[ReadableSpan, ...]) -> list[ReadableSpan]:
    return [
        s for s in spans if s.instrumentation_scope and s.instrumentation_scope.name == NATIVE_SCOPE
    ]


def _span_id(span: ReadableSpan) -> int:
    assert span.context is not None
    return span.context.span_id


def _trace_id(span: ReadableSpan) -> int:
    assert span.context is not None
    return span.context.trace_id


def _parent_id(span: ReadableSpan) -> int | None:
    return span.parent.span_id if span.parent else None


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _exception_events(span: ReadableSpan) -> list[dict[str, Any]]:
    return [dict(e.attributes or {}) for e in span.events if e.name == "exception"]


def _all_recorded_text(spans: tuple[ReadableSpan, ...]) -> str:
    chunks: list[str] = []
    for span in spans:
        sources = [span.attributes or {}, *(e.attributes or {} for e in span.events)]
        chunks.extend(str(value) for attributes in sources for value in attributes.values())
        if span.status.description:
            chunks.append(span.status.description)
    return "\n".join(chunks)


@pytest.mark.asyncio
async def test_tool_call_client_and_server_spans(
    span_exporter: InMemorySpanExporter, target: Target
):
    async with Client(target(_server())) as client:
        result = await client.call_tool("add", {"a": 2, "b": 3})
    assert result.is_error is False

    spans = span_exporter.get_finished_spans()
    client_span = _one(spans, "add.tool", SpanKind.CLIENT)
    server_span = _one(spans, "add.tool", SpanKind.SERVER)

    attrs = _attrs(client_span)
    assert attrs["lmnr.span.type"] == "TOOL"
    assert attrs["mcp.method.name"] == "tools/call"
    assert attrs["gen_ai.tool.name"] == "add"
    assert attrs["lmnr.span.instrumentation_scope.name"] == "mcp"
    assert json.loads(attrs["lmnr.span.input"]) == {
        "tool_name": "add",
        "arguments": {"a": 2, "b": 3},
    }
    output = json.loads(attrs["lmnr.span.output"])
    assert output["content"][0]["text"] == "5"
    assert output["isError"] is False
    assert not _exception_events(client_span)

    server_attrs = _attrs(server_span)
    assert server_attrs["lmnr.span.type"] == "TOOL"
    assert server_attrs["lmnr.span.instrumentation_scope.name"] == "mcp"
    assert server_attrs["mcp.request.id"]
    assert json.loads(server_attrs["lmnr.span.input"]) == json.loads(attrs["lmnr.span.input"])
    assert json.loads(server_attrs["lmnr.span.output"])["content"][0]["text"] == "5"


@pytest.mark.asyncio
async def test_trace_context_propagates_from_client_to_server(
    span_exporter: InMemorySpanExporter, target: Target
):
    with Laminar.start_as_current_span("outer"):
        async with Client(target(_server())) as client:
            _ = await client.call_tool("add", {"a": 1, "b": 1})

    spans = span_exporter.get_finished_spans()
    outer = _one(spans, "outer", SpanKind.INTERNAL)
    client_span = _one(spans, "add.tool", SpanKind.CLIENT)
    server_span = _one(spans, "add.tool", SpanKind.SERVER)
    inside_tool = _one(spans, "inside_tool", SpanKind.INTERNAL)

    assert {_trace_id(s) for s in spans} == {_trace_id(outer)}
    assert _parent_id(client_span) == _span_id(outer)
    assert _parent_id(server_span) == _span_id(client_span)
    assert _parent_id(inside_tool) == _span_id(server_span)
    # ours stand in for mcp's own request and middleware spans
    assert not _native(spans)


@pytest.mark.asyncio
async def test_discover_handshake(span_exporter: InMemorySpanExporter, target: Target):
    async with Client(target(_server())) as client:
        tools = await client.list_tools()
    assert {t.name for t in tools.tools} == {"add", "echo", "fail"}

    spans = span_exporter.get_finished_spans()
    # `send_discover` bypasses `send_request`
    discover = _one(spans, "server/discover.mcp", SpanKind.CLIENT)
    attrs = _attrs(discover)
    assert attrs["mcp.method.name"] == "server/discover"
    assert attrs["mcp.protocol.version"]
    assert json.loads(attrs["lmnr.span.input"]) == {"method": "server/discover"}
    assert json.loads(attrs["lmnr.span.output"])["supportedVersions"]
    server_discover = _one(spans, "server/discover.mcp", SpanKind.SERVER)
    assert _parent_id(server_discover) == _span_id(discover)

    list_tools = _one(spans, "tools/list.mcp", SpanKind.CLIENT)
    attrs = _attrs(list_tools)
    assert attrs["lmnr.span.type"] == "DEFAULT"
    assert "gen_ai.tool.name" not in attrs
    assert json.loads(attrs["lmnr.span.input"])["method"] == "tools/list"
    output = json.loads(attrs["lmnr.span.output"])
    assert {t["name"] for t in output["tools"]} == {"add", "echo", "fail"}
    server_list_tools = _one(spans, "tools/list.mcp", SpanKind.SERVER)
    assert _parent_id(server_list_tools) == _span_id(list_tools)
    assert not _native(spans)


@pytest.mark.asyncio
async def test_legacy_handshake(span_exporter: InMemorySpanExporter, target: Target):
    async with Client(target(_server()), mode="legacy") as client:
        _ = await client.list_tools()

    spans = span_exporter.get_finished_spans()
    # unlike 1.x, `initialize` goes through the server middleware too
    initialize = _one(spans, "initialize.mcp", SpanKind.CLIENT)
    server_initialize = _one(spans, "initialize.mcp", SpanKind.SERVER)
    assert _parent_id(server_initialize) == _span_id(initialize)
    assert not _by_name(spans, "server/discover.mcp")
    # `notifications/initialized` is not traced
    assert not [s for s in spans if "notifications/" in s.name]
    assert not _native(spans)


@pytest.mark.asyncio
async def test_tool_error_result(span_exporter: InMemorySpanExporter, target: Target):
    async with Client(target(_server())) as client:
        result = await client.call_tool("fail", {"reason": "bad input"})
    assert result.is_error is True

    spans = span_exporter.get_finished_spans()
    for kind in (SpanKind.CLIENT, SpanKind.SERVER):
        span = _one(spans, "fail.tool", kind)
        assert span.status.status_code == StatusCode.ERROR
        assert _attrs(span)["error.type"] == "tool_error"
        events = _exception_events(span)
        assert len(events) == 1
        assert events[0]["exception.type"] == "tool_error"
        # MCPServer masks the tool's exception before it reaches the result
        assert events[0]["exception.message"] == "Error executing tool fail"
        assert json.loads(_attrs(span)["lmnr.span.output"])["isError"] is True


@pytest.mark.asyncio
async def test_json_rpc_error(span_exporter: InMemorySpanExporter, target: Target):
    # a low-level server with no prompts handler answers "Method not found"
    async with Client(target(Server("bare"))) as client:
        with pytest.raises(MCPError):
            _ = await client.session.send_request(
                types.ListPromptsRequest(), types.ListPromptsResult
            )

    spans = span_exporter.get_finished_spans()
    client_span = _one(spans, "prompts/list.mcp", SpanKind.CLIENT)
    server_span = _one(spans, "prompts/list.mcp", SpanKind.SERVER)
    assert _parent_id(server_span) == _span_id(client_span)
    for span in (client_span, server_span):
        assert span.status.status_code == StatusCode.ERROR
        assert _attrs(span)["error.type"] == "MCPError"
        assert _attrs(span)["rpc.response.status_code"] == "-32601"
        events = _exception_events(span)
        assert len(events) == 1
        assert events[0]["exception.type"] == "MCPError"
        assert "Method not found" in events[0]["exception.message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("trace_content", [True, False])
async def test_invalid_params(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    target: Target,
    trace_content: bool,
):
    monkeypatch.setenv("LMNR_TRACE_CONTENT", str(trace_content).lower())
    # a modern request without the `_meta` envelope `send_request` stamps is
    # rejected before the middleware chain
    async with Client(target(_server()), mode="legacy") as client:
        with pytest.raises(MCPError):
            # no tool name; the typed request would refuse to build
            _ = await client.session._dispatcher.send_raw_request(
                "tools/call", {"arguments": {"text": MARKER}}, {}
            )

    span = _one(span_exporter.get_finished_spans(), "tools/call.mcp", SpanKind.SERVER)
    assert _attrs(span)["rpc.response.status_code"] == "-32602"
    events = _exception_events(span)
    assert len(events) == 1
    assert events[0]["exception.type"] == "ValidationError"
    # pydantic quotes the client's input
    assert (MARKER in _all_recorded_text((span,))) is trace_content


@pytest.mark.asyncio
async def test_content_not_recorded_when_disabled(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch, target: Target
):
    monkeypatch.setenv("LMNR_TRACE_CONTENT", "false")
    async with Client(target(_server())) as client:
        _ = await client.call_tool("echo", {"text": MARKER})
        _ = await client.call_tool("fail", {"reason": MARKER})

    spans = span_exporter.get_finished_spans()
    assert _by_name(spans, "echo.tool")
    assert MARKER not in _all_recorded_text(spans)
    for span in spans:
        assert "lmnr.span.input" not in _attrs(span)
        assert "lmnr.span.output" not in _attrs(span)
    # the failure is still marked, without the tool's message
    for kind in (SpanKind.CLIENT, SpanKind.SERVER):
        fail = _one(spans, "fail.tool", kind)
        assert _exception_events(fail)[0]["exception.type"] == "tool_error"
    # propagation is not content: the server span still joins the client's trace
    server_echo = _one(spans, "echo.tool", SpanKind.SERVER)
    assert _parent_id(server_echo) == _span_id(_one(spans, "echo.tool", SpanKind.CLIENT))


@pytest.mark.asyncio
async def test_caller_traceparent(span_exporter: InMemorySpanExporter, target: Target):
    remote_trace_id = 0x0AF7651916CD43DD8448EB211C80319C
    remote_span_id = 0xB7AD6B7169203331
    traceparent = f"00-{remote_trace_id:032x}-{remote_span_id:016x}-01"
    request = types.CallToolRequest(
        params=types.CallToolRequestParams(
            name="add", arguments={"a": 1, "b": 2}, _meta={"traceparent": traceparent}
        )
    )

    async with Client(target(_server())) as client:
        result = await client.session.send_request(request, types.CallToolResult)
    assert result.content[0].model_dump()["text"] == "3"
    # the caller's request object is never mutated
    assert request.params.meta == {"traceparent": traceparent}

    spans = span_exporter.get_finished_spans()
    server_span = _one(spans, "add.tool", SpanKind.SERVER)
    if target is not InMemoryTransport:
        assert _trace_id(server_span) == remote_trace_id
        assert _parent_id(server_span) == remote_span_id
    else:
        # the JSON-RPC dispatcher injects the current context over `_meta`
        # itself, as it does without us; the current span is our client span
        assert _parent_id(server_span) == _span_id(_one(spans, "add.tool", SpanKind.CLIENT))


@pytest.mark.asyncio
async def test_request_is_not_mutated(span_exporter: InMemorySpanExporter, target: Target):
    request = types.ListToolsRequest()
    async with Client(target(_server())) as client:
        _ = await client.session.send_request(request, types.ListToolsResult)
    assert request.params is None
    assert _by_name(span_exporter.get_finished_spans(), "tools/list.mcp")


@pytest.mark.asyncio
async def test_suppressed_instrumentation(span_exporter: InMemorySpanExporter, target: Target):
    token = context_api.attach(
        context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
        async with Client(target(_server())) as client:
            _ = await client.call_tool("echo", {"text": "hi"})
    finally:
        context_api.detach(token)

    assert not _ours(span_exporter.get_finished_spans())


@pytest.fixture
def _uninstrumented() -> Iterator[McpInstrumentor]:
    instrumentor = McpInstrumentor()
    assert instrumentor.is_instrumented_by_opentelemetry
    try:
        instrumentor.uninstrument()
        yield instrumentor
    finally:
        instrumentor.instrument()


def test_uninstrument_removes_wrappers(_uninstrumented: McpInstrumentor):
    assert not is_wrapt_proxy(ClientSession.__dict__["send_request"])
    assert not is_wrapt_proxy(ClientSession.__dict__["send_discover"])
    assert not is_wrapt_proxy(ServerSession.__dict__["send_request"])
    assert not is_wrapt_proxy(_otel.OpenTelemetryMiddleware.__dict__["__call__"])
    assert not is_wrapt_proxy(jsonrpc_dispatcher.__dict__["otel_span"])


@pytest.mark.asyncio
async def test_native_spans_after_uninstrument(
    span_exporter: InMemorySpanExporter, _uninstrumented: McpInstrumentor
):
    async with Client(InMemoryTransport(_server())) as client:
        _ = await client.call_tool("echo", {"text": "hi"})

    spans = span_exporter.get_finished_spans()
    assert not _ours(spans)
    # mcp's own spans are back
    assert _native(spans)

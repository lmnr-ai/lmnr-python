"""Tests for the MCP instrumentation.

Client and server talk over mcp's in-memory transport
(`create_connected_server_and_client_session`), so both sides run in this
process through the real SDK, with no network and no cassettes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from _wrapt_compat import is_wrapt_proxy
from mcp import types
from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel.server import Server
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.shared.session import BaseSession
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp import McpInstrumentor

MARKER = "content-marker-9f3a"


def _server() -> FastMCP:
    server = FastMCP("test-server")

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


def _by_name(spans: tuple[ReadableSpan, ...], name: str) -> list[ReadableSpan]:
    return [s for s in spans if s.name == name]


def _one(spans: tuple[ReadableSpan, ...], name: str, kind: SpanKind) -> ReadableSpan:
    matching = [s for s in _by_name(spans, name) if s.kind == kind]
    assert len(matching) == 1, [(s.name, s.kind) for s in spans]
    return matching[0]


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
async def test_tool_call_client_and_server_spans(span_exporter: InMemorySpanExporter):
    async with create_connected_server_and_client_session(_server()) as client:
        result = await client.call_tool("add", {"a": 2, "b": 3})
    assert result.isError is False

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
    assert server_attrs["mcp.request.id"]
    assert json.loads(server_attrs["lmnr.span.input"]) == json.loads(attrs["lmnr.span.input"])
    assert json.loads(server_attrs["lmnr.span.output"])["content"][0]["text"] == "5"


@pytest.mark.asyncio
async def test_trace_context_propagates_from_client_to_server(
    span_exporter: InMemorySpanExporter,
):
    with Laminar.start_as_current_span("outer"):
        async with create_connected_server_and_client_session(_server()) as client:
            _ = await client.call_tool("add", {"a": 1, "b": 1})

    spans = span_exporter.get_finished_spans()
    outer = _one(spans, "outer", SpanKind.INTERNAL)
    client_span = _one(spans, "add.tool", SpanKind.CLIENT)
    server_span = _one(spans, "add.tool", SpanKind.SERVER)
    inside_tool = _one(spans, "inside_tool", SpanKind.INTERNAL)

    trace_id = _trace_id(outer)
    assert {_trace_id(s) for s in (client_span, server_span, inside_tool)} == {trace_id}
    assert _parent_id(client_span) == _span_id(outer)
    # the server span is parented to the client span through `_meta.traceparent`
    assert _parent_id(server_span) == _span_id(client_span)
    assert _parent_id(inside_tool) == _span_id(server_span)


@pytest.mark.asyncio
async def test_non_tool_requests(span_exporter: InMemorySpanExporter):
    async with create_connected_server_and_client_session(_server()) as client:
        tools = await client.list_tools()
    assert {t.name for t in tools.tools} == {"add", "echo", "fail"}

    spans = span_exporter.get_finished_spans()
    # `initialize` is answered by `ServerSession` itself: client span only
    initialize = _one(spans, "initialize.mcp", SpanKind.CLIENT)
    assert not [s for s in _by_name(spans, "initialize.mcp") if s.kind == SpanKind.SERVER]
    assert _attrs(initialize)["mcp.method.name"] == "initialize"

    list_tools = _one(spans, "tools/list.mcp", SpanKind.CLIENT)
    attrs = _attrs(list_tools)
    assert attrs["lmnr.span.type"] == "DEFAULT"
    assert "gen_ai.tool.name" not in attrs
    assert json.loads(attrs["lmnr.span.input"])["method"] == "tools/list"
    output = json.loads(attrs["lmnr.span.output"])
    assert {t["name"] for t in output["tools"]} == {"add", "echo", "fail"}

    server_list_tools = _one(spans, "tools/list.mcp", SpanKind.SERVER)
    assert _parent_id(server_list_tools) == _span_id(list_tools)


@pytest.mark.asyncio
async def test_tool_error_result(span_exporter: InMemorySpanExporter):
    async with create_connected_server_and_client_session(_server()) as client:
        result = await client.call_tool("fail", {"reason": "bad input"})
    assert result.isError is True

    spans = span_exporter.get_finished_spans()
    for kind in (SpanKind.CLIENT, SpanKind.SERVER):
        span = _one(spans, "fail.tool", kind)
        assert span.status.status_code == StatusCode.ERROR
        assert _attrs(span)["error.type"] == "tool_error"
        events = _exception_events(span)
        assert len(events) == 1
        assert events[0]["exception.type"] == "tool_error"
        assert "boom: bad input" in events[0]["exception.message"]
        assert json.loads(_attrs(span)["lmnr.span.output"])["isError"] is True


@pytest.mark.asyncio
async def test_json_rpc_error(span_exporter: InMemorySpanExporter):
    # a low-level server with no prompts handler answers "Method not found"
    server: Server[Any] = Server("bare")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return []

    async with create_connected_server_and_client_session(server) as client:
        with pytest.raises(McpError):
            _ = await client.list_prompts()

    spans = span_exporter.get_finished_spans()
    client_span = _one(spans, "prompts/list.mcp", SpanKind.CLIENT)
    assert client_span.status.status_code == StatusCode.ERROR
    assert _attrs(client_span)["error.type"] == "McpError"
    events = _exception_events(client_span)
    assert len(events) == 1
    assert events[0]["exception.type"] == "McpError"
    assert "Method not found" in events[0]["exception.message"]

    server_span = _one(spans, "prompts/list.mcp", SpanKind.SERVER)
    assert server_span.status.status_code == StatusCode.ERROR
    assert _exception_events(server_span)[0]["exception.type"] == "McpError"
    assert _parent_id(server_span) == _span_id(client_span)


@pytest.mark.asyncio
async def test_content_not_recorded_when_disabled(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LMNR_TRACE_CONTENT", "false")
    async with create_connected_server_and_client_session(_server()) as client:
        _ = await client.call_tool("echo", {"text": MARKER})
        _ = await client.call_tool("fail", {"reason": MARKER})

    spans = span_exporter.get_finished_spans()
    assert _by_name(spans, "echo.tool")
    assert MARKER not in _all_recorded_text(spans)
    for span in spans:
        assert "lmnr.span.input" not in _attrs(span)
        assert "lmnr.span.output" not in _attrs(span)
    # the failure is still marked, without the tool's message
    fail = _one(spans, "fail.tool", SpanKind.CLIENT)
    assert _exception_events(fail)[0]["exception.type"] == "tool_error"
    # propagation is not content: the server span still joins the client's trace
    server_echo = _one(spans, "echo.tool", SpanKind.SERVER)
    assert _parent_id(server_echo) == _span_id(_one(spans, "echo.tool", SpanKind.CLIENT))


@pytest.mark.asyncio
async def test_caller_traceparent_is_kept(span_exporter: InMemorySpanExporter):
    remote_trace_id = 0x0AF7651916CD43DD8448EB211C80319C
    remote_span_id = 0xB7AD6B7169203331
    traceparent = f"00-{remote_trace_id:032x}-{remote_span_id:016x}-01"
    request = types.ClientRequest(
        types.CallToolRequest(
            params=types.CallToolRequestParams(
                name="add",
                arguments={"a": 1, "b": 2},
                _meta=types.RequestParams.Meta.model_validate({"traceparent": traceparent}),
            ),
        )
    )

    async with create_connected_server_and_client_session(_server()) as client:
        result = await client.send_request(request, types.CallToolResult)
    assert result.content[0].model_dump()["text"] == "3"
    # the caller's request object is never mutated
    params = request.root.params
    assert isinstance(params, types.CallToolRequestParams)
    assert params.meta is not None
    assert params.meta.model_extra == {"traceparent": traceparent}

    spans = span_exporter.get_finished_spans()
    server_span = _one(spans, "add.tool", SpanKind.SERVER)
    assert _trace_id(server_span) == remote_trace_id
    assert _parent_id(server_span) == remote_span_id


@pytest.mark.asyncio
async def test_request_is_not_mutated(span_exporter: InMemorySpanExporter):
    request = types.ClientRequest(types.ListToolsRequest())
    async with create_connected_server_and_client_session(_server()) as client:
        _ = await client.send_request(request, types.ListToolsResult)
    assert request.root.params is None
    assert _by_name(span_exporter.get_finished_spans(), "tools/list.mcp")


@pytest.mark.asyncio
async def test_suppressed_instrumentation(span_exporter: InMemorySpanExporter):
    token = context_api.attach(
        context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
        async with create_connected_server_and_client_session(_server()) as client:
            _ = await client.call_tool("echo", {"text": "hi"})
    finally:
        context_api.detach(token)

    # the in-memory server task is spawned under the suppressed context too
    assert not [s for s in span_exporter.get_finished_spans() if s.name.endswith((".tool", ".mcp"))]


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
    assert not is_wrapt_proxy(BaseSession.__dict__["send_request"])
    assert not is_wrapt_proxy(Server.__dict__["_handle_request"])


@pytest.mark.asyncio
async def test_no_spans_after_uninstrument(
    span_exporter: InMemorySpanExporter, _uninstrumented: McpInstrumentor
):
    async with create_connected_server_and_client_session(_server()) as client:
        _ = await client.call_tool("echo", {"text": "hi"})
    assert not [s for s in span_exporter.get_finished_spans() if s.name.endswith((".tool", ".mcp"))]

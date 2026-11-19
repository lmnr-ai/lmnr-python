"""`wrap_mcp_server` on mcp 2.x (`MCPServer`, low-level `Server`) and FastMCP 4.

Client and server run in-process through the real SDKs: `Client(server)`
(`DirectDispatcher`) and `Client(InMemoryTransport(server))` (JSON-RPC
dispatcher). The session-wide `McpInstrumentor` is on, so every test also
checks that a wrapped server's request gets exactly one SERVER span.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

import httpx2
import pytest
from fastmcp import Client as FastMCPClient
from fastmcp import FastMCP
from mcp import Client, types
from mcp.client._memory import InMemoryTransport
from mcp.client.streamable_http import streamable_http_client
from mcp.server import _otel
from mcp.server.lowlevel.server import Server
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from lmnr import Laminar, wrap_mcp_server
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.observed import (
    OBSERVED_SERVER_ATTR,
)
from lmnr.sdk.mcp_server import observe
from lmnr.sdk.mcp_server import wrap as wrap_module
from lmnr.sdk.mcp_server.config import McpServerObservability
from lmnr.sdk.mcp_server.observe import CONTEXT_DESCRIPTION
from lmnr.sdk.mcp_server.turns import turn_trace_id
from lmnr.sdk.mcp_server.v2 import ObservedServerMiddleware

MARKER = "content-marker-7c1e"
AI_INVOCATION = "io.modelcontextprotocol/aiInvocation"
CLIENT_INFO = types.Implementation(name="claude-code", version="2.1.0")

Target = Callable[[Any], Any]


def _server() -> MCPServer:
    server = MCPServer("docs")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        with Laminar.start_as_current_span("inside_tool"):
            return a + b

    @server.tool()
    def login(user: str, password: str) -> dict[str, str]:
        """Returns a token."""
        return {"user": user, "access_token": "tok-" + MARKER}

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


def _strict_lowlevel(received: list[dict[str, Any]]) -> Server:
    """A low-level server whose tool rejects unknown arguments and can raise."""

    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="search",
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                ),
                types.Tool(name="explode", input_schema={"type": "object"}),
            ]
        )

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        arguments = dict(params.arguments or {})
        received.append(arguments)
        if params.name == "explode":
            raise RuntimeError("kaboom " + MARKER)
        if set(arguments) - {"query"}:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="unexpected arguments")],
                is_error=True,
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"found {arguments['query']}")]
        )

    return Server("strict", on_list_tools=list_tools, on_call_tool=call_tool)


@pytest.fixture(params=["direct", "jsonrpc"])
def target(request: pytest.FixtureRequest) -> Target:
    if request.param == "direct":
        return lambda server: server
    return InMemoryTransport


@pytest.fixture
def no_client_spans() -> Iterator[None]:
    """Client calls without the instrumentor's CLIENT span and `traceparent`.

    Only with `Client(server)`: the JSON-RPC dispatcher would still open mcp's
    own client span and send its `traceparent`.
    """
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


def _config(server: Any) -> McpServerObservability:
    config = getattr(server, OBSERVED_SERVER_ATTR)
    assert isinstance(config, McpServerObservability)
    return config


def _never(_: object) -> bool:
    return False


def _not_initialized() -> bool:
    return False


def _meta(values: dict[str, Any]) -> Any:
    """`_meta` with keys the typed `RequestParamsMeta` doesn't declare."""
    return values


def _text(result: Any) -> str:
    return result.content[0].text


@pytest.mark.asyncio
async def test_tool_call_span(span_exporter: InMemorySpanExporter, target: Target):
    server = wrap_mcp_server(_server())
    async with Client(target(server), client_info=CLIENT_INFO) as client:
        _ = await client.list_tools()
        result = await client.call_tool("add", {"a": 2, "b": 3})
    assert result.is_error is False
    assert _text(result) == "5"

    spans = span_exporter.get_finished_spans()
    # discover / tools/list are not traced; no native or instrumentor server span
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
    assert attrs["lmnr.mcp.client.name"] == "claude-code"
    assert attrs["lmnr.mcp.client.version"] == "2.1.0"
    assert json.loads(attrs["gen_ai.tool.call.arguments"]) == {"a": 2, "b": 3}
    tool_result = json.loads(attrs["gen_ai.tool.call.result"])
    assert tool_result["content"][0]["text"] == "5"
    assert tool_result["isError"] is False
    # app-server lifts gen_ai.tool.call.* into input / output for TOOL spans
    assert "lmnr.span.input" not in attrs
    assert "lmnr.span.output" not in attrs
    assert "error.type" not in attrs
    assert "mcp.session.id" not in attrs
    assert not _exception_events(span)

    inner = next(s for s in spans if s.name == "inside_tool")
    assert inner.parent is not None
    assert span.context is not None
    assert inner.parent.span_id == span.context.span_id
    assert _trace_id(inner) == _trace_id(span)


@pytest.mark.asyncio
async def test_legacy_handshake_client_info(span_exporter: InMemorySpanExporter, target: Target):
    server = wrap_mcp_server(_server())
    async with Client(target(server), client_info=CLIENT_INFO, mode="legacy") as client:
        _ = await client.call_tool("add", {"a": 1, "b": 1})

    spans = span_exporter.get_finished_spans()
    # `initialize` is not traced
    assert [s.name for s in _server_spans(spans)] == ["tools/call add"]
    attrs = _attrs(_server_spans(spans)[0])
    assert attrs["lmnr.mcp.client.name"] == "claude-code"
    assert attrs["mcp.protocol.version"]


@pytest.mark.asyncio
async def test_context_injected_into_tools_list(
    span_exporter: InMemorySpanExporter, target: Target, caplog: pytest.LogCaptureFixture
):
    unwrapped = await _server().list_tools()
    with caplog.at_level(logging.WARNING):
        logger = observe.logger
        logger.propagate = True
        try:
            server = wrap_mcp_server(_server())
            async with Client(target(server)) as client:
                tools = {t.name: t for t in (await client.list_tools()).tools}
                _ = await client.list_tools(cache_mode="refresh")
        finally:
            logger.propagate = False

    for name in ("add", "login", "fail"):
        schema = tools[name].input_schema
        assert schema["properties"]["context"] == {
            "type": "string",
            "description": CONTEXT_DESCRIPTION,
        }
        assert "context" not in schema.get("required", [])
        # everything else is as the server declared it
        original = next(t for t in unwrapped if t.name == name).input_schema
        assert {k: v for k, v in schema["properties"].items() if k != "context"} == original[
            "properties"
        ]
    # the tool's own `context` is left alone, with one warning naming it
    original_note = next(t for t in unwrapped if t.name == "note").input_schema
    assert tools["note"].input_schema == original_note
    warnings = [r for r in caplog.records if "'note'" in r.getMessage()]
    assert len(warnings) == 1
    assert not span_exporter.get_finished_spans() or not _server_spans(
        span_exporter.get_finished_spans()
    )


@pytest.mark.asyncio
async def test_context_stripped_before_validation_and_recorded(
    span_exporter: InMemorySpanExporter, target: Target
):
    received: list[dict[str, Any]] = []
    server = wrap_mcp_server(_strict_lowlevel(received))
    goal = "  Find the onboarding doc for a new hire  "
    async with Client(target(server)) as client:
        listed = (await client.list_tools()).tools[0].input_schema
        result = await client.call_tool("search", {"query": "onboarding", "context": goal})
    assert "context" in listed["properties"]
    assert listed["additionalProperties"] is False
    assert result.is_error is False
    assert received == [{"query": "onboarding"}]

    attrs = _attrs(_one_server_span(span_exporter.get_finished_spans(), "tools/call search"))
    assert attrs["lmnr.mcp.context"] == goal.strip()
    # the recorded arguments are what the client sent, `context` included
    assert json.loads(attrs["gen_ai.tool.call.arguments"]) == {
        "query": "onboarding",
        "context": goal,
    }


@pytest.mark.asyncio
async def test_call_without_context(span_exporter: InMemorySpanExporter, target: Target):
    received: list[dict[str, Any]] = []
    server = wrap_mcp_server(_strict_lowlevel(received))
    async with Client(target(server)) as client:
        result = await client.call_tool("search", {"query": "x"})
    assert result.is_error is False
    assert received == [{"query": "x"}]
    attrs = _attrs(_one_server_span(span_exporter.get_finished_spans(), "tools/call search"))
    assert "lmnr.mcp.context" not in attrs


@pytest.mark.asyncio
async def test_context_is_capped(span_exporter: InMemorySpanExporter, target: Target):
    server = wrap_mcp_server(_strict_lowlevel([]))
    async with Client(target(server)) as client:
        _ = await client.call_tool("search", {"query": "x", "context": "y" * 5000})
    attrs = _attrs(_one_server_span(span_exporter.get_finished_spans(), "tools/call search"))
    assert attrs["lmnr.mcp.context"] == "y" * 1000


@pytest.mark.asyncio
async def test_user_intent_fallback(span_exporter: InMemorySpanExporter, target: Target):
    server = wrap_mcp_server(_server())
    async with Client(target(server)) as client:
        _ = await client.call_tool(
            "add", {"a": 1, "b": 2}, meta=_meta({AI_INVOCATION: {"userIntent": "plan a budget"}})
        )
    attrs = _attrs(_one_server_span(span_exporter.get_finished_spans(), "tools/call add"))
    assert attrs["lmnr.mcp.context"] == "plan a budget"


@pytest.mark.asyncio
async def test_tool_with_own_context_untouched(span_exporter: InMemorySpanExporter, target: Target):
    server = wrap_mcp_server(_server())
    # no `tools/list` first: the framework's schema lookup decides
    async with Client(target(server)) as client:
        result = await client.call_tool("note", {"context": "mine", "text": "t"})
    assert _text(result) == "mine|t"
    attrs = _attrs(_one_server_span(span_exporter.get_finished_spans(), "tools/call note"))
    assert "lmnr.mcp.context" not in attrs
    assert json.loads(attrs["gen_ai.tool.call.arguments"]) == {"context": "mine", "text": "t"}


@pytest.mark.asyncio
async def test_context_off(span_exporter: InMemorySpanExporter, target: Target):
    received: list[dict[str, Any]] = []
    server = wrap_mcp_server(_strict_lowlevel(received), context=False)
    async with Client(target(server)) as client:
        listed = (await client.list_tools()).tools[0].input_schema
        result = await client.call_tool("search", {"query": "x", "context": "goal"})
    assert "context" not in listed["properties"]
    # nothing is stripped either: the strict tool sees and rejects it
    assert result.is_error is True
    assert received == [{"query": "x", "context": "goal"}]


@pytest.mark.asyncio
async def test_unknown_tool_and_invalid_arguments(
    span_exporter: InMemorySpanExporter, target: Target
):
    server = wrap_mcp_server(_server())
    async with Client(target(server)) as client:
        unknown = await client.call_tool("nope", {})
        invalid = await client.call_tool("add", {"a": "x"})
    assert unknown.is_error and invalid.is_error

    spans = span_exporter.get_finished_spans()
    for name in ("tools/call nope", "tools/call add"):
        span = _one_server_span(spans, name)
        attrs = _attrs(span)
        assert attrs["error.type"] == "-32602"
        # folded into an `isError` result, not a JSON-RPC error
        assert "rpc.response.status_code" not in attrs
        assert span.status.status_code == StatusCode.ERROR
        assert _exception_events(span)[0]["exception.type"] == "-32602"


@pytest.mark.asyncio
async def test_tool_error_result(span_exporter: InMemorySpanExporter, target: Target):
    server = wrap_mcp_server(_server())
    async with Client(target(server)) as client:
        result = await client.call_tool("fail", {"reason": "r"})
    assert result.is_error is True

    span = _one_server_span(span_exporter.get_finished_spans(), "tools/call fail")
    assert _attrs(span)["error.type"] == "tool_error"
    events = _exception_events(span)
    assert len(events) == 1
    assert events[0]["exception.type"] == "tool_error"
    assert events[0]["exception.message"] == "Error executing tool fail"
    assert json.loads(_attrs(span)["gen_ai.tool.call.result"])["isError"] is True


@pytest.mark.asyncio
async def test_raising_handler(span_exporter: InMemorySpanExporter, target: Target):
    server = wrap_mcp_server(_strict_lowlevel([]))
    async with Client(target(server)) as client:
        with pytest.raises(MCPError):
            _ = await client.call_tool("explode", {})

    span = _one_server_span(span_exporter.get_finished_spans(), "tools/call explode")
    assert _attrs(span)["error.type"] == "RuntimeError"
    events = _exception_events(span)
    assert events[0]["exception.type"] == "RuntimeError"
    assert MARKER in events[0]["exception.message"]
    assert span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_json_rpc_error(span_exporter: InMemorySpanExporter, target: Target):
    # no prompts handler: "Method not found"
    server = wrap_mcp_server(_strict_lowlevel([]))
    async with Client(target(server), mode="legacy") as client:
        with pytest.raises(MCPError):
            _ = await client.session.send_request(
                types.GetPromptRequest(params=types.GetPromptRequestParams(name="p")),
                types.GetPromptResult,
            )

    span = _one_server_span(span_exporter.get_finished_spans(), "prompts/get p")
    attrs = _attrs(span)
    assert attrs["error.type"] == "-32601"
    assert attrs["rpc.response.status_code"] == "-32601"
    assert attrs["gen_ai.prompt.name"] == "p"
    assert attrs["lmnr.span.type"] == "DEFAULT"
    assert _exception_events(span)[0]["exception.type"] == "-32601"


@pytest.mark.asyncio
async def test_prompt_and_resource_spans(span_exporter: InMemorySpanExporter, target: Target):
    server = wrap_mcp_server(_server())
    async with Client(target(server)) as client:
        _ = await client.get_prompt("summarize", {"topic": "otel"})
        _ = await client.read_resource("docs://intro")

    spans = span_exporter.get_finished_spans()
    prompt = _attrs(_one_server_span(spans, "prompts/get summarize"))
    assert prompt["gen_ai.prompt.name"] == "summarize"
    assert prompt["lmnr.span.type"] == "DEFAULT"
    assert json.loads(prompt["lmnr.span.input"]) == {
        "name": "summarize",
        "arguments": {"topic": "otel"},
    }
    assert "Summarize otel" in prompt["lmnr.span.output"]
    assert "gen_ai.tool.call.arguments" not in prompt

    # never the URI in the name
    resource = _attrs(_one_server_span(spans, "resources/read"))
    assert resource["mcp.resource.uri"] == "docs://intro"
    assert json.loads(resource["lmnr.span.input"]) == {"uri": "docs://intro"}
    assert "doc intro" in resource["lmnr.span.output"]


@pytest.mark.asyncio
async def test_redaction(span_exporter: InMemorySpanExporter, target: Target):
    server = wrap_mcp_server(_server(), redact_keys=["User"])
    async with Client(target(server)) as client:
        result = await client.call_tool("login", {"user": "ann", "password": MARKER})
    # the handler result is unchanged
    assert MARKER in _text(result)

    attrs = _attrs(_one_server_span(span_exporter.get_finished_spans(), "tools/call login"))
    assert json.loads(attrs["gen_ai.tool.call.arguments"]) == {
        "user": "[REDACTED]",
        "password": "[REDACTED]",
    }
    # key-based: the structured result is redacted (its JSON rendering in the
    # text block is just a string to the redactor)
    assert json.loads(attrs["gen_ai.tool.call.result"])["structuredContent"] == {
        "user": "[REDACTED]",
        "access_token": "[REDACTED]",
    }


@pytest.mark.asyncio
async def test_record_payloads_off(span_exporter: InMemorySpanExporter, target: Target):
    server = wrap_mcp_server(_server(), record_payloads=False)
    async with Client(target(server)) as client:
        listed = {t.name: t for t in (await client.list_tools()).tools}
        _ = await client.call_tool(
            "add", {"a": 1, "b": 2}, meta=_meta({AI_INVOCATION: {"userIntent": MARKER}})
        )
        _ = await client.call_tool("fail", {"reason": MARKER})
        _ = await client.read_resource("docs://" + MARKER)
        _ = await client.get_prompt("summarize", {"topic": MARKER})
    # no `context` field to fill in
    assert "context" not in listed["add"].input_schema["properties"]

    spans = _server_spans(span_exporter.get_finished_spans())
    assert len(spans) == 4
    for span in spans:
        attrs = _attrs(span)
        for key in (
            "gen_ai.tool.call.arguments",
            "gen_ai.tool.call.result",
            "lmnr.span.input",
            "lmnr.span.output",
            "lmnr.mcp.context",
            "mcp.resource.uri",
        ):
            assert key not in attrs
        recorded = json.dumps(attrs) + json.dumps(_exception_events(span))
        assert MARKER not in recorded
    fail = _one_server_span(span_exporter.get_finished_spans(), "tools/call fail")
    assert _attrs(fail)["error.type"] == "tool_error"
    assert _exception_events(fail)[0]["exception.message"] == "Tool returned an error"
    # names stay
    assert (
        _attrs(_one_server_span(span_exporter.get_finished_spans(), "tools/call add"))[
            "gen_ai.tool.name"
        ]
        == "add"
    )


@pytest.mark.asyncio
async def test_payloads_off_raising_handler(span_exporter: InMemorySpanExporter, target: Target):
    server = wrap_mcp_server(_strict_lowlevel([]), record_payloads=False)
    async with Client(target(server)) as client:
        with pytest.raises(MCPError):
            _ = await client.call_tool("explode", {})
    span = _one_server_span(span_exporter.get_finished_spans(), "tools/call explode")
    assert _attrs(span)["error.type"] == "RuntimeError"
    event = _exception_events(span)[0]
    assert MARKER not in json.dumps(event)
    assert MARKER not in (span.status.description or "")


@pytest.mark.asyncio
async def test_session_turns(span_exporter: InMemorySpanExporter, no_client_spans: None):
    server = wrap_mcp_server(_server())
    now = [1000.0]
    _config(server).turns._clock = lambda: now[0]
    session = _meta({"openai/session": "chat-1"})
    async with Client(server) as client:
        _ = await client.call_tool("add", {"a": 1, "b": 1}, meta=session)
        now[0] += 19
        _ = await client.call_tool("add", {"a": 1, "b": 2}, meta=session)
        now[0] += 21
        _ = await client.call_tool("add", {"a": 1, "b": 3}, meta=session)
        # another session, and no session at all, get traces of their own
        _ = await client.call_tool(
            "add", {"a": 1, "b": 4}, meta=_meta({"openai/session": "chat-2"})
        )
        _ = await client.call_tool("add", {"a": 1, "b": 5})
        _ = await client.call_tool("add", {"a": 1, "b": 6})

    spans = [s for s in _server_spans(span_exporter.get_finished_spans())]
    assert len(spans) == 6
    ids = [_trace_id(s) for s in spans]
    assert ids[0] == ids[1]
    assert len(set(ids[1:])) == 5
    for span in spans:
        assert span.parent is None
    for span in spans[:3]:
        assert _attrs(span)["mcp.session.id"] == "chat-1"
        assert _attrs(span)["lmnr.association.properties.session_id"] == "chat-1"
    assert "mcp.session.id" not in _attrs(spans[4])


@pytest.mark.asyncio
async def test_streamable_http_session(span_exporter: InMemorySpanExporter):
    server = wrap_mcp_server(_server())
    # the host must pass the transport's DNS-rebinding check
    url = "http://127.0.0.1:8000/mcp"
    app = server.streamable_http_app(json_response=True)
    async with server.session_manager.run():
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport) as http:
            # stateful (handshake) sessions carry `Mcp-Session-Id`
            async with Client(
                streamable_http_client(url, http_client=http), mode="legacy"
            ) as client:
                _ = await client.call_tool("add", {"a": 1, "b": 2})
                _ = await client.call_tool("add", {"a": 1, "b": 3})
            # 2026-07-28 requests are stateless: no session
            async with Client(streamable_http_client(url, http_client=http)) as client:
                _ = await client.call_tool("add", {"a": 1, "b": 4})

    spans = _server_spans(span_exporter.get_finished_spans())
    assert len(spans) == 3
    first, second, stateless = (_attrs(s) for s in spans)
    assert first["mcp.session.id"]
    assert first["mcp.session.id"] == second["mcp.session.id"]
    assert first["lmnr.association.properties.session_id"] == first["mcp.session.id"]
    assert "mcp.session.id" not in stateless
    for attrs in (first, second, stateless):
        assert attrs["network.transport"] == "tcp"
        assert attrs["network.protocol.name"] == "http"
        assert attrs["mcp.protocol.version"]


@pytest.mark.asyncio
async def test_turn_id(span_exporter: InMemorySpanExporter, no_client_spans: None):
    server = wrap_mcp_server(_server())
    meta = _meta({"openai/session": "s-1", AI_INVOCATION: {"turnId": "turn-42"}})
    async with Client(server) as client:
        _ = await client.call_tool("add", {"a": 1, "b": 1}, meta=meta)
        _ = await client.call_tool("add", {"a": 1, "b": 2}, meta=meta)
        # no session: hashed with ""
        _ = await client.call_tool(
            "add", {"a": 1, "b": 3}, meta=_meta({AI_INVOCATION: {"turnId": "t"}})
        )

    spans = _server_spans(span_exporter.get_finished_spans())
    assert [_trace_id(s) for s in spans] == [
        turn_trace_id("s-1", "turn-42"),
        turn_trace_id("s-1", "turn-42"),
        turn_trace_id(None, "t"),
    ]
    assert turn_trace_id("s-1", "turn-42") == int(
        __import__("hashlib").sha256(b"s-1\nturn-42").hexdigest()[:32], 16
    )


@pytest.mark.asyncio
async def test_traceparent_is_linked_not_parented(
    span_exporter: InMemorySpanExporter, no_client_spans: None
):
    remote_trace_id = 0x0AF7651916CD43DD8448EB211C80319C
    remote_span_id = 0xB7AD6B7169203331
    traceparent = f"00-{remote_trace_id:032x}-{remote_span_id:016x}-01"
    server = wrap_mcp_server(_server())
    async with Client(server) as client:
        request = types.CallToolRequest(
            params=types.CallToolRequestParams(
                name="add", arguments={"a": 1, "b": 2}, _meta={"traceparent": traceparent}
            )
        )
        _ = await client.session.send_request(request, types.CallToolResult)

    span = _one_server_span(span_exporter.get_finished_spans(), "tools/call add")
    assert _trace_id(span) == remote_trace_id
    assert span.parent is None
    assert len(span.links) == 1
    link_context = span.links[0].context
    assert (link_context.trace_id, link_context.span_id) == (remote_trace_id, remote_span_id)


@pytest.mark.asyncio
async def test_client_span_trace_is_reused(span_exporter: InMemorySpanExporter, target: Target):
    # with the instrumentor on, the client's span rides in `_meta.traceparent`
    server = wrap_mcp_server(_server())
    async with Client(server) as client:
        _ = await client.call_tool("add", {"a": 1, "b": 2})

    spans = span_exporter.get_finished_spans()
    server_span = _one_server_span(spans, "tools/call add")
    client_span = next(s for s in spans if s.kind == SpanKind.CLIENT and s.name == "add.tool")
    assert server_span.parent is None
    assert _trace_id(server_span) == _trace_id(client_span)
    assert client_span.context is not None
    assert server_span.links[0].context.span_id == client_span.context.span_id


@pytest.mark.asyncio
async def test_unforceable_provider_degrades_to_random(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    no_client_spans: None,
):
    monkeypatch.setattr(observe, "can_force_trace_id", _never)
    server = wrap_mcp_server(_server())
    meta = _meta({AI_INVOCATION: {"turnId": "turn-1"}})
    async with Client(server) as client:
        _ = await client.call_tool("add", {"a": 1, "b": 1}, meta=meta)
    span = _one_server_span(span_exporter.get_finished_spans(), "tools/call add")
    assert _trace_id(span) != turn_trace_id(None, "turn-1")
    assert span.parent is None


def test_turn_gap_is_clamped(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wrap_module.logger, "propagate", True)
    with caplog.at_level(logging.WARNING):
        server = wrap_mcp_server(_server(), turn_gap_seconds=60)
    assert _config(server).turns.turn_gap_seconds == 25
    assert any("turn_gap_seconds" in r.getMessage() for r in caplog.records)
    assert _config(wrap_mcp_server(_server(), turn_gap_seconds=5)).turns.turn_gap_seconds == 5
    assert _config(wrap_mcp_server(_server())).turns.turn_gap_seconds == 20


def test_wrapping_twice_is_a_no_op():
    server = _server()
    assert wrap_mcp_server(server) is server
    config = _config(server)
    assert wrap_mcp_server(server) is server
    assert _config(server) is config
    middleware = server._lowlevel_server.middleware
    assert sum(isinstance(m, ObservedServerMiddleware) for m in middleware) == 1
    assert not any(type(m).__name__ == "OpenTelemetryMiddleware" for m in middleware)
    assert isinstance(middleware[0], ObservedServerMiddleware)


@pytest.mark.asyncio
async def test_native_middleware_added_after_wrapping(
    span_exporter: InMemorySpanExporter, target: Target
):
    server = wrap_mcp_server(_server())
    # the instrumentor's replacement of it stands aside for the wrapped request
    server._lowlevel_server.middleware.append(_otel.OpenTelemetryMiddleware())
    async with Client(target(server)) as client:
        _ = await client.call_tool("add", {"a": 1, "b": 2})
    assert [s.name for s in _server_spans(span_exporter.get_finished_spans())] == ["tools/call add"]


def test_unsupported_object_is_returned_unchanged():
    thing = object()
    assert wrap_mcp_server(thing) is thing


@pytest.mark.asyncio
async def test_results_unchanged(span_exporter: InMemorySpanExporter, target: Target):
    async with Client(target(_server())) as client:
        plain = await client.call_tool("login", {"user": "u", "password": "p"})
    async with Client(target(wrap_mcp_server(_server()))) as client:
        wrapped = await client.call_tool("login", {"user": "u", "password": "p"})
    assert wrapped.model_dump() == plain.model_dump()


@pytest.mark.asyncio
async def test_not_initialized(
    span_exporter: InMemorySpanExporter, target: Target, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(Laminar, "is_initialized", staticmethod(_not_initialized))
    received: list[dict[str, Any]] = []
    server = wrap_mcp_server(_strict_lowlevel(received))
    async with Client(target(server)) as client:
        result = await client.call_tool("search", {"query": "q", "context": "goal"})
    assert result.is_error is False
    assert received == [{"query": "q"}]
    assert not _server_spans(span_exporter.get_finished_spans())


# --- FastMCP 4 (standalone `fastmcp`, on mcp 2.x) ---------------------------------


def _fastmcp_server_spans(spans: tuple[ReadableSpan, ...]) -> list[ReadableSpan]:
    return [
        s
        for s in spans
        if s.kind == SpanKind.SERVER
        and s.instrumentation_scope is not None
        and s.instrumentation_scope.name == "fastmcp"
    ]


def _fastmcp() -> FastMCP:
    server = FastMCP("fm")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        with Laminar.start_as_current_span("inside_tool"):
            return a + b

    @server.tool
    def fail(reason: str) -> str:
        """Always fails."""
        raise ValueError(f"boom: {reason}")

    return server


@pytest.mark.asyncio
async def test_fastmcp_tool_call(span_exporter: InMemorySpanExporter):
    server = wrap_mcp_server(_fastmcp())
    async with FastMCPClient(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
        # FastMCP rejects unexpected arguments: `context` must be gone first
        result = await client.call_tool("add", {"a": 2, "b": 3, "context": "sum things"})
    assert "context" in tools["add"].input_schema["properties"]
    assert result.data == 5

    spans = span_exporter.get_finished_spans()
    # FastMCP's own seam / server spans are suppressed (its client's spans stay)
    assert not _fastmcp_server_spans(spans)
    span = _one_server_span(spans, "tools/call add")
    assert [s.name for s in _server_spans(spans)] == ["tools/call add"]
    assert span.parent is None
    attrs = _attrs(span)
    assert attrs["lmnr.mcp.context"] == "sum things"
    assert attrs["gen_ai.tool.name"] == "add"
    inner = next(s for s in spans if s.name == "inside_tool")
    assert span.context is not None and inner.parent is not None
    assert inner.parent.span_id == span.context.span_id


@pytest.mark.asyncio
async def test_fastmcp_errors(span_exporter: InMemorySpanExporter):
    server = wrap_mcp_server(_fastmcp())
    async with FastMCPClient(server) as client:
        for name, arguments in (("fail", {"reason": "r"}), ("nope", {}), ("add", {"a": "x"})):
            result = await client.call_tool(name, arguments, raise_on_error=False)
            assert result.is_error

    spans = span_exporter.get_finished_spans()
    assert _attrs(_one_server_span(spans, "tools/call fail"))["error.type"] == "tool_error"
    assert _attrs(_one_server_span(spans, "tools/call nope"))["error.type"] == "-32602"
    assert _attrs(_one_server_span(spans, "tools/call add"))["error.type"] == "-32602"
    assert not _fastmcp_server_spans(spans)


@pytest.mark.asyncio
async def test_fastmcp_unwrapped_keeps_native_spans(span_exporter: InMemorySpanExporter):
    # wrapping one server does not silence another
    _ = wrap_mcp_server(_fastmcp())
    async with FastMCPClient(_fastmcp()) as client:
        _ = await client.call_tool("add", {"a": 1, "b": 2})
    assert _fastmcp_server_spans(span_exporter.get_finished_spans())

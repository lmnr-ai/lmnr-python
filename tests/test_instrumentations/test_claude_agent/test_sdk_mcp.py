import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent.sdk_mcp import (
    build_traceparent,
    restrict_server_spans_to_tool_calls,
    with_traceparent,
)

TP = build_traceparent("a" * 32, "b" * 16)


def _transport(traceparent: str | None = TP) -> Any:
    transport = SimpleNamespace()
    if traceparent:
        setattr(transport, "__lmnr_context", {"traceparent": traceparent})
    return transport


def test_traceparent_is_stamped_into_meta_without_mutating_input():
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    out = with_traceparent(_transport(), message)
    assert out["params"]["_meta"]["traceparent"] == TP
    assert "params" not in message


def test_notification_without_params_gets_meta():
    out = with_traceparent(_transport(), {"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert out["params"] == {"_meta": {"traceparent": TP}}


def test_existing_params_and_meta_are_preserved():
    message = {"method": "tools/call", "params": {"name": "x", "_meta": {"progressToken": 1}}}
    out = with_traceparent(_transport(), message)
    assert out["params"]["name"] == "x"
    assert out["params"]["_meta"] == {"progressToken": 1, "traceparent": TP}


def test_caller_traceparent_wins():
    message = {"method": "x", "params": {"_meta": {"traceparent": "mine"}}}
    assert with_traceparent(_transport(), message) is message


def test_no_published_context_leaves_message_alone():
    message = {"method": "tools/list"}
    assert with_traceparent(_transport(None), message) is message
    assert with_traceparent(None, message) is message


@pytest.mark.asyncio
async def test_only_tool_calls_keep_mcp_spans(monkeypatch: pytest.MonkeyPatch):
    # `mcp.server._otel` only exists in mcp 2.x; fake it so this runs on any version.
    class OpenTelemetryMiddleware:
        async def __call__(self, ctx: Any, call_next: Any) -> str:
            return f"spanned:{await call_next(ctx)}"

    fake = ModuleType("mcp.server._otel")
    fake.OpenTelemetryMiddleware = OpenTelemetryMiddleware  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "mcp.server._otel", fake)

    other = object()
    server = SimpleNamespace(middleware=[other, OpenTelemetryMiddleware()])
    restrict_server_spans_to_tool_calls(server)
    assert server.middleware[0] is other
    wrapper = server.middleware[1]
    assert not isinstance(wrapper, OpenTelemetryMiddleware)

    # a second pass must not double-wrap
    restrict_server_spans_to_tool_calls(server)
    assert server.middleware[1] is wrapper

    async def call_next(ctx: Any) -> str:
        return "ok"

    assert await wrapper(SimpleNamespace(method="tools/list"), call_next) == "ok"
    assert await wrapper(SimpleNamespace(method="tools/call"), call_next) == "spanned:ok"


def test_servers_without_middleware_are_left_alone():
    restrict_server_spans_to_tool_calls(SimpleNamespace())
    restrict_server_spans_to_tool_calls(None)


@pytest.mark.asyncio
async def test_server_built_before_instrumentation_only_traces_tool_calls(
    span_exporter: InMemorySpanExporter,
):
    """Servers created at import time, before `Laminar.initialize()`, never went
    through the `create_sdk_mcp_server` wrapper. The bridge wrapper must still
    restrict their spans and parent them."""
    otel = pytest.importorskip("mcp.server._otel")
    from claude_agent_sdk import create_sdk_mcp_server, tool
    from claude_agent_sdk._internal.query import Query

    @tool("add", "Add two numbers", {"a": int, "b": int})
    async def add(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": str(args["a"] + args["b"])}]}

    server = create_sdk_mcp_server(name="toy", tools=[add])["instance"]
    # Undo the creation-time swap: this is what an import-time server looks like.
    server.middleware = [otel.OpenTelemetryMiddleware()]

    with Laminar.start_as_current_span("turn") as turn:
        ctx = turn.get_span_context()
        transport = _transport(build_traceparent(f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"))
    query = Query(transport=transport, is_streaming_mode=True, sdk_mcp_servers={"toy": server})
    try:
        messages: list[dict[str, Any]] = [
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "cli", "version": "0"},
                },
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "add", "arguments": {"a": 2, "b": 3}},
            },
        ]
        responses = [await query._handle_sdk_mcp_request("toy", m) for m in messages]
    finally:
        for bridge in query._sdk_mcp_bridges.values():
            await bridge.aclose()

    assert "error" not in (responses[3] or {}), responses
    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    assert set(spans) == {"turn", "add.tool"}, sorted(spans)
    tool_span = spans["add.tool"]
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == ctx.span_id
    assert tool_span.context is not None
    assert tool_span.context.trace_id == ctx.trace_id

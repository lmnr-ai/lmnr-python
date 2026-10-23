import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

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
    config = {"type": "sdk", "name": "t", "instance": server}

    restrict_server_spans_to_tool_calls(config)
    assert server.middleware[0] is other
    wrapper = server.middleware[1]
    assert not isinstance(wrapper, OpenTelemetryMiddleware)

    # a second pass must not double-wrap
    restrict_server_spans_to_tool_calls(config)
    assert server.middleware[1] is wrapper

    async def call_next(ctx: Any) -> str:
        return "ok"

    assert await wrapper(SimpleNamespace(method="tools/list"), call_next) == "ok"
    assert await wrapper(SimpleNamespace(method="tools/call"), call_next) == "spanned:ok"


def test_servers_without_middleware_are_left_alone():
    restrict_server_spans_to_tool_calls({"type": "sdk", "instance": SimpleNamespace()})
    restrict_server_spans_to_tool_calls(None)

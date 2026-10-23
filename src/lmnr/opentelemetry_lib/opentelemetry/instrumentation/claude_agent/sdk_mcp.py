"""Helpers for in-process (SDK) MCP servers.

`claude-agent-sdk` serves these through an `SdkMcpBridge` whose server task is
detached from any Laminar span, and `mcp` 2.x ships `OpenTelemetryMiddleware`
on every server. Left alone, each handshake / `tools/list` / `tools/call`
message becomes a parentless root span. We (1) stamp a `traceparent` into every
bridged message so `mcp` nests its span under the active Claude Agent span and
(2) keep only the `tools/call` spans, since the lifecycle ones carry no signal.
"""

import importlib
from collections.abc import Awaitable, Callable
from typing import Any, cast

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


def build_traceparent(trace_id: str, span_id: str) -> str:
    return f"00-{trace_id}-{span_id}-01"


def with_traceparent(transport: Any, message: dict[str, Any]) -> dict[str, Any]:
    """Return `message` with the transport's latest published span as `_meta.traceparent`.

    The message is copied, never mutated. A caller-supplied `traceparent` wins.
    """
    context: dict[str, Any] | None = getattr(transport, "__lmnr_context", None)
    traceparent = context.get("traceparent") if context else None
    if not traceparent:
        return message

    params = message.get("params")
    if params is not None and not isinstance(params, dict):
        return message
    params = params or {}
    meta = params.get("_meta")
    if meta is not None and not isinstance(meta, dict):
        return message
    meta = meta or {}
    if "traceparent" in meta:
        return message

    return {**message, "params": {**params, "_meta": {**meta, "traceparent": traceparent}}}


class _ToolCallsOnly:
    """Wraps `mcp`'s `OpenTelemetryMiddleware`, tracing only `tools/call`."""

    _inner: Callable[[Any, Any], Awaitable[Any]]

    def __init__(self, inner: Callable[[Any, Any], Awaitable[Any]]):
        self._inner = inner

    async def __call__(self, ctx: Any, call_next: Any) -> Any:
        if ctx.method != "tools/call":
            return await call_next(ctx)
        return await self._inner(ctx, call_next)


def restrict_server_spans_to_tool_calls(server_config: Any) -> None:
    """Swap the default `mcp` span middleware on an SDK server for a `tools/call`-only one.

    No-op on `mcp` versions without server middleware.
    """
    try:
        otel_middleware_cls: Any = importlib.import_module(
            "mcp.server._otel"
        ).OpenTelemetryMiddleware
    except (ImportError, AttributeError):
        return

    server: Any = server_config.get("instance") if isinstance(server_config, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    middleware: Any = getattr(server, "middleware", None)
    if not isinstance(middleware, list):
        return
    for i, mw in enumerate(cast(list[Any], middleware)):
        if type(mw) is otel_middleware_cls:
            middleware[i] = _ToolCallsOnly(mw)

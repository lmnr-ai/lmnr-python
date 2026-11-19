"""What a server's request spans record (`McpServerSettings`) and what a request carries.

`wrap_mcp_server` attaches settings to a server (`SETTINGS_ATTR`); every other
server is recorded with `UNWRAPPED`. The server hooks in `v1.py` / `v2.py` read
them; nothing else changes between wrapped and unwrapped servers.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, cast

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.turns import TurnTracker
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    should_send_prompts,
)

TOOLS_CALL = "tools/call"
PROMPTS_GET = "prompts/get"
RESOURCES_READ = "resources/read"
TOOLS_LIST = "tools/list"
# what a wrapped server traces; listing, `initialize`, `ping` would mint noise traces
WRAPPED_METHODS = frozenset({TOOLS_CALL, PROMPTS_GET, RESOURCES_READ})

AI_INVOCATION_META_KEY = "io.modelcontextprotocol/aiInvocation"
OPENAI_SESSION_META_KEY = "openai/session"

# on the low-level `Server` (and, on mcp 2.x, its `OpenTelemetryMiddleware`)
SETTINGS_ATTR = "_lmnr_mcp_server_settings"

# A tool's advertised input schema by name, or None when unknown.
SchemaLookup = Callable[[str], Awaitable[Mapping[str, Any] | None]]


def _no_suppression() -> AbstractContextManager[Any]:
    return nullcontext()


@dataclass(eq=False)
class McpServerSettings:
    # None for an unwrapped server: spans are parented on the client's
    # `traceparent` and every request is traced. Set: root spans grouped by turn.
    turns: TurnTracker | None = None
    context: bool = False
    record_payloads: bool = True
    redact_keys: frozenset[str] = frozenset()
    # the framework's lookup of a tool's schema; None when the server has none
    lookup_tool_schema: SchemaLookup | None = None
    # turns the framework's own (native) spans off for the block (FastMCP)
    suppress_native: Callable[[], AbstractContextManager[Any]] = _no_suppression
    # tool names seen in `tools/list` that declare their own `context`
    own_context_tools: set[str] = field(default_factory=set[str])
    warned_tools: set[str] = field(default_factory=set[str])
    lock: threading.Lock = field(default_factory=threading.Lock)

    def traces(self, method: str) -> bool:
        return self.turns is None or method in WRAPPED_METHODS

    def payloads(self) -> bool:
        """Record arguments, results and context (`LMNR_TRACE_CONTENT=false` also turns it off)."""
        return self.record_payloads and should_send_prompts()

    def context_field(self) -> bool:
        """Inject / strip the `context` argument."""
        return self.context and self.payloads()


UNWRAPPED = McpServerSettings()


def settings_of(obj: object) -> McpServerSettings | None:
    settings = getattr(obj, SETTINGS_ATTR, None)
    return settings if isinstance(settings, McpServerSettings) else None


@dataclass
class McpRequest:
    """What a request carries, read off either mcp major."""

    method: str
    # the wire params, without `_meta`
    params: Mapping[str, Any]
    meta: Mapping[str, Any] = field(default_factory=dict[str, Any])
    request_id: str | None = None
    # the transport's session id (`Mcp-Session-Id`)
    transport_session_id: str | None = None
    protocol_version: str | None = None
    # True for HTTP transports, None when unknown
    http: bool | None = None
    client_name: str | None = None
    client_version: str | None = None

    @property
    def target(self) -> str | None:
        name = self.params.get("name")
        return name if self.method in (TOOLS_CALL, PROMPTS_GET) and isinstance(name, str) else None

    @property
    def session_id(self) -> str | None:
        if self.transport_session_id:
            return self.transport_session_id
        session = self.meta.get(OPENAI_SESSION_META_KEY)
        return session if isinstance(session, str) and session else None

    @property
    def ai_invocation(self) -> Mapping[str, Any]:
        value = self.meta.get(AI_INVOCATION_META_KEY)
        return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def as_dict(value: Any) -> dict[str, Any]:
    """A model or mapping as a plain dict in its wire shape; {} otherwise."""
    if value is None:
        return {}
    if hasattr(value, "model_dump") and not isinstance(value, type):
        value = value.model_dump(by_alias=True, mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, Any], value))
    return {}

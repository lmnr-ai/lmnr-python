"""What a server's request spans record (`McpServerSettings`) and what a request carries.

`Laminar.initialize(mcp=...)` turns MCP observability mode on for the process
(`tracing/mcp_options.py`); `current_settings()` gives the hooks in `v1.py` /
`v2.py` its settings, or `OFF` without it. Nothing else changes between the two.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.redaction import normalize_keys
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.turns import TurnTracker
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    should_send_prompts,
)
from lmnr.opentelemetry_lib.tracing.mcp_options import ResolvedMcpOptions, get_mcp_options

TOOLS_CALL = "tools/call"
PROMPTS_GET = "prompts/get"
RESOURCES_READ = "resources/read"
TOOLS_LIST = "tools/list"
# what MCP observability mode traces; listing, `initialize`, `ping` would mint noise traces
OBSERVED_METHODS = frozenset({TOOLS_CALL, PROMPTS_GET, RESOURCES_READ})

AI_INVOCATION_META_KEY = "io.modelcontextprotocol/aiInvocation"
OPENAI_SESSION_META_KEY = "openai/session"

# A tool's advertised input schema by name, or None when unknown.
SchemaLookup = Callable[[str], Awaitable[Mapping[str, Any] | None]]


@dataclass(eq=False)
class McpServerSettings:
    # None with the mode off: spans are parented on the client's `traceparent`
    # and every request is traced. Set: root spans grouped by turn.
    options: ResolvedMcpOptions | None = None
    turns: TurnTracker | None = None
    redact_keys: frozenset[str] = frozenset()
    # tool names seen in `tools/list` that declare their own `context`
    own_context_tools: set[str] = field(default_factory=set[str])
    warned_tools: set[str] = field(default_factory=set[str])
    lock: threading.Lock = field(default_factory=threading.Lock)

    def traces(self, method: str) -> bool:
        return self.options is None or method in OBSERVED_METHODS

    def payloads(self) -> bool:
        """Record arguments, results and context (`LMNR_TRACE_CONTENT=false` also turns it off)."""
        return (self.options is None or self.options.record_payloads) and should_send_prompts()

    def context_field(self) -> bool:
        """Inject / strip the `context` argument."""
        return self.options is not None and self.options.context and self.payloads()


OFF = McpServerSettings()
_on: McpServerSettings | None = None


def current_settings() -> McpServerSettings:
    """The process's settings: `OFF`, or one set per `Laminar.initialize(mcp=...)`."""
    global _on
    options = get_mcp_options()
    if options is None:
        return OFF
    settings = _on
    if settings is None or settings.options is not options:
        # a re-initialize starts fresh turns
        settings = _on = McpServerSettings(
            options=options,
            turns=TurnTracker(options.turn_gap_seconds),
            redact_keys=normalize_keys(options.redact_keys),
        )
    return settings


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

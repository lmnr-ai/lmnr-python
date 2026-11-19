"""MCP observability mode as the hooks read it, and what a request carries.

`Laminar.initialize(mcp=...)` stores the mode (`tracing/mcp_options.py`); the
hooks in `v1.py` / `v2.py` read it per request with `get_mcp_options()`, None
being mode off. Nothing else changes between the two modes.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.redaction import normalize_keys
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    should_send_prompts,
)
from lmnr.opentelemetry_lib.tracing.mcp_options import ResolvedMcpOptions

TOOLS_CALL = "tools/call"
PROMPTS_GET = "prompts/get"
RESOURCES_READ = "resources/read"
TOOLS_LIST = "tools/list"
# what MCP observability mode traces; listing, `initialize`, `ping` would mint noise traces
OBSERVED_METHODS = frozenset({TOOLS_CALL, PROMPTS_GET, RESOURCES_READ})

AI_INVOCATION_META_KEY = "io.modelcontextprotocol/aiInvocation"
OPENAI_SESSION_META_KEY = "openai/session"

# None: mode off
Mode = ResolvedMcpOptions | None


def traces(mode: Mode, method: str) -> bool:
    return mode is None or method in OBSERVED_METHODS


def payloads(mode: Mode) -> bool:
    """Record arguments, results and context (`LMNR_TRACE_CONTENT=false` also turns it off)."""
    return (mode is None or mode.record_payloads) and should_send_prompts()


def context_field(mode: Mode) -> bool:
    """Inject / strip the `context` argument."""
    return mode is not None and mode.context and payloads(mode)


@functools.cache
def redact_keys(mode: Mode) -> frozenset[str]:
    """The customer's extra secret keys, normalized."""
    return normalize_keys(mode.redact_keys) if mode is not None else frozenset()


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

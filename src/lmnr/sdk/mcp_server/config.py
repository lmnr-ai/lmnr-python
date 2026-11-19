"""Per-server state of `wrap_mcp_server`: the options plus what it learns."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    should_send_prompts,
)
from lmnr.sdk.mcp_server.turns import TurnTracker

# A tool's advertised input schema by name, or None when unknown.
SchemaLookup = Callable[[str], Awaitable[Mapping[str, Any] | None]]


def _no_suppression() -> AbstractContextManager[Any]:
    return nullcontext()


@dataclass(eq=False)
class McpServerObservability:
    context: bool
    record_payloads: bool
    redact_keys: frozenset[str]
    turns: TurnTracker
    # stamped as `lmnr.span.instrumentation_scope.*`, like the instrumentor's spans
    scope: LaminarInstrumentationScopeAttributes | None = None
    # a framework lookup of a tool's schema; None when the server has none
    lookup_tool_schema: SchemaLookup | None = None
    # turns the framework's own (native) spans off for the block (FastMCP)
    suppress_native: Callable[[], AbstractContextManager[Any]] = _no_suppression
    # tool names seen in `tools/list` that declare their own `context`
    own_context_tools: set[str] = field(default_factory=set[str])
    warned_tools: set[str] = field(default_factory=set[str])
    lock: threading.Lock = field(default_factory=threading.Lock)

    def payloads(self) -> bool:
        """Record arguments, results and context (`LMNR_TRACE_CONTENT=false` also turns it off)."""
        return self.record_payloads and should_send_prompts()

    def context_field(self) -> bool:
        """Inject / strip the `context` argument."""
        return self.context and self.payloads()

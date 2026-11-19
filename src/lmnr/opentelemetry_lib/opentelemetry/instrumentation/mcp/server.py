"""The SERVER span of an MCP request: one code path for mcp 1.x and 2.x, mode on or off.

`v1.py` (`Server._handle_request`) and `v2.py` (the server middleware chain)
read an `McpRequest` off their SDK and drive a `ServerSpan` around the handler.
What differs is the `McpServerSettings` (`Laminar.initialize(mcp=...)`):

- off (the default): every request, parented on the client's `_meta.traceparent`
  (claude_agent nests SDK MCP tool calls under its agent span this way);
- MCP observability mode: `tools/call`, `prompts/get` and `resources/read`
  only, each a ROOT span whose trace id groups one user turn and which LINKS to
  the client's `traceparent` (the trace view drops spans whose parent is missing).

Names, attributes, payloads and error types are the same for both.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Link, Span, SpanKind
from opentelemetry.util.types import AttributeValue
from pydantic import ValidationError

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.context_field import (
    trimmed_context,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.redaction import redact
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.settings import (
    PROMPTS_GET,
    RESOURCES_READ,
    TOOLS_CALL,
    McpRequest,
    McpServerSettings,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.turns import turn_trace_id
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.utils import (
    TOOL_ERROR,
    dump,
    error_code,
    extract_trace_context,
    first_text,
    is_tool_error,
    record_error,
    set_status_code,
    use_span,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import dont_throw
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    stamp_instrumentation_scope,
)
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SESSION_ID,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_TYPE,
)
from lmnr.opentelemetry_lib.tracing.id_generator import can_force_trace_id, forced_trace_id
from lmnr.opentelemetry_lib.tracing.tracer import (
    get_laminar_tracer_provider,
    get_tracer_with_context,
)
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import json_dumps

logger = get_default_logger(__name__)

INVALID_PARAMS_CODE = -32602
INVALID_PARAMS = str(INVALID_PARAMS_CODE)
TOOL_ERROR_MESSAGE = "Tool returned an error"

# How SDKs word an unknown tool or rejected arguments when they fold it into an
# `isError` result instead of a JSON-RPC -32602 error
_INVALID_PARAMS_TEXT = re.compile(
    r"input validation error|validation errors? for|unknown tool|tool\b.{0,200}\bnot found"
    + r"|invalid (?:params|arguments)|-32602",
    re.IGNORECASE | re.DOTALL,
)

_warned_unforceable = False


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _forced_trace_id_supported() -> bool:
    global _warned_unforceable
    if can_force_trace_id(get_laminar_tracer_provider()):
        return True
    if not _warned_unforceable:
        _warned_unforceable = True
        logger.debug(
            "Laminar's tracer provider has no LaminarIdGenerator; MCP calls of one turn "
            + "get separate traces"
        )
    return False


def _start_attributes(
    settings: McpServerSettings, request: McpRequest, context_text: str | None
) -> dict[str, AttributeValue]:
    method = request.method
    target = request.target
    attributes: dict[str, AttributeValue] = {
        "mcp.method.name": method,
        SPAN_TYPE: "TOOL" if method == TOOLS_CALL else "DEFAULT",
    }
    if method == TOOLS_CALL:
        attributes["gen_ai.operation.name"] = "execute_tool"
        if target is not None:
            attributes["gen_ai.tool.name"] = target
    elif method == PROMPTS_GET and target is not None:
        attributes["gen_ai.prompt.name"] = target
    optional: dict[str, str | None] = {
        "jsonrpc.request.id": request.request_id,
        "mcp.protocol.version": request.protocol_version,
        "mcp.session.id": request.session_id,
        f"{ASSOCIATION_PROPERTIES}.{SESSION_ID}": request.session_id,
        "lmnr.mcp.client.name": request.client_name,
        "lmnr.mcp.client.version": request.client_version,
    }
    if request.http:
        optional["network.transport"] = "tcp"
        optional["network.protocol.name"] = "http"
    if settings.payloads():
        if method == RESOURCES_READ:
            optional["mcp.resource.uri"] = _string(request.params.get("uri"))
        optional["lmnr.mcp.context"] = context_text or trimmed_context(
            request.ai_invocation.get("userIntent")
        )
        if method == TOOLS_CALL:
            # lifted into the TOOL span's input by ingestion
            optional["gen_ai.tool.call.arguments"] = json_dumps(
                redact(request.params.get("arguments") or {}, settings.redact_keys)
            )
        elif request.params:
            optional[SPAN_INPUT] = json_dumps(redact(request.params, settings.redact_keys))
    attributes.update({k: v for k, v in optional.items() if v})
    return attributes


class ServerSpan:
    """The span of one request. One that could not start records nothing."""

    def __init__(
        self,
        settings: McpServerSettings,
        request: McpRequest,
        suppress_native: Callable[[], AbstractContextManager[Any]],
    ) -> None:
        self.settings: McpServerSettings = settings
        self.request: McpRequest = request
        # turns the framework's own spans off while the handler runs (FastMCP)
        self.suppress_native: Callable[[], AbstractContextManager[Any]] = suppress_native
        self.span: Span | None = None
        self._tracked_session: str | None = None

    @classmethod
    def start(
        cls,
        to_wrap: WrappedFunctionSpec,
        settings: McpServerSettings,
        request: McpRequest,
        context_text: str | None = None,
        suppress_native: Callable[[], AbstractContextManager[Any]] = nullcontext,
    ) -> ServerSpan:
        # With the mode off the hooks skip requests under OTel's
        # suppress-instrumentation flag; observability mode's root spans ignore
        # it (an in-process caller's flag is not the server's).
        server_span = cls(settings, request, suppress_native)
        if Laminar.is_initialized():
            dont_throw(server_span._start)(to_wrap, context_text)
        return server_span

    def _start(self, to_wrap: WrappedFunctionSpec, context_text: str | None) -> None:
        request = self.request
        name = f"{request.method} {request.target}" if request.target else request.method
        with get_tracer_with_context() as (tracer, current):
            links: list[Link] | None = None
            trace_id: int | None = None
            if self.settings.turns is None:
                # the client's span when it sent one; the server handles each
                # request in a task of its own, so `_meta` is the only carrier
                parent = extract_trace_context(request.meta, current) or current
            else:
                remote = extract_trace_context(request.meta, Context())
                remote_span = trace.get_current_span(remote).get_span_context() if remote else None
                links = [Link(remote_span)] if remote_span is not None else None
                trace_id = self._turn_trace_id(remote_span.trace_id if remote_span else None)
                parent = Context()
            with forced_trace_id(trace_id):
                span = tracer.start_span(
                    name,
                    context=parent,
                    kind=SpanKind.SERVER,
                    attributes=_start_attributes(self.settings, request, context_text),
                    links=links,
                )
        stamp_instrumentation_scope(span, to_wrap)
        self.span = span

    def _turn_trace_id(self, remote_trace_id: int | None) -> int | None:
        # turn id > session turn tracker > client traceparent > random. A
        # traceparent only starts a session's turn, never splits it:
        # OTel-instrumented clients send a fresh one per call.
        turns = self.settings.turns
        session_id = self.request.session_id
        turn_id = _string(self.request.ai_invocation.get("turnId"))
        trace_id = turn_trace_id(session_id, turn_id) if turn_id is not None else None
        if turns is not None and session_id is not None:
            trace_id = turns.begin(session_id, trace_id, remote_trace_id)
            self._tracked_session = session_id
        elif trace_id is None:
            trace_id = remote_trace_id
        if trace_id is not None and not _forced_trace_id_supported():
            return None
        return trace_id

    @contextmanager
    def activate(self) -> Generator[None]:
        """The handler runs inside: its spans nest under this one."""
        if self.span is None:
            yield
            return
        # `use_span` attaches Laminar's context as the global OTel context,
        # which drops values set on the previous one (FastMCP 4's suppression
        # key), so suppress again inside
        with use_span(self.span), self.suppress_native():
            yield

    async def run(self, handler: Awaitable[Any], *, record_result: bool) -> Any:
        """Await `handler` inside the span, record how it went and end the span."""
        try:
            with self.activate():
                result = await handler
        except BaseException as e:
            self.record_exception(e)
            raise
        else:
            if record_result:
                self.record_result(result)
        finally:
            self.end()
        return result

    @dont_throw
    def record_result(self, result: Any) -> None:
        span = self.span
        if span is None:
            return
        if type(result).__name__ == "ErrorData":
            # mcp 1.x answers a failed request with one (`McpError`, or code 0
            # for any other exception a handler raised)
            self._record_error(getattr(result, "code", None), getattr(result, "message", None))
            return
        # as it goes on the wire (1.x wraps results in a `RootModel`)
        result = dump(getattr(result, "root", result))
        payloads = self.settings.payloads()
        if payloads:
            recorded = json_dumps(redact(result, self.settings.redact_keys))
            key = "gen_ai.tool.call.result" if self.request.method == TOOLS_CALL else SPAN_OUTPUT
            span.set_attribute(key, recorded)
        if self.request.method == TOOLS_CALL and is_tool_error(result):
            text = first_text(result)
            error_type = (
                INVALID_PARAMS if text and _INVALID_PARAMS_TEXT.search(text) else TOOL_ERROR
            )
            self._record_failure(error_type, (text if payloads else None) or TOOL_ERROR_MESSAGE)

    @dont_throw
    def record_exception(self, error: BaseException) -> None:
        if self.span is None or not isinstance(error, Exception):
            return
        code = error_code(error)
        if isinstance(code, int):
            # `McpError` (1.x) / `MCPError` (2.x): a JSON-RPC error response
            self._record_error(code, getattr(getattr(error, "error", None), "message", None))
        elif isinstance(error, ValidationError):
            # mcp 2.x validates params inside the middleware chain; the client
            # gets -32602 "Invalid request parameters"
            self._record_error(INVALID_PARAMS_CODE, str(error), error)
        else:
            message = str(error) if self.settings.payloads() else None
            self._record_failure(type(error).__name__, message or type(error).__name__, error)

    def _record_error(self, code: Any, message: Any, error: Exception | None = None) -> None:
        if self.span is None:
            return
        set_status_code(self.span, code)
        text = message if isinstance(message, str) and self.settings.payloads() else None
        self._record_failure(str(code), text or f"JSON-RPC error {code}", error)

    def _record_failure(
        self, error_type: str, message: str, error: Exception | None = None
    ) -> None:
        if self.span is None:
            return
        # without payloads, never the raised exception (its text may be input)
        exception = error if error is not None and self.settings.payloads() else Exception(message)
        record_error(self.span, exception, error_type)

    @dont_throw
    def end(self) -> None:
        if self._tracked_session is not None and self.settings.turns is not None:
            self.settings.turns.end(self._tracked_session)
            self._tracked_session = None
        if self.span is not None:
            self.span.end()

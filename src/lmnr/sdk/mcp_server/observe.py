"""The span of one traced MCP request, independent of the mcp major.

`v1.py` / `v2.py` turn what their SDK hands them into an `McpRequest`, then
drive an `McpCall`: `start` → `activate` around the handler → `record_result`
or `record_exception` → `end`. Every step is guarded: instrumentation never
changes what the handler sees (beyond the stripped `context` argument) or
returns.
"""

from __future__ import annotations

import re
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Link, Span, SpanContext, SpanKind
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.types import AttributeValue
from pydantic import ValidationError

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.utils import (
    extract_trace_context,
    use_span,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import dont_throw
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    set_instrumentation_scope_attributes,
)
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SESSION_ID,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_TYPE,
)
from lmnr.opentelemetry_lib.tracing.context import get_event_attributes_from_context
from lmnr.opentelemetry_lib.tracing.id_generator import can_force_trace_id, forced_trace_id
from lmnr.opentelemetry_lib.tracing.tracer import (
    get_laminar_tracer_provider,
    get_tracer_with_context,
)
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.mcp_server.config import McpServerObservability
from lmnr.sdk.mcp_server.redaction import redact
from lmnr.sdk.mcp_server.turns import turn_trace_id
from lmnr.sdk.utils import json_dumps

logger = get_default_logger(__name__)

TOOLS_CALL = "tools/call"
PROMPTS_GET = "prompts/get"
RESOURCES_READ = "resources/read"
TOOLS_LIST = "tools/list"
TRACED_METHODS = frozenset({TOOLS_CALL, PROMPTS_GET, RESOURCES_READ})

CONTEXT_ARGUMENT = "context"
CONTEXT_DESCRIPTION = (
    "Optional. One sentence on what the user is trying to accomplish overall (their goal, "
    "not this call's arguments). Shared with this server's developer to improve the server; "
    "it does not change the result."
)
MAX_CONTEXT_CHARS = 1000

AI_INVOCATION_META_KEY = "io.modelcontextprotocol/aiInvocation"
OPENAI_SESSION_META_KEY = "openai/session"

TOOL_ERROR = "tool_error"
INVALID_PARAMS = "-32602"
TOOL_ERROR_MESSAGE = "Tool returned an error"

# How SDKs word an unknown tool or rejected arguments when they fold it into an
# `isError` result instead of a JSON-RPC -32602 error
_INVALID_PARAMS_TEXT = re.compile(
    r"input validation error|validation errors? for|unknown tool|tool\b.{0,200}\bnot found"
    + r"|invalid (?:params|arguments)|-32602",
    re.IGNORECASE | re.DOTALL,
)

_warned_unforceable = False


@dataclass
class McpRequest:
    """What a traced request carries, read off either mcp major."""

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


def wire(value: Any) -> Any:
    """A result as it goes on the wire (1.x wraps results in a `RootModel`)."""
    value = getattr(value, "root", value)
    if hasattr(value, "model_dump") and not isinstance(value, type):
        return value.model_dump(by_alias=True, mode="json", exclude_none=True)
    return value


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def trimmed_context(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:MAX_CONTEXT_CHARS] or None


# --- the `context` argument ---------------------------------------------------


def schema_with_context(
    config: McpServerObservability, tool_name: str, schema: Any
) -> Mapping[str, Any] | None:
    """`schema` plus the optional `context` property, or None to leave it alone.

    A tool that already declares `context` keeps its schema untouched and is
    remembered, so its calls keep their `context` argument; a warning is logged
    once per tool.
    """
    if not isinstance(schema, Mapping):
        return None
    schema = cast(Mapping[str, Any], schema)
    properties: Any = schema.get("properties")
    if properties is not None and not isinstance(properties, Mapping):
        return None
    properties = cast(Mapping[str, Any], properties or {})
    with config.lock:
        if CONTEXT_ARGUMENT in properties:
            config.own_context_tools.add(tool_name)
            warn = tool_name not in config.warned_tools
            config.warned_tools.add(tool_name)
        else:
            config.own_context_tools.discard(tool_name)
            warn = False
    if warn:
        logger.warning(
            "MCP tool %r already has a `context` argument; Laminar leaves its schema and "
            + "arguments as they are and does not add its own `context` field to it.",
            tool_name,
        )
    if CONTEXT_ARGUMENT in properties:
        return None
    return {
        **schema,
        "properties": {
            **properties,
            CONTEXT_ARGUMENT: {"type": "string", "description": CONTEXT_DESCRIPTION},
        },
    }


def _tool_with_context(config: McpServerObservability, tool: Any) -> Any:
    if isinstance(tool, Mapping):
        tool_dict = cast(Mapping[str, Any], tool)
        name = tool_dict.get("name")
        if not isinstance(name, str):
            return tool_dict
        schema = schema_with_context(config, name, tool_dict.get("inputSchema"))
        return tool_dict if schema is None else {**tool_dict, "inputSchema": schema}
    model: Any = tool
    name = getattr(model, "name", None)
    fields: Any = getattr(model.__class__, "model_fields", None) or {}
    attr = "input_schema" if "input_schema" in fields else "inputSchema"
    if not isinstance(name, str) or attr not in fields:
        return model
    schema = schema_with_context(config, name, getattr(model, attr, None))
    return model if schema is None else model.model_copy(update={attr: dict(schema)})


def tools_with_context(config: McpServerObservability, result: Any) -> Any:
    """A copy of a `tools/list` result whose tools advertise `context`.

    Handles wire dicts (mcp 2.x middleware), `ListToolsResult` models and
    1.x's `ServerResult` root model. Returns None when it can't, so the caller
    keeps the original. Never raises.
    """
    try:
        return _tools_with_context(config, result)
    except Exception:
        logger.debug("Failed to add `context` to the MCP tools list", exc_info=True)
        return None


def _tools_with_context(config: McpServerObservability, result: Any) -> Any:
    root = getattr(result, "root", None)
    if root is not None and hasattr(result, "model_dump"):
        new_root = _tools_with_context(config, root)
        return None if new_root is None else type(result)(new_root)
    if isinstance(result, Mapping):
        result_dict = cast(Mapping[str, Any], result)
        tools: Any = result_dict.get("tools")
        if not isinstance(tools, list):
            return None
        return {
            **result_dict,
            "tools": [_tool_with_context(config, t) for t in cast(list[Any], tools)],
        }
    tools = getattr(result, "tools", None)
    if not isinstance(tools, list) or not hasattr(result, "model_copy"):
        return None
    return result.model_copy(
        update={"tools": [_tool_with_context(config, t) for t in cast(list[Any], tools)]}
    )


async def declares_own_context(config: McpServerObservability, tool_name: str) -> bool:
    """Whether the tool's real schema has a `context` property.

    The framework's own registry wins (another server copy may have served
    `tools/list`); without one, what this process saw in `tools/list`.
    """
    if config.lookup_tool_schema is not None:
        try:
            schema = await config.lookup_tool_schema(tool_name)
        except Exception:
            logger.debug("Failed to look up the schema of MCP tool %r", tool_name, exc_info=True)
            schema = None
        if schema is not None:
            properties = schema.get("properties")
            return isinstance(properties, Mapping) and CONTEXT_ARGUMENT in properties
    with config.lock:
        return tool_name in config.own_context_tools


async def split_context(
    config: McpServerObservability, request: McpRequest
) -> tuple[dict[str, Any] | None, str | None]:
    """(arguments for the handler, or None to keep them; the `context` text)."""
    if request.method != TOOLS_CALL or not config.context_field():
        return None, None
    arguments: Any = request.params.get("arguments")
    name = request.target
    if not isinstance(arguments, Mapping) or CONTEXT_ARGUMENT not in arguments or name is None:
        return None, None
    if await declares_own_context(config, name):
        return None, None
    arguments = cast(Mapping[str, Any], arguments)
    stripped = {k: v for k, v in arguments.items() if k != CONTEXT_ARGUMENT}
    return stripped, trimmed_context(arguments[CONTEXT_ARGUMENT])


# --- the span -------------------------------------------------------------------


def _remote_parent(meta: Mapping[str, Any]) -> SpanContext | None:
    if "traceparent" not in meta:
        return None
    context = extract_trace_context(meta, Context())
    if context is None:
        return None
    span_context = trace.get_current_span(context).get_span_context()
    return span_context if span_context.is_valid else None


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
    config: McpServerObservability, request: McpRequest, context_text: str | None
) -> dict[str, AttributeValue]:
    method = request.method
    target = request.target
    payloads = config.payloads()
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
    if payloads:
        if method == RESOURCES_READ:
            optional["mcp.resource.uri"] = _string(request.params.get("uri"))
        optional["lmnr.mcp.context"] = context_text or trimmed_context(
            request.ai_invocation.get("userIntent")
        )
        if method == TOOLS_CALL:
            optional["gen_ai.tool.call.arguments"] = json_dumps(
                redact(request.params.get("arguments") or {}, config.redact_keys)
            )
        else:
            optional[SPAN_INPUT] = json_dumps(redact(request.params, config.redact_keys))
    attributes.update({k: v for k, v in optional.items() if v})
    return attributes


class McpCall:
    """One traced request. A call whose span could not start records nothing."""

    def __init__(self, config: McpServerObservability, request: McpRequest) -> None:
        self.config: McpServerObservability = config
        self.request: McpRequest = request
        self.span: Span | None = None
        self._tracked_session: str | None = None

    @classmethod
    def start(
        cls, config: McpServerObservability, request: McpRequest, context_text: str | None
    ) -> McpCall:
        call = cls(config, request)
        dont_throw(call._start)(context_text)
        return call

    def _start(self, context_text: str | None) -> None:
        if not Laminar.is_initialized():
            return
        request = self.request
        session_id = request.session_id
        remote = _remote_parent(request.meta)
        turn_id = _string(request.ai_invocation.get("turnId"))
        # turn id > session turn tracker > client traceparent > random. A
        # traceparent only starts a session's turn, never splits it:
        # OTel-instrumented clients send a fresh one per call. It is always a link.
        trace_id = turn_trace_id(session_id, turn_id) if turn_id is not None else None
        remote_trace_id = remote.trace_id if remote is not None else None
        if session_id is not None:
            trace_id = self.config.turns.begin(session_id, trace_id, remote_trace_id)
            self._tracked_session = session_id
        elif trace_id is None:
            trace_id = remote_trace_id
        if trace_id is not None and not _forced_trace_id_supported():
            trace_id = None
        name = f"{request.method} {request.target}" if request.target else request.method
        with get_tracer_with_context() as (tracer, _), forced_trace_id(trace_id):
            # An empty context: always a root span, linked to (never parented
            # on) a client span, so the trace view renders it.
            span = tracer.start_span(
                name,
                context=Context(),
                kind=SpanKind.SERVER,
                attributes=_start_attributes(self.config, request, context_text),
                links=[Link(remote)] if remote is not None else None,
            )
        set_instrumentation_scope_attributes(span, self.config.scope)
        self.span = span

    @contextmanager
    def activate(self) -> Generator[None]:
        """The handler runs inside: its spans nest under this one."""
        if self.span is None:
            yield
            return
        # `use_span` attaches Laminar's context as the global OTel context,
        # which drops values set on the previous one (FastMCP 4's suppression
        # key), so suppress again inside
        with use_span(self.span), self.config.suppress_native():
            yield

    @dont_throw
    def record_result(self, result: Any) -> None:
        span = self.span
        if span is None:
            return
        result = wire(result)
        if type(result).__name__ == "ErrorData":
            self._record_error(str(getattr(result, "code", "")), getattr(result, "message", None))
            return
        payloads = self.config.payloads()
        if payloads:
            recorded = json_dumps(redact(result, self.config.redact_keys))
            key = "gen_ai.tool.call.result" if self.request.method == TOOLS_CALL else SPAN_OUTPUT
            span.set_attribute(key, recorded)
        if self.request.method == TOOLS_CALL and _is_error(result):
            text = _first_text(result)
            error_type = (
                INVALID_PARAMS if text and _INVALID_PARAMS_TEXT.search(text) else TOOL_ERROR
            )
            self._record_failure(error_type, (text if payloads else None) or TOOL_ERROR_MESSAGE)

    @dont_throw
    def record_exception(self, error: BaseException) -> None:
        if self.span is None or not isinstance(error, Exception):
            return
        code = getattr(getattr(error, "error", None), "code", None)
        if isinstance(code, int):
            # `McpError` (1.x) / `MCPError` (2.x): a JSON-RPC error response
            self._record_error(str(code), getattr(getattr(error, "error", None), "message", None))
        elif isinstance(error, ValidationError):
            # mcp 2.x validates params inside the middleware chain; the client
            # gets -32602 "Invalid request parameters"
            self._record_error(INVALID_PARAMS, str(error), error)
        else:
            message = str(error) if self.config.payloads() else None
            self._record_failure(type(error).__name__, message or type(error).__name__, error)

    def _record_error(self, code: str, message: Any, error: Exception | None = None) -> None:
        span = self.span
        if span is None:
            return
        span.set_attribute("rpc.response.status_code", code)
        text = message if isinstance(message, str) and self.config.payloads() else None
        self._record_failure(code, text or f"JSON-RPC error {code}", error)

    def _record_failure(
        self, error_type: str, message: str, error: Exception | None = None
    ) -> None:
        span = self.span
        if span is None:
            return
        span.set_attribute("error.type", error_type)
        # the `exception` event (not the status) marks the span failed in Laminar;
        # without payloads, never the raised exception (its text may be input)
        exception = error if error is not None and self.config.payloads() else Exception(message)
        span.record_exception(
            exception,
            attributes={
                **get_event_attributes_from_context(),
                "exception.type": error_type,
                "exception.message": message,
            },
        )
        span.set_status(Status(StatusCode.ERROR, message))

    @dont_throw
    def end(self) -> None:
        if self._tracked_session is not None:
            self.config.turns.end(self._tracked_session)
            self._tracked_session = None
        if self.span is not None:
            self.span.end()


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return cast(Mapping[str, Any], obj).get(key)
    return getattr(obj, key, None)


def _is_error(result: Any) -> bool:
    return _get(result, "isError") is True or _get(result, "is_error") is True


def _first_text(result: Any) -> str | None:
    content: Any = _get(result, "content") or []
    if not isinstance(content, Sequence):
        return None
    for block in cast(Sequence[Any], content):
        text = _get(block, "text")
        if isinstance(text, str):
            return text
    return None

"""The optional `context` argument tools advertise in MCP observability mode.

Injected into `tools/list` results and stripped from `tools/call` arguments
before the SDK validates them (FastMCP rejects unexpected arguments). A tool
that declares its own `context` is left alone.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Mapping
from typing import Any, cast

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.settings import (
    TOOLS_CALL,
    McpRequest,
    Mode,
    context_field,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

CONTEXT_ARGUMENT = "context"
CONTEXT_DESCRIPTION = (
    "Optional. One sentence on what the user is trying to accomplish overall (their goal, "
    "not this call's arguments). Shared with this server's developer to improve the server; "
    "it does not change the result."
)
MAX_CONTEXT_CHARS = 1000

# Tool names seen in `tools/list` that declare their own `context`, across the
# process (another server copy may serve the call).
_own_context_tools: set[str] = set()
_warned_tools: set[str] = set()
_lock = threading.Lock()


def trimmed_context(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:MAX_CONTEXT_CHARS] or None


def _schema_with_context(tool_name: str, schema: Any) -> Mapping[str, Any] | None:
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
    own = CONTEXT_ARGUMENT in properties
    with _lock:
        if own:
            _own_context_tools.add(tool_name)
            warn = tool_name not in _warned_tools
            _warned_tools.add(tool_name)
        else:
            _own_context_tools.discard(tool_name)
            warn = False
    if warn:
        logger.warning(
            "MCP tool %r already has a `context` argument; Laminar leaves its schema and "
            + "arguments as they are and does not add its own `context` field to it.",
            tool_name,
        )
    if own:
        return None
    return {
        **schema,
        "properties": {
            **properties,
            CONTEXT_ARGUMENT: {"type": "string", "description": CONTEXT_DESCRIPTION},
        },
    }


def _tool_with_context(tool: Any) -> Any:
    if isinstance(tool, Mapping):
        tool_dict = cast(Mapping[str, Any], tool)
        name = tool_dict.get("name")
        if not isinstance(name, str):
            return tool_dict
        schema = _schema_with_context(name, tool_dict.get("inputSchema"))
        return tool_dict if schema is None else {**tool_dict, "inputSchema": schema}
    model: Any = tool
    name = getattr(model, "name", None)
    fields: Any = getattr(model.__class__, "model_fields", None) or {}
    attr = "input_schema" if "input_schema" in fields else "inputSchema"
    if not isinstance(name, str) or attr not in fields:
        return model
    schema = _schema_with_context(name, getattr(model, attr, None))
    return model if schema is None else model.model_copy(update={attr: dict(schema)})


def tools_with_context(mode: Mode, result: Any) -> Any:
    """A copy of a `tools/list` result whose tools advertise `context`.

    Handles wire dicts (mcp 2.x middleware), `ListToolsResult` models and
    1.x's `ServerResult` root model. Returns `result` itself when the field is
    off or the shape is unknown. Never raises.
    """
    if not context_field(mode):
        return result
    try:
        return _tools_with_context(result)
    except Exception:
        logger.debug("Failed to add `context` to the MCP tools list", exc_info=True)
        return result


def _tools_with_context(result: Any) -> Any:
    root = getattr(result, "root", None)
    if root is not None and hasattr(result, "model_dump"):
        new_root = _tools_with_context(root)
        return result if new_root is root else type(result)(new_root)
    if isinstance(result, Mapping):
        result_dict = cast(Mapping[str, Any], result)
        tools: Any = result_dict.get("tools")
        if not isinstance(tools, list):
            return result
        return {
            **result_dict,
            "tools": [_tool_with_context(t) for t in cast(list[Any], tools)],
        }
    tools = getattr(result, "tools", None)
    if not isinstance(tools, list) or not hasattr(result, "model_copy"):
        return result
    return result.model_copy(
        update={"tools": [_tool_with_context(t) for t in cast(list[Any], tools)]}
    )


async def _registered_schema(server: Any, tool_name: str) -> Any:
    """The tool's input schema from the low-level `server`'s registry, or None."""
    # mcp 2.x `Server` (set by `MCPServer` and FastMCP 4)
    schema_of: Any = getattr(server, "get_tool_input_schema", None)
    if callable(schema_of):
        schema: Any = schema_of(tool_name)
        return await cast(Awaitable[Any], schema) if isinstance(schema, Awaitable) else schema
    # mcp 1.x `Server`: its `tools/list` cache, refreshed from the list handler when missing
    cached: Any = getattr(server, "_get_cached_tool_definition", None)
    if callable(cached):
        return getattr(await cast(Any, cached)(tool_name), "inputSchema", None)
    return None


async def _declares_own_context(server: Any, tool_name: str) -> bool:
    """Whether the tool's real schema has a `context` property.

    The server's own registry wins (another server copy may have served
    `tools/list`); without one, what this process saw in `tools/list`.
    """
    try:
        schema = await _registered_schema(server, tool_name)
    except Exception:
        logger.debug("Failed to look up the schema of MCP tool %r", tool_name, exc_info=True)
        schema = None
    if isinstance(schema, Mapping):
        properties = cast(Mapping[str, Any], schema).get("properties")
        return isinstance(properties, Mapping) and CONTEXT_ARGUMENT in properties
    with _lock:
        return tool_name in _own_context_tools


async def split_context(
    mode: Mode, request: McpRequest, server: Any
) -> tuple[dict[str, Any] | None, str | None]:
    """(arguments for the handler, or None to keep them; the `context` text). Never raises."""
    if request.method != TOOLS_CALL or not context_field(mode):
        return None, None
    arguments: Any = request.params.get("arguments")
    name = request.target
    if not isinstance(arguments, Mapping) or CONTEXT_ARGUMENT not in arguments or name is None:
        return None, None
    if await _declares_own_context(server, name):
        return None, None
    arguments = cast(Mapping[str, Any], arguments)
    stripped = {k: v for k, v in arguments.items() if k != CONTEXT_ARGUMENT}
    return stripped, trimmed_context(arguments[CONTEXT_ARGUMENT])

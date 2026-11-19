"""The optional `context` argument a wrapped server's tools advertise.

Injected into `tools/list` results and stripped from `tools/call` arguments
before the SDK validates them (FastMCP rejects unexpected arguments). A tool
that declares its own `context` is left alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.settings import (
    TOOLS_CALL,
    McpRequest,
    McpServerSettings,
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


def trimmed_context(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:MAX_CONTEXT_CHARS] or None


def _schema_with_context(
    settings: McpServerSettings, tool_name: str, schema: Any
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
    own = CONTEXT_ARGUMENT in properties
    with settings.lock:
        if own:
            settings.own_context_tools.add(tool_name)
            warn = tool_name not in settings.warned_tools
            settings.warned_tools.add(tool_name)
        else:
            settings.own_context_tools.discard(tool_name)
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


def _tool_with_context(settings: McpServerSettings, tool: Any) -> Any:
    if isinstance(tool, Mapping):
        tool_dict = cast(Mapping[str, Any], tool)
        name = tool_dict.get("name")
        if not isinstance(name, str):
            return tool_dict
        schema = _schema_with_context(settings, name, tool_dict.get("inputSchema"))
        return tool_dict if schema is None else {**tool_dict, "inputSchema": schema}
    model: Any = tool
    name = getattr(model, "name", None)
    fields: Any = getattr(model.__class__, "model_fields", None) or {}
    attr = "input_schema" if "input_schema" in fields else "inputSchema"
    if not isinstance(name, str) or attr not in fields:
        return model
    schema = _schema_with_context(settings, name, getattr(model, attr, None))
    return model if schema is None else model.model_copy(update={attr: dict(schema)})


def tools_with_context(settings: McpServerSettings, result: Any) -> Any:
    """A copy of a `tools/list` result whose tools advertise `context`.

    Handles wire dicts (mcp 2.x middleware), `ListToolsResult` models and
    1.x's `ServerResult` root model. Returns `result` itself when the field is
    off or the shape is unknown. Never raises.
    """
    if not settings.context_field():
        return result
    try:
        return _tools_with_context(settings, result)
    except Exception:
        logger.debug("Failed to add `context` to the MCP tools list", exc_info=True)
        return result


def _tools_with_context(settings: McpServerSettings, result: Any) -> Any:
    root = getattr(result, "root", None)
    if root is not None and hasattr(result, "model_dump"):
        new_root = _tools_with_context(settings, root)
        return result if new_root is root else type(result)(new_root)
    if isinstance(result, Mapping):
        result_dict = cast(Mapping[str, Any], result)
        tools: Any = result_dict.get("tools")
        if not isinstance(tools, list):
            return result
        return {
            **result_dict,
            "tools": [_tool_with_context(settings, t) for t in cast(list[Any], tools)],
        }
    tools = getattr(result, "tools", None)
    if not isinstance(tools, list) or not hasattr(result, "model_copy"):
        return result
    return result.model_copy(
        update={"tools": [_tool_with_context(settings, t) for t in cast(list[Any], tools)]}
    )


async def _declares_own_context(settings: McpServerSettings, tool_name: str) -> bool:
    """Whether the tool's real schema has a `context` property.

    The framework's own registry wins (another server copy may have served
    `tools/list`); without one, what this process saw in `tools/list`.
    """
    if settings.lookup_tool_schema is not None:
        try:
            schema = await settings.lookup_tool_schema(tool_name)
        except Exception:
            logger.debug("Failed to look up the schema of MCP tool %r", tool_name, exc_info=True)
            schema = None
        if schema is not None:
            properties = schema.get("properties")
            return isinstance(properties, Mapping) and CONTEXT_ARGUMENT in properties
    with settings.lock:
        return tool_name in settings.own_context_tools


async def split_context(
    settings: McpServerSettings, request: McpRequest
) -> tuple[dict[str, Any] | None, str | None]:
    """(arguments for the handler, or None to keep them; the `context` text). Never raises."""
    if request.method != TOOLS_CALL or not settings.context_field():
        return None, None
    arguments: Any = request.params.get("arguments")
    name = request.target
    if not isinstance(arguments, Mapping) or CONTEXT_ARGUMENT not in arguments or name is None:
        return None, None
    if await _declares_own_context(settings, name):
        return None, None
    arguments = cast(Mapping[str, Any], arguments)
    stripped = {k: v for k, v in arguments.items() if k != CONTEXT_ARGUMENT}
    return stripped, trimmed_context(arguments[CONTEXT_ARGUMENT])

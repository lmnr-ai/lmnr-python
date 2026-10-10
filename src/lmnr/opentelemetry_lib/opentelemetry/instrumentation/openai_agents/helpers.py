"""Span naming, type mapping, and utility helpers for OpenAI Agents instrumentation."""

import contextvars
from typing import Any

from agents.tracing import Span
from opentelemetry.context import create_key

from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import LaminarSpanType

logger = get_default_logger(__name__)


DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY_RAW = (
    "LMNR_DISABLE_OPENAI_RESPONSES_INSTRUMENTATION"
)
DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY = create_key(
    DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY_RAW
)


# Task-local system instructions for the currently-executing model call.
# Set by the wrapped get_response / stream_response methods in instrumentor.py
# and read by the response/generation span_data handlers so the system prompt
# can be prepended to gen_ai.input.messages.
_current_system_instructions: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("lmnr_openai_agents_system_instructions", default=None)
)


def get_current_system_instructions() -> str | None:
    return _current_system_instructions.get()


def set_current_system_instructions(
    value: str | None,
) -> "contextvars.Token[str | None]":
    return _current_system_instructions.set(value)


def reset_current_system_instructions(
    token: "contextvars.Token[str | None]",
) -> None:
    _current_system_instructions.reset(token)


def span_name(span: Span[Any], span_data: Any) -> str:
    name = getattr(span, "name", None)
    if name:
        return name
    kind = span_kind(span_data)
    if kind:
        if kind in ["agent", "custom", "function", "tool"]:
            return name_from_span_data(span_data) or f"agents.{kind}"
        return f"agents.{kind}"
    return "agents.span"


def span_kind(span_data: Any) -> str:
    if span_data is None:
        return ""
    return getattr(span_data, "type", "")


def map_span_type(span_data: Any) -> LaminarSpanType:
    kind = span_kind(span_data)
    if kind in {"generation", "response", "transcription", "speech", "speech_group"}:
        return "LLM"
    if kind in {"function", "tool", "mcp_list_tools", "mcp_tools", "handoff"}:
        return "TOOL"
    return "DEFAULT"


def export_span_data(span_data: Any) -> dict[str, Any]:
    if span_data is None:
        return {}
    if hasattr(span_data, "export"):
        try:
            exported = span_data.export()
            if isinstance(exported, dict):
                return exported  # pyright: ignore[reportUnknownVariableType]
        except Exception:
            return {}
    return {}


def normalize_messages(data: Any, role: str = "user") -> list[dict[str, Any]]:
    """Normalize various input/output formats into a list of message dicts."""
    if data is None:
        return []

    if isinstance(data, str):
        return [{"role": role, "content": data}]

    if isinstance(data, list):
        messages = []
        for item in data:  # pyright: ignore[reportUnknownVariableType]
            if isinstance(item, dict):
                messages.append(item)  # pyright: ignore[reportUnknownMemberType]
            elif hasattr(item, "model_dump"):  # pyright: ignore[reportUnknownArgumentType]
                try:
                    messages.append(item.model_dump())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                except Exception:
                    messages.append({"content": str(item)})  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            else:
                item_dict = model_as_dict(item)
                if item_dict:
                    messages.append(item_dict)  # pyright: ignore[reportUnknownMemberType]
                else:
                    messages.append({"content": str(item)})  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        return messages  # pyright: ignore[reportUnknownVariableType]

    if isinstance(data, dict):
        return [data]

    # If it's a pydantic model or similar
    as_dict = model_as_dict(data)
    if as_dict:
        return [as_dict]

    return [{"content": str(data)}]


def model_as_dict(obj: Any) -> dict[str, Any] | None:
    """Convert a pydantic model or similar object to a dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj  # pyright: ignore[reportUnknownVariableType]
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            logger.debug("failed to dump openai agents model", exc_info=True)
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            logger.debug("failed to dump openai agents model", exc_info=True)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return None


def name_from_span_data(agent: Any) -> str:
    if isinstance(agent, dict):
        return agent.get("name") or "" ## pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if isinstance(agent, str):
        return agent
    if hasattr(agent, "name"):
        return getattr(agent, "name", "") or ""
    return ""


def get_first_not_none(d: dict[str, Any], *keys: str) -> Any:
    """Get the first key whose value is not None from a dict."""
    for key in keys:
        val = d.get(key)
        if val is not None:
            return val
    return None


def get_attr_not_none(obj: Any, *attrs: str) -> Any:
    """Get the first attribute whose value is not None from an object."""
    for attr in attrs:
        val = getattr(obj, attr, None)
        if val is not None:
            return val
    return None



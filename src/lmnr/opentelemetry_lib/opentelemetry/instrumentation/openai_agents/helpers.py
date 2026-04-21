"""Span naming, type mapping, and utility helpers for OpenAI Agents instrumentation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmnr.sdk.types import LaminarSpanType


def span_name(span: Any, span_data: Any) -> str:
    name = getattr(span, "name", None)
    if name:
        return name
    kind = span_kind(span_data)
    if kind:
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
    if kind in {"function", "tool", "mcp_list_tools", "mcp_tools"}:
        return "TOOL"
    return "DEFAULT"


def export_span_data(span_data: Any) -> dict[str, Any]:
    if span_data is None:
        return {}
    if hasattr(span_data, "export"):
        try:
            exported = span_data.export()
            if isinstance(exported, dict):
                return exported
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
        for item in data:
            if isinstance(item, dict):
                messages.append(item)
            elif hasattr(item, "model_dump"):
                try:
                    messages.append(item.model_dump())
                except Exception:
                    messages.append({"content": str(item)})
            else:
                item_dict = model_as_dict(item)
                if item_dict:
                    messages.append(item_dict)
                else:
                    messages.append({"content": str(item)})
        return messages

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
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return None


def agent_name(agent: Any) -> str:
    if isinstance(agent, dict):
        return agent.get("name") or ""
    if isinstance(agent, str):
        return agent
    if hasattr(agent, "name"):
        return getattr(agent, "name") or ""
    return ""


def get_first_not_none(d: dict, *keys: str) -> Any:
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

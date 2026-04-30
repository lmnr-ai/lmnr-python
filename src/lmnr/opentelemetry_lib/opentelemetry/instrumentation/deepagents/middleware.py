"""LaminarMiddleware - an `AgentMiddleware` that emits Laminar TOOL spans."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger

try:
    from langchain.agents.middleware.types import AgentMiddleware
except ImportError:  # pragma: no cover - guarded by the instrumentor
    AgentMiddleware = object  # type: ignore[assignment,misc]

logger = get_default_logger(__name__)


def _summarize_messages(messages: Any) -> Any:
    """Extract role + content pairs from langchain/langgraph message objects."""
    if not isinstance(messages, list):
        return messages
    out = []
    for m in messages:
        role = getattr(m, "type", None) or getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role is None and isinstance(m, dict):
            role = m.get("role") or m.get("type")
            content = m.get("content", content)
        out.append({"role": role, "content": content} if role is not None else m)
    return out


def _tool_result_to_json(result: Any) -> Any:
    """Return a JSON-friendly view of a ToolMessage / Command."""
    if hasattr(result, "content"):
        return getattr(result, "content", None)
    # A langgraph `Command` exposes `.update` as a data attribute, but
    # `dict` (and other mapping types) expose it as a callable method —
    # skip those to avoid serializing the bound method as the tool output.
    update_attr = getattr(result, "update", None)
    if update_attr is not None and not callable(update_attr):
        try:
            return {"update": update_attr}
        except Exception:
            return repr(result)
    return result


def _tool_span_name(request: Any) -> str:
    call = getattr(request, "tool_call", None) or {}
    if isinstance(call, dict):
        name = call.get("name")
        if name:
            return f"{name}"
    tool = getattr(request, "tool", None)
    return getattr(tool, "name", None) or "tool"


def _tool_span_input(request: Any) -> Any:
    call = getattr(request, "tool_call", None) or {}
    if isinstance(call, dict):
        return call.get("args")
    return None


class LaminarMiddleware(AgentMiddleware):  # type: ignore[misc,valid-type]
    """Emits Laminar TOOL spans around every agent tool call.

    Injected automatically by `DeepagentsInstrumentor` into every agent
    built via `deepagents.create_deep_agent`. Safe to add manually — it's a
    no-op when Laminar isn't initialised (`Laminar.start_as_current_span`
    yields a non-recording span when `Laminar.initialize` hasn't run).

    The matching DEFAULT root span that parents these tool spans is opened
    by `DeepagentsInstrumentor` around the compiled graph's
    `invoke`/`stream`, not by `before_agent`/`after_agent` hooks:
    LangGraph runs middleware hooks as separate graph nodes, so OTel
    context attached in `before_agent` doesn't survive into later tool
    nodes. Wrapping `invoke` instead keeps the root span's context active
    for the entire graph execution.
    """

    def wrap_tool_call(  # type: ignore[override]
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        with Laminar.start_as_current_span(
            name=_tool_span_name(request),
            input=_tool_span_input(request),
            span_type="TOOL",
        ) as span:
            result = handler(request)
            span.set_output(_tool_result_to_json(result))
            return result

    async def awrap_tool_call(  # type: ignore[override]
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        with Laminar.start_as_current_span(
            name=_tool_span_name(request),
            input=_tool_span_input(request),
            span_type="TOOL",
        ) as span:
            result = await handler(request)
            span.set_output(_tool_result_to_json(result))
            return result

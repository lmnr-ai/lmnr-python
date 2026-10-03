"""LaminarMiddleware - an `AgentMiddleware` that emits Laminar TOOL spans."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from typing_extensions import TypeVar, override

from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

T = TypeVar("T")


def summarize_messages(messages: Any) -> Any:  # pyright: ignore[reportAny, reportExplicitAny]
    """Extract role + content pairs from langchain/langgraph message objects."""
    if not isinstance(messages, list):
        return messages  # pyright: ignore[reportAny]
    out = []
    for m in messages:  # pyright: ignore[reportUnknownVariableType]
        role = getattr(m, "type", None) or getattr(m, "role", None)  # pyright: ignore[reportUnknownArgumentType]
        content = getattr(m, "content", None)  # pyright: ignore[reportUnknownArgumentType]
        if role is None and isinstance(m, dict):
            role = m.get("role") or m.get("type")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            content = m.get("content", content)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        out.append({"role": role, "content": content} if role is not None else m)  # pyright: ignore[reportUnknownMemberType]
    return out  # pyright: ignore[reportUnknownVariableType]


def _tool_result_to_json(result: Any) -> Any:  # pyright: ignore[reportAny, reportExplicitAny]:
    """Return a JSON-friendly view of a ToolMessage / Command."""
    if hasattr(result, "content"):  # pyright: ignore[reportAny]
        return getattr(result, "content", None)  # pyright: ignore[reportAny]
    # A langgraph `Command` exposes `.update` as a data attribute, but
    # `dict` (and other mapping types) expose it as a callable method —
    # skip those to avoid serializing the bound method as the tool output.
    update_attr = getattr(result, "update", None)  # pyright: ignore[reportAny]
    if update_attr is not None and not callable(update_attr):  # pyright: ignore[reportAny]
        try:
            return {"update": update_attr}
        except Exception:
            return repr(result)  # pyright: ignore[reportAny]
    return result  # pyright: ignore[reportAny]


def _tool_span_name(request: Any) -> str:  # pyright: ignore[reportAny, reportExplicitAny]::
    call = getattr(request, "tool_call", None) or {}  # pyright: ignore[reportAny, reportUnknownVariableType]
    if isinstance(call, dict):
        name = call.get("name")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if name:
            return f"{name}"
    tool = getattr(request, "tool", None)  # pyright: ignore[reportAny]
    return getattr(tool, "name", None) or "tool"


def _tool_span_input(request: Any) -> Any:  # pyright: ignore[reportAny, reportExplicitAny]:
    call = getattr(request, "tool_call", None) or {}  # pyright: ignore[reportAny, reportUnknownVariableType]
    if isinstance(call, dict):
        return call.get("args")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return None

try:
    from langchain.agents.middleware.types import AgentMiddleware
    class LaminarMiddleware(AgentMiddleware):  # pyright: ignore[reportRedeclaration]
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

        @override
        def wrap_tool_call(
            self,
            request: Any,  # pyright: ignore[reportAny, reportExplicitAny]
            handler: Callable[..., T],
        ) -> T:
            with Laminar.start_as_current_span(
                name=_tool_span_name(request),
                input=_tool_span_input(request),
                span_type="TOOL",
            ) as span:
                result = handler(request)
                span.set_output(_tool_result_to_json(result))
                return result

        @override
        async def awrap_tool_call(
            self,
            request: Any,  # pyright: ignore[reportAny, reportExplicitAny]
            handler: Callable[..., Awaitable[T]],
        ) -> T:
            with Laminar.start_as_current_span(
                name=_tool_span_name(request),
                input=_tool_span_input(request),
                span_type="TOOL",
            ) as span:
                result = await handler(request)
                span.set_output(_tool_result_to_json(result))
                return result

except ImportError:
    logger.debug("failed to import DeepAgents", exc_info=True)
    class LaminarMiddleware:
        pass

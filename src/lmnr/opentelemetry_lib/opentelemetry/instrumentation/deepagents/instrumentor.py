"""DeepagentsInstrumentor - wraps deep agents with a Laminar root span.

Strategy:

1. Monkey-patch `create_deep_agent` to inject a `LaminarMiddleware` into the
   user-middleware slot. That middleware implements `wrap_tool_call` /
   `awrap_tool_call`, so every tool invocation — including the `task` tool
   that spawns a subagent — runs inside a Laminar TOOL span.

2. Wrap the returned compiled graph's `invoke` / `ainvoke` / `stream` /
   `astream` methods so each top-level call runs inside a single Laminar
   DEFAULT root span. Doing it here (instead of via `before_agent` /
   `after_agent` middleware hooks) is important: LangGraph runs each
   middleware node as its own task, so OTel context attached in
   `before_agent` is popped before the next node starts and the root span
   fails to become the parent of subsequent tool spans. Wrapping
   `invoke`/`ainvoke` keeps the span context active for the entire
   execution of the graph.

Subagent spans (and their nested tool / LLM spans) automatically nest
under the `task` tool span via OTel context propagation, so the frontend's
existing `lmnr.span.prompt_hash` / `lmnr.span.ids_path` fingerprinting
picks up subagent boundaries without any extra span machinery.
"""

from __future__ import annotations

import contextvars
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from lmnr.sdk.log import get_default_logger

from .middleware import LaminarMiddleware, _SpanHandle, _summarize_messages

logger = get_default_logger(__name__)

_instruments = ("deepagents >= 0.5.0",)

_ROOT_SPAN_NAME = "deep_agent"

# Tracks whether a deep_agent root span is already active on this task.
# `Pregel.invoke` internally calls `self.stream`, so wrapping both would
# open two spans per top-level call; the guard collapses those into one.
_root_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "lmnr_deepagents_root_active", default=False
)


def _extract_messages(input_payload: Any) -> Any:
    if isinstance(input_payload, dict):
        return input_payload.get("messages")
    return None


def _set_output_from_result(handle: _SpanHandle, result: Any) -> None:
    out_messages = _extract_messages(result)
    if not out_messages:
        return
    last = out_messages[-1] if isinstance(out_messages, list) else None
    content = getattr(last, "content", None) if last is not None else None
    if content is None and isinstance(last, dict):
        content = last.get("content")
    handle.set_output(content if content is not None else _summarize_messages(out_messages))


def _wrap_graph_invoke(wrapped, instance, args, kwargs):
    if _root_active.get():
        return wrapped(*args, **kwargs)
    input_payload = args[0] if args else kwargs.get("input")
    with _SpanHandle(_ROOT_SPAN_NAME, "DEFAULT") as handle:
        messages = _extract_messages(input_payload)
        if messages is not None:
            handle.set_input({"messages": _summarize_messages(messages)})
        token = _root_active.set(True)
        try:
            try:
                result = wrapped(*args, **kwargs)
            except BaseException as exc:
                handle.record_exception(exc)
                raise
            _set_output_from_result(handle, result)
            return result
        finally:
            _root_active.reset(token)


async def _awrap_graph_invoke(wrapped, instance, args, kwargs):
    if _root_active.get():
        return await wrapped(*args, **kwargs)
    input_payload = args[0] if args else kwargs.get("input")
    with _SpanHandle(_ROOT_SPAN_NAME, "DEFAULT") as handle:
        messages = _extract_messages(input_payload)
        if messages is not None:
            handle.set_input({"messages": _summarize_messages(messages)})
        token = _root_active.set(True)
        try:
            try:
                result = await wrapped(*args, **kwargs)
            except BaseException as exc:
                handle.record_exception(exc)
                raise
            _set_output_from_result(handle, result)
            return result
        finally:
            _root_active.reset(token)


def _wrap_graph_stream(wrapped, instance, args, kwargs):
    if _root_active.get():
        return wrapped(*args, **kwargs)
    input_payload = args[0] if args else kwargs.get("input")

    def _gen():
        with _SpanHandle(_ROOT_SPAN_NAME, "DEFAULT") as handle:
            messages = _extract_messages(input_payload)
            if messages is not None:
                handle.set_input({"messages": _summarize_messages(messages)})
            token = _root_active.set(True)
            try:
                try:
                    for chunk in wrapped(*args, **kwargs):
                        yield chunk
                except BaseException as exc:
                    handle.record_exception(exc)
                    raise
            finally:
                _root_active.reset(token)

    return _gen()


def _awrap_graph_stream(wrapped, instance, args, kwargs):
    if _root_active.get():
        return wrapped(*args, **kwargs)
    input_payload = args[0] if args else kwargs.get("input")

    async def _gen():
        with _SpanHandle(_ROOT_SPAN_NAME, "DEFAULT") as handle:
            messages = _extract_messages(input_payload)
            if messages is not None:
                handle.set_input({"messages": _summarize_messages(messages)})
            token = _root_active.set(True)
            try:
                try:
                    async for chunk in wrapped(*args, **kwargs):
                        yield chunk
                except BaseException as exc:
                    handle.record_exception(exc)
                    raise
            finally:
                _root_active.reset(token)

    return _gen()


_INSTRUMENTED_GRAPH_FLAG = "_lmnr_deepagents_instrumented"


def _wrap_graph_methods(graph: Any) -> None:
    if getattr(graph, _INSTRUMENTED_GRAPH_FLAG, False):
        return
    cls = graph.__class__
    # Wrap on the instance rather than the class so we don't affect every
    # other Pregel graph in the process (LangGraph is also used by plain
    # LangChain agents, and we don't want to double-wrap those).
    if hasattr(cls, "invoke"):
        wrap_function_wrapper(graph, "invoke", _wrap_graph_invoke)
    if hasattr(cls, "ainvoke"):
        wrap_function_wrapper(graph, "ainvoke", _awrap_graph_invoke)
    if hasattr(cls, "stream"):
        wrap_function_wrapper(graph, "stream", _wrap_graph_stream)
    if hasattr(cls, "astream"):
        wrap_function_wrapper(graph, "astream", _awrap_graph_stream)
    try:
        setattr(graph, _INSTRUMENTED_GRAPH_FLAG, True)
    except Exception:
        pass


def _inject_middleware(wrapped, instance, args, kwargs):
    existing = kwargs.get("middleware") or ()
    if not any(isinstance(m, LaminarMiddleware) for m in existing):
        kwargs["middleware"] = (LaminarMiddleware(), *existing)
    graph = wrapped(*args, **kwargs)
    try:
        _wrap_graph_methods(graph)
    except Exception:
        logger.debug("Failed to wrap deep-agent graph invoke/stream", exc_info=True)
    return graph


_WRAPPED_TARGETS: tuple[tuple[str, str, Any], ...] = (
    ("deepagents.graph", "create_deep_agent", _inject_middleware),
    ("deepagents", "create_deep_agent", _inject_middleware),
)


class DeepagentsInstrumentor(BaseInstrumentor):
    """Instrumentor for langchain-ai/deepagents.

    Emits two kinds of Laminar spans on top of whatever the Anthropic /
    OpenAI / LangChain instrumentors already produce:

    - one `DEFAULT` root span (`deep_agent`) per top-level `invoke`/`stream`
      call on a compiled deep agent, and
    - one `TOOL` span per tool call (including the `task` subagent tool,
      which becomes the natural parent of the subagent's spans via OTel
      context propagation).
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        for module, name, wrapper in _WRAPPED_TARGETS:
            try:
                wrap_function_wrapper(module, name, wrapper)
            except (AttributeError, ModuleNotFoundError, ImportError):
                logger.debug("Failed to wrap %s.%s", module, name)

    def _uninstrument(self, **kwargs):
        for module, name, _ in _WRAPPED_TARGETS:
            try:
                mod = __import__(module, fromlist=[name])
                unwrap(mod, name)
            except Exception:
                logger.debug("Failed to unwrap %s.%s", module, name)

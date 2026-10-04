"""OpenAIAgentsInstrumentor - BaseInstrumentor for OpenAI Agents SDK."""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Collection, Sequence
from typing import Any, cast

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from typing_extensions import TypeVar, override
from wrapt import wrap_function_wrapper

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.helpers import (
    reset_current_system_instructions,
    set_current_system_instructions,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.processor import (
    LaminarAgentsTraceProcessor,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

instruments = ("openai-agents >= 0.7.0",)

T = TypeVar("T")


def _extract_system_instructions(
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    if "system_instructions" in kwargs:
        return kwargs["system_instructions"]
    if args:
        return args[0]
    return None


async def _wrap_get_response(
    wrapped: Callable[..., Awaitable[T]],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T:
    token = set_current_system_instructions(_extract_system_instructions(args, kwargs))
    try:
        return await wrapped(*args, **kwargs)
    finally:
        reset_current_system_instructions(token)


def _wrap_stream_response(
    wrapped: Callable[..., AsyncIterable[T]],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> AsyncIterable[T]:
    # wrapped(*args, **kwargs) returns an async generator; wrap iteration so the
    # ContextVar stays set while generation_span / response_span exits (which is
    # where on_span_end fires and we read the system instructions).
    system_instructions = _extract_system_instructions(args, kwargs)

    async def _gen():
        token = set_current_system_instructions(system_instructions)
        try:
            async for chunk in wrapped(*args, **kwargs):
                yield chunk
        finally:
            reset_current_system_instructions(token)

    return _gen()


_WRAPPED_TARGETS: tuple[tuple[str, str, Callable[..., Any]], ...] = (
    (
        "agents.models.openai_responses",
        "OpenAIResponsesModel.get_response",
        _wrap_get_response,
    ),
    (
        "agents.models.openai_responses",
        "OpenAIResponsesModel.stream_response",
        _wrap_stream_response,
    ),
    (
        "agents.models.openai_chatcompletions",
        "OpenAIChatCompletionsModel.get_response",
        _wrap_get_response,
    ),
    (
        "agents.models.openai_chatcompletions",
        "OpenAIChatCompletionsModel.stream_response",
        _wrap_stream_response,
    ),
)


class OpenAIAgentsInstrumentor(BaseInstrumentor):
    """Instrumentor for the OpenAI Agents SDK tracing module."""

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return instruments

    @override
    def _instrument(self, **kwargs: Any):
        try:
            from agents.tracing import add_trace_processor
        except Exception:
            logger.debug("OpenAI Agents SDK not available; skipping instrumentation")
            return

        processor = LaminarAgentsTraceProcessor()
        try:
            add_trace_processor(processor)
        except Exception:
            logger.warning("Failed to register Laminar Agents processor: %s", exc_info=True)
            raise
        self._processor: LaminarAgentsTraceProcessor | None = processor

        for module, name, wrapper in _WRAPPED_TARGETS:
            try:
                wrap_function_wrapper(module, name, wrapper)
            except (AttributeError, ModuleNotFoundError, ImportError):
                logger.debug("Failed to wrap %s.%s", module, name)

        logger.debug("Laminar OpenAI Agents trace processor registered")

    @override
    def _uninstrument(self, **kwargs: Any):
        for module, name, _ in _WRAPPED_TARGETS:
            try:
                cls_name, func_name = name.split(".", 1)
                mod = __import__(module, fromlist=[cls_name])
                cls = getattr(mod, cls_name, None)
                if cls is not None:
                    unwrap(cls, func_name)
            except Exception:
                logger.debug("Failed to unwrap %s.%s", module, name)

        processor = cast(LaminarAgentsTraceProcessor | None, getattr(self, "_processor", None))
        if processor is None:
            return
        try:
            from agents.tracing import get_trace_provider

            provider = get_trace_provider()
            # The SDK has no remove_trace_processor API, so read the
            # internal list and write back via the public set_processors.
            mp = getattr(provider, "_multi_processor", None)
            if mp is not None:
                current = getattr(mp, "_processors", ())
                provider.set_processors([p for p in current if p is not processor])
        except Exception:
            logger.debug("Failed to set OpenAI agents processor", exc_info=True)
        processor.shutdown()
        self._processor = None

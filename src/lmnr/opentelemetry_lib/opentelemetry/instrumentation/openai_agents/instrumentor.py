"""OpenAIAgentsInstrumentor - BaseInstrumentor for OpenAI Agents SDK."""

from __future__ import annotations

from typing import Any, Collection

from lmnr.sdk.log import get_default_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from .helpers import (
    reset_current_model_name,
    reset_current_system_instructions,
    set_current_model_name,
    set_current_system_instructions,
)
from .processor import LaminarAgentsTraceProcessor

logger = get_default_logger(__name__)

_instruments = ("openai-agents >= 0.7.0",)


def _extract_system_instructions(args: tuple, kwargs: dict) -> Any:
    if "system_instructions" in kwargs:
        return kwargs["system_instructions"]
    if args:
        return args[0]
    return None


def _model_name(instance: Any) -> str | None:
    name = getattr(instance, "model", None)
    return name if isinstance(name, str) else None


def _enter_model_call(instance: Any, args: tuple, kwargs: dict) -> tuple:
    return (
        set_current_system_instructions(_extract_system_instructions(args, kwargs)),
        set_current_model_name(_model_name(instance)),
    )


def _exit_model_call(tokens: tuple) -> None:
    instructions_token, model_token = tokens
    reset_current_model_name(model_token)
    reset_current_system_instructions(instructions_token)


async def _wrap_get_response(wrapped, instance, args, kwargs):
    tokens = _enter_model_call(instance, args, kwargs)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        _exit_model_call(tokens)


def _wrap_stream_response(wrapped, instance, args, kwargs):
    # wrapped(*args, **kwargs) returns an async generator; wrap iteration so the
    # ContextVars stay set while generation_span / response_span exits (which is
    # where on_span_end fires and we read them).
    async def _gen():
        tokens = _enter_model_call(instance, args, kwargs)
        try:
            async for chunk in wrapped(*args, **kwargs):
                yield chunk
        finally:
            _exit_model_call(tokens)

    return _gen()


_WRAPPED_TARGETS: tuple[tuple[str, str, Any], ...] = (
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

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        try:
            from agents.tracing import add_trace_processor
        except Exception:
            logger.debug("OpenAI Agents SDK not available; skipping instrumentation")
            return

        processor = LaminarAgentsTraceProcessor()
        try:
            add_trace_processor(processor)
        except Exception as exc:
            logger.warning("Failed to register Laminar Agents processor: %s", exc)
            raise
        self._processor = processor

        for module, name, wrapper in _WRAPPED_TARGETS:
            try:
                wrap_function_wrapper(module, name, wrapper)
            except (AttributeError, ModuleNotFoundError, ImportError):
                logger.debug("Failed to wrap %s.%s", module, name)

        logger.debug("Laminar OpenAI Agents trace processor registered")

    def _uninstrument(self, **kwargs):
        for module, name, _ in _WRAPPED_TARGETS:
            try:
                cls_name, func_name = name.split(".", 1)
                mod = __import__(module, fromlist=[cls_name])
                cls = getattr(mod, cls_name, None)
                if cls is not None:
                    unwrap(cls, func_name)
            except Exception:
                logger.debug("Failed to unwrap %s.%s", module, name)

        processor = getattr(self, "_processor", None)
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
            pass
        processor.shutdown()
        self._processor = None

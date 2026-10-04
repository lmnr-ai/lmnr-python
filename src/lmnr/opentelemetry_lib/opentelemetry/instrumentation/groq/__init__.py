"""OpenTelemetry Groq instrumentation"""

import logging
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Collection,
    Generator,
    Iterable,
    Sequence,
)
from importlib.metadata import version
from typing import Any, cast

from opentelemetry import context as context_api
from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode
from typing_extensions import TypeVar, override

from groq._streaming import AsyncStream, Stream
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.groq.event_models import Usage
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.groq.span_utils import (
    set_input_attributes,
    set_model_input_attributes,
    set_model_response_attributes,
    set_model_streaming_response_attributes,
    set_response_attributes,
    set_streaming_response_attributes,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    safe_start_span,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    stamp_instrumentation_scope,
)

logger = logging.getLogger(__name__)

_instruments = ("groq >= 0.9.0",)

T = TypeVar("T")

def is_streaming_response(response: Any) -> bool:
    return isinstance(response, (Stream, AsyncStream))


def _process_streaming_chunk(chunk: Any) -> tuple[Any, Any, Any]:
    """Extract content, finish_reason and usage from a streaming chunk."""
    if not getattr(chunk, "choices", None):
        return None, None, None

    delta = chunk.choices[0].delta
    content = delta.content if hasattr(delta, "content") else None
    finish_reason = chunk.choices[0].finish_reason

    # Extract usage from x_groq if present in the final chunk
    usage = None
    if hasattr(chunk, "x_groq") and chunk.x_groq and getattr(chunk.x_groq, "usage", None):
        usage = chunk.x_groq.usage

    return content, finish_reason, usage


@dont_throw
def _handle_streaming_response(
    span: Span,
    accumulated_content: Any,
    finish_reason: str | None,
    usage: Usage | None,
):
    set_model_streaming_response_attributes(span, usage)
    set_streaming_response_attributes(span, accumulated_content, finish_reason, usage)


def _create_stream_processor(
    response: Iterable[T],
    span: Span,
)-> Generator[T]:
    """Create a generator that processes a stream while collecting telemetry."""
    accumulated_content = ""
    finish_reason = None
    usage = None

    for chunk in response:
        try:
            content, chunk_finish_reason, chunk_usage = _process_streaming_chunk(chunk)
            if content:
                accumulated_content += content
            if chunk_finish_reason:
                finish_reason = chunk_finish_reason
            if chunk_usage:
                usage = chunk_usage
        except Exception:
            logger.warning("Failed to process streaming chunk for groq span", exc_info=True)
        finally:
            yield chunk

    _handle_streaming_response(span, accumulated_content, finish_reason, usage)

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))

    span.end()


async def _create_async_stream_processor(
    response: AsyncIterable[T],
    span: Span
) -> AsyncGenerator[T]:
    """Create an async generator that processes a stream while collecting telemetry."""
    accumulated_content = ""
    finish_reason = None
    usage = None

    async for chunk in response:
        try:
            content, chunk_finish_reason, chunk_usage = _process_streaming_chunk(chunk)
            if content:
                accumulated_content += content
            if chunk_finish_reason:
                finish_reason = chunk_finish_reason
            if chunk_usage:
                usage = chunk_usage
        except Exception:
            logger.warning(
                "Failed to process streaming chunk for groq span", exc_info=True,
            )
        finally:
            yield chunk

    _handle_streaming_response(span, accumulated_content, finish_reason, usage)

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))

    span.end()


@dont_throw
def _handle_input(span: Span, kwargs: dict[str, Any]):
    set_model_input_attributes(span, kwargs)
    set_input_attributes(span, kwargs)


@dont_throw
def _handle_response(span: Span, response: Any):
    set_model_response_attributes(span, response)
    set_response_attributes(span, response)


def _wrap(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T | Generator[T]:
    """Instruments and calls every function defined in WRAPPED_FUNCTIONS."""
    if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name") or "groq.chat"
    span = safe_start_span(
        name=name, attributes={"gen_ai.system": "groq"}, span_type="LLM"
    )
    if not span:
        logger.warning("Failed to start span for groq chat")
        return wrapped(*args, **kwargs)

    stamp_instrumentation_scope(span, to_wrap)
    _handle_input(span, kwargs)

    response = wrapped(*args, **kwargs)


    if is_streaming_response(response):
        try:
            return _create_stream_processor(cast(Iterable[Any], response), span)
        except Exception as ex:
            logger.warning("Failed to process streaming response for groq span", exc_info=True)
            span.record_exception(ex)
            span.set_status(Status(StatusCode.ERROR))
            span.end()
            raise
    elif response:
        try:
            _handle_response(span, response)

        except Exception:
            logger.warning("Failed to set response attributes for groq span", exc_info=True)

        if span.is_recording():
            span.set_status(Status(StatusCode.OK))
    span.end()
    return response


async def _awrap(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[T]],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T | AsyncGenerator[T]:
    """Instruments and calls every function defined in WRAPPED_FUNCTIONS."""
    if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    name = to_wrap.get("span_name") or "groq.chat"
    span = safe_start_span(
        name=name, attributes={"gen_ai.system": "groq"}, span_type="LLM"
    )
    if not span:
        logger.warning("Failed to start span for groq chat")
        return await wrapped(*args, **kwargs)

    stamp_instrumentation_scope(span, to_wrap)
    _handle_input(span, kwargs)

    response = await wrapped(*args, **kwargs)


    if is_streaming_response(response):
        try:
            return _create_async_stream_processor(cast(AsyncIterable[Any], response), span)
        except Exception as ex:
            logger.warning(
                "Failed to process streaming response for groq span",
                exc_info=True,
            )
            span.record_exception(ex)
            span.set_status(Status(StatusCode.ERROR))
            span.end()
            raise
    elif response:
        _handle_response(span, response)

        if span.is_recording():
            span.set_status(Status(StatusCode.OK))
    span.end()
    return response


WRAPPED_FUNCTIONS: list[WrappedFunctionSpec] = [
    WrappedFunctionSpec(
        package_name="groq.resources.chat.completions",
        object_name="Completions",
        method_name="create",
        span_name="groq.chat",
        is_async=False,
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="groq.resources.chat.completions",
        object_name="AsyncCompletions",
        method_name="create",
        span_name="groq.chat",
        is_async=True,
        wrapper_function=_awrap,
    ),
]


class GroqInstrumentor(BaseLaminarInstrumentor):
    """An instrumentor for Groq's client library."""

    _scope: LaminarInstrumentationScopeAttributes | None = None

    @override
    def __init__(self):
        super().__init__()
        self.instrumentor_config: LaminarInstrumentorConfig = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                groq_version = version("groq")
            except Exception:
                logger.debug("Failed to get groq version", exc_info=True)
                groq_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="groq",
                version=groq_version,
            )
        return self._scope

"""OpenTelemetry Groq instrumentation"""

import logging
from importlib.metadata import version
from typing import Any, Collection, Sequence

from opentelemetry import context as context_api
from .config import Config
from .span_utils import (
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
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace.status import Status, StatusCode

from groq._streaming import AsyncStream, Stream

logger = logging.getLogger(__name__)

_instruments = ("groq >= 0.9.0",)


def is_streaming_response(response):
    return isinstance(response, Stream) or isinstance(response, AsyncStream)


def _process_streaming_chunk(chunk):
    """Extract content, finish_reason and usage from a streaming chunk."""
    if not chunk.choices:
        return None, None, None

    delta = chunk.choices[0].delta
    content = delta.content if hasattr(delta, "content") else None
    finish_reason = chunk.choices[0].finish_reason

    # Extract usage from x_groq if present in the final chunk
    usage = None
    if hasattr(chunk, "x_groq") and chunk.x_groq and chunk.x_groq.usage:
        usage = chunk.x_groq.usage

    return content, finish_reason, usage


@dont_throw
def _handle_streaming_response(span, accumulated_content, finish_reason, usage):
    set_model_streaming_response_attributes(span, usage)
    set_streaming_response_attributes(span, accumulated_content, finish_reason, usage)


def _create_stream_processor(
    response,
    span,
):
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
        except Exception as e:
            logger.warning(
                "Failed to process streaming chunk for groq span, error: %s", str(e)
            )
        finally:
            yield chunk

    _handle_streaming_response(span, accumulated_content, finish_reason, usage)

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))

    span.end()


async def _create_async_stream_processor(response, span):
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
        except Exception as e:
            logger.warning(
                "Failed to process streaming chunk for groq span, error: %s", str(e)
            )
        finally:
            yield chunk

    _handle_streaming_response(span, accumulated_content, finish_reason, usage)

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))

    span.end()


@dont_throw
def _handle_input(span, kwargs):
    set_model_input_attributes(span, kwargs)
    set_input_attributes(span, kwargs)


@dont_throw
def _handle_response(span, response):
    set_model_response_attributes(span, response)
    set_response_attributes(span, response)


def _wrap(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    """Instruments and calls every function defined in WRAPPED_FUNCTIONS."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
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

    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        raise e

    if is_streaming_response(response):
        try:
            return _create_stream_processor(response, span)
        except Exception as ex:
            logger.warning(
                "Failed to process streaming response for groq span, error: %s",
                str(ex),
            )
            span.set_status(Status(StatusCode.ERROR))
            span.end()
            raise
    elif response:
        try:
            _handle_response(span, response)

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set response attributes for groq span, error: %s",
                str(ex),
            )

        if span.is_recording():
            span.set_status(Status(StatusCode.OK))
    span.end()
    return response


async def _awrap(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    """Instruments and calls every function defined in WRAPPED_FUNCTIONS."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
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

    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        raise e

    if is_streaming_response(response):
        try:
            return await _create_async_stream_processor(response, span)
        except Exception as ex:
            logger.warning(
                "Failed to process streaming response for groq span, error: %s",
                str(ex),
            )
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

    def __init__(
        self,
        enrich_token_usage: bool = False,
        use_legacy_attributes: bool = True,
    ):
        super().__init__()
        Config.enrich_token_usage = enrich_token_usage
        Config.use_legacy_attributes = use_legacy_attributes
        self.instrumentor_config = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                groq_version = version("groq")
            except Exception as e:
                logger.debug(f"Failed to get groq version {e}")
                groq_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="groq",
                version=groq_version,
            )
        return self._scope

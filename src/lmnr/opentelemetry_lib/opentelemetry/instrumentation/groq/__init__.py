"""OpenTelemetry Groq instrumentation"""

import logging
import os
from typing import Collection

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
from .version import __version__
from lmnr.opentelemetry_lib.tracing.context import get_current_context
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

from groq._streaming import AsyncStream, Stream

logger = logging.getLogger(__name__)

_instruments = ("groq >= 0.9.0",)


WRAPPED_METHODS = [
    {
        "package": "groq.resources.chat.completions",
        "object": "Completions",
        "method": "create",
        "span_name": "groq.chat",
    },
]
WRAPPED_AMETHODS = [
    {
        "package": "groq.resources.chat.completions",
        "object": "AsyncCompletions",
        "method": "create",
        "span_name": "groq.chat",
    },
]


def is_streaming_response(response):
    return isinstance(response, Stream) or isinstance(response, AsyncStream)


def _with_chat_telemetry_wrapper(func):
    """Helper for providing tracer for wrapper functions. Includes metric collectors."""

    def _with_chat_telemetry(
        tracer,
        to_wrap,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                tracer,
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_chat_telemetry


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
        content, chunk_finish_reason, chunk_usage = _process_streaming_chunk(chunk)
        if content:
            accumulated_content += content
        if chunk_finish_reason:
            finish_reason = chunk_finish_reason
        if chunk_usage:
            usage = chunk_usage
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
        content, chunk_finish_reason, chunk_usage = _process_streaming_chunk(chunk)
        if content:
            accumulated_content += content
        if chunk_finish_reason:
            finish_reason = chunk_finish_reason
        if chunk_usage:
            usage = chunk_usage
        yield chunk

    _handle_streaming_response(span, accumulated_content, finish_reason, usage)

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))

    span.end()


def _handle_input(span, kwargs):
    set_model_input_attributes(span, kwargs)
    set_input_attributes(span, kwargs)


def _handle_response(span, response):
    set_model_response_attributes(span, response)
    set_response_attributes(span, response)


@_with_chat_telemetry_wrapper
def _wrap(
    tracer: Tracer,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Groq",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
        context=get_current_context(),
    )

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


@_with_chat_telemetry_wrapper
async def _awrap(
    tracer,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Groq",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
        context=get_current_context(),
    )

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


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


class GroqInstrumentor(BaseInstrumentor):
    """An instrumentor for Groq's client library."""

    def __init__(
        self,
        enrich_token_usage: bool = False,
        use_legacy_attributes: bool = True,
    ):
        super().__init__()
        Config.enrich_token_usage = enrich_token_usage
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(
                        tracer,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some methods do not exist

        for wrapped_method in WRAPPED_AMETHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _awrap(
                        tracer,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some methods do not exist

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )
        for wrapped_method in WRAPPED_AMETHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )

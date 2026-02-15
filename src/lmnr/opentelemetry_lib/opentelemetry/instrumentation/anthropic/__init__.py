"""OpenTelemetry Anthropic instrumentation"""

import logging
import time
from typing import Callable, Collection, Optional

from opentelemetry import context as context_api
from .config import Config
from .span_utils import (
    aset_input_attributes,
    aset_response_attributes,
    set_response_attributes,
)
from .streaming import (
    abuild_from_streaming_response,
    build_from_streaming_response,
)
from .utils import (
    dont_throw,
    run_async,
    set_span_attribute,
)
from .streaming import (
    WrappedAsyncMessageStreamManager,
    WrappedMessageStreamManager,
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
from opentelemetry.trace import Span, SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from typing_extensions import Coroutine
from wrapt import wrap_function_wrapper

from anthropic._streaming import AsyncStream, Stream

logger = logging.getLogger(__name__)

_instruments = ("anthropic >= 0.3.11",)


WRAPPED_METHODS = [
    {
        "package": "anthropic.resources.completions",
        "object": "Completions",
        "method": "create",
        "span_name": "anthropic.completion",
    },
    {
        "package": "anthropic.resources.messages",
        "object": "Messages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
    {
        "package": "anthropic.resources.messages",
        "object": "Messages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
    # This method is on an async resource, but is meant to be called as
    # an async context manager (async with), which we don't need to await;
    # thus, we wrap it with a sync wrapper
    {
        "package": "anthropic.resources.messages",
        "object": "AsyncMessages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
    # Beta API methods (regular Anthropic SDK)
    {
        "package": "anthropic.resources.beta.messages.messages",
        "object": "Messages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
    {
        "package": "anthropic.resources.beta.messages.messages",
        "object": "Messages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
    # read note on async with above
    {
        "package": "anthropic.resources.beta.messages.messages",
        "object": "AsyncMessages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
    # Beta API methods (Bedrock SDK)
    {
        "package": "anthropic.lib.bedrock._beta_messages",
        "object": "Messages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
    {
        "package": "anthropic.lib.bedrock._beta_messages",
        "object": "Messages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
    # read note on async with above
    {
        "package": "anthropic.lib.bedrock._beta_messages",
        "object": "AsyncMessages",
        "method": "stream",
        "span_name": "anthropic.chat",
    },
]

WRAPPED_AMETHODS = [
    {
        "package": "anthropic.resources.completions",
        "object": "AsyncCompletions",
        "method": "create",
        "span_name": "anthropic.completion",
    },
    {
        "package": "anthropic.resources.messages",
        "object": "AsyncMessages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
    # Beta API async methods (regular Anthropic SDK)
    {
        "package": "anthropic.resources.beta.messages.messages",
        "object": "AsyncMessages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
    # Beta API async methods (Bedrock SDK)
    {
        "package": "anthropic.lib.bedrock._beta_messages",
        "object": "AsyncMessages",
        "method": "create",
        "span_name": "anthropic.chat",
    },
]


def is_streaming_response(response):
    return isinstance(response, Stream) or isinstance(response, AsyncStream)


def is_stream_manager(response):
    """Check if response is a MessageStreamManager or AsyncMessageStreamManager"""
    try:
        from anthropic.lib.streaming._messages import (
            MessageStreamManager,
            AsyncMessageStreamManager,
        )

        return isinstance(response, (MessageStreamManager, AsyncMessageStreamManager))
    except ImportError:
        # Check by class name as fallback
        return (
            response.__class__.__name__ == "MessageStreamManager"
            or response.__class__.__name__ == "AsyncMessageStreamManager"
        )


@dont_throw
async def _aset_token_usage(
    span,
    anthropic,
    request,
    response,
):
    # Handle with_raw_response wrapped responses first
    if response and hasattr(response, "parse") and callable(response.parse):
        try:
            response = response.parse()
        except Exception as e:
            logger.debug(f"Failed to parse with_raw_response: {e}")
            return

    usage = getattr(response, "usage", None) if response else None

    if usage:
        prompt_tokens = getattr(usage, "input_tokens", 0)
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    else:
        prompt_tokens = 0
        cache_read_tokens = 0
        cache_creation_tokens = 0

    input_tokens = prompt_tokens + cache_read_tokens + cache_creation_tokens

    if usage:
        completion_tokens = getattr(usage, "output_tokens", 0)
    else:
        completion_tokens = 0
        if hasattr(anthropic, "count_tokens"):
            completion_attr = getattr(response, "completion", None)
            content_attr = getattr(response, "content", None)
            if completion_attr:
                completion_tokens = await anthropic.count_tokens(completion_attr)
            elif content_attr:
                completion_tokens = await anthropic.count_tokens(content_attr[0].text)

    total_tokens = input_tokens + completion_tokens

    content_attr = getattr(response, "content", None)
    completion_attr = getattr(response, "completion", None)

    set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)
    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
    )
    set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cache_read_tokens
    )
    set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
        cache_creation_tokens,
    )


@dont_throw
def _set_token_usage(
    span,
    anthropic,
    request,
    response,
):
    # Handle with_raw_response wrapped responses first
    if response and hasattr(response, "parse") and callable(response.parse):
        try:
            response = response.parse()
        except Exception as e:
            logger.debug(f"Failed to parse with_raw_response: {e}")
            return

    usage = getattr(response, "usage", None) if response else None

    if usage:
        prompt_tokens = getattr(usage, "input_tokens", 0)
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    else:
        prompt_tokens = 0
        cache_read_tokens = 0
        cache_creation_tokens = 0

    input_tokens = prompt_tokens + cache_read_tokens + cache_creation_tokens

    if usage:
        completion_tokens = getattr(usage, "output_tokens", 0)
    else:
        completion_tokens = 0
        if hasattr(anthropic, "count_tokens"):
            completion_attr = getattr(response, "completion", None)
            content_attr = getattr(response, "content", None)
            if completion_attr:
                completion_tokens = anthropic.count_tokens(completion_attr)
            elif content_attr:
                completion_tokens = anthropic.count_tokens(content_attr[0].text)

    total_tokens = input_tokens + completion_tokens

    content_attr = getattr(response, "content", None)
    completion_attr = getattr(response, "completion", None)

    set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)
    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
    )
    set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cache_read_tokens
    )
    set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
        cache_creation_tokens,
    )


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


@dont_throw
def _handle_input(span: Span, kwargs):
    if not span.is_recording():
        return
    run_async(aset_input_attributes(span, kwargs))


@dont_throw
async def _ahandle_input(span: Span, kwargs):
    if not span.is_recording():
        return
    await aset_input_attributes(span, kwargs)


@dont_throw
def _handle_response(span: Span, response):
    if not span.is_recording():
        return
    set_response_attributes(span, response)


@dont_throw
async def _ahandle_response(span: Span, response):
    if not span.is_recording():
        return
    await aset_response_attributes(span, response)


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
            SpanAttributes.LLM_SYSTEM: "anthropic",
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
        return build_from_streaming_response(
            span,
            response,
            instance._client,
            kwargs,
        )
    elif is_stream_manager(response):
        if response.__class__.__name__ == "AsyncMessageStreamManager":
            return WrappedAsyncMessageStreamManager(
                response,
                span,
                instance._client,
                kwargs,
            )
        else:
            return WrappedMessageStreamManager(
                response,
                span,
                instance._client,
                kwargs,
            )
    elif response:
        try:
            _handle_response(span, response)
            if span.is_recording():
                _set_token_usage(
                    span,
                    instance._client,
                    kwargs,
                    response,
                )
        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set response attributes for anthropic span, error: %s",
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
            SpanAttributes.LLM_SYSTEM: "anthropic",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
        context=get_current_context(),
    )
    await _ahandle_input(span, kwargs)

    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        raise e

    if is_streaming_response(response):
        return abuild_from_streaming_response(
            span,
            response,
            instance._client,
            kwargs,
        )
    elif is_stream_manager(response):
        if response.__class__.__name__ == "AsyncMessageStreamManager":
            return WrappedAsyncMessageStreamManager(
                response,
                span,
                instance._client,
                kwargs,
            )
        else:
            return WrappedMessageStreamManager(
                response,
                span,
                instance._client,
                kwargs,
            )
    elif response:
        await _ahandle_response(span, response)

        if span.is_recording():
            await _aset_token_usage(
                span,
                instance._client,
                kwargs,
                response,
            )
            span.set_status(Status(StatusCode.OK))
    span.end()
    return response


class AnthropicInstrumentor(BaseInstrumentor):
    """An instrumentor for Anthropic's client library."""

    def __init__(
        self,
        enrich_token_usage: bool = False,
        exception_logger=None,
        use_legacy_attributes: bool = True,
        get_common_metrics_attributes: Callable[[], dict] = lambda: {},
        upload_base64_image: Optional[
            Callable[[str, str, str, str], Coroutine[None, None, str]]
        ] = None,
    ):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.enrich_token_usage = enrich_token_usage
        Config.get_common_metrics_attributes = get_common_metrics_attributes
        Config.upload_base64_image = upload_base64_image
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
                logger.debug(
                    f"Successfully wrapped {wrap_package}.{wrap_object}.{wrap_method}"
                )
            except Exception as e:
                logger.debug(
                    f"Failed to wrap {wrap_package}.{wrap_object}.{wrap_method}: {e}"
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
            except Exception:
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

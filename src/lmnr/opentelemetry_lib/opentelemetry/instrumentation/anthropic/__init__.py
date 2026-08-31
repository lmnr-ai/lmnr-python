"""OpenTelemetry Anthropic instrumentation"""

import logging
from importlib.metadata import version
from typing import Any, Callable, Collection, Sequence

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode

from anthropic._streaming import AsyncStream, Stream
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    safe_start_span,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    stamp_instrumentation_scope,
)

from .config import Config
from .rollout import get_anthropic_rollout_wrapper
from .span_utils import (
    aset_input_attributes,
    aset_response_attributes,
    set_response_attributes,
)
from .streaming import (
    WrappedAsyncMessageStreamManager,
    WrappedMessageStreamManager,
    abuild_from_streaming_response,
    build_from_streaming_response,
)
from .utils import (
    dont_throw,
    run_async,
    set_span_attribute,
)

logger = logging.getLogger(__name__)

_instruments = ("anthropic >= 0.3.11",)


def is_streaming_response(response):
    if isinstance(response, (Stream, AsyncStream)):
        return True

    # For cached streams, they are generators, not Message objects.
    # We check for __next__ and __iter__ for sync generators,
    # and __anext__ and __aiter__ for async generators.
    # This prevents identifying Pydantic models (like Message) as streams.
    return (hasattr(response, "__next__") and hasattr(response, "__iter__")) or (
        hasattr(response, "__anext__") and hasattr(response, "__aiter__")
    )


def is_stream_manager(response):
    """Check if response is a MessageStreamManager or AsyncMessageStreamManager"""
    try:
        from anthropic.lib.streaming._messages import (
            AsyncMessageStreamManager,
            MessageStreamManager,
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

    set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
    set_span_attribute(span, "llm.usage.total_tokens", total_tokens)

    set_span_attribute(span, "gen_ai.usage.cache_read_input_tokens", cache_read_tokens)
    set_span_attribute(
        span,
        "gen_ai.usage.cache_creation_input_tokens",
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

    set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
    set_span_attribute(span, "llm.usage.total_tokens", total_tokens)

    set_span_attribute(span, "gen_ai.usage.cache_read_input_tokens", cache_read_tokens)
    set_span_attribute(
        span,
        "gen_ai.usage.cache_creation_input_tokens",
        cache_creation_tokens,
    )


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
def _handle_response(span: Span, response, record_raw_response=False):
    if not span.is_recording():
        return
    set_response_attributes(span, response)

    if record_raw_response:
        try:
            from lmnr.sdk.utils import json_dumps

            from .utils import _extract_response_data, model_as_dict

            response_data = _extract_response_data(response)
            response_dict = model_as_dict(response_data)
            set_span_attribute(span, "lmnr.sdk.raw.response", json_dumps(response_dict))
        except Exception:
            pass


@dont_throw
async def _ahandle_response(span: Span, response, record_raw_response=False):
    if not span.is_recording():
        return
    await aset_response_attributes(span, response)

    if record_raw_response:
        try:
            from lmnr.sdk.utils import json_dumps

            from .utils import _aextract_response_data, model_as_dict

            response_data = await _aextract_response_data(response)
            response_dict = model_as_dict(response_data)
            set_span_attribute(span, "lmnr.sdk.raw.response", json_dumps(response_dict))
        except Exception:
            pass


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

    span = safe_start_span(
        name=to_wrap.get("span_name") or "anthropic.chat",
        attributes={"gen_ai.system": "anthropic"},
        span_type="LLM",
    )

    if not span:
        logger.warning("Failed to start span for anthropic chat")
        return wrapped(*args, **kwargs)

    stamp_instrumentation_scope(span, to_wrap)
    _handle_input(span, kwargs)

    rollout_wrapper = get_anthropic_rollout_wrapper()
    is_rollout = rollout_wrapper is not None

    try:
        if rollout_wrapper:
            response = rollout_wrapper.wrap_create(
                wrapped,
                instance,
                args,
                kwargs,
                span=span,
                is_streaming=kwargs.get("stream", False),
                is_async=False,
            )
        else:
            response = wrapped(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        raise e

    if kwargs.get("stream") or is_streaming_response(response):
        return build_from_streaming_response(
            span,
            response,
            instance._client,
            kwargs,
            record_raw_response=is_rollout,
        )
    elif is_stream_manager(response):
        if response.__class__.__name__ == "AsyncMessageStreamManager":
            return WrappedAsyncMessageStreamManager(
                response,
                span,
                instance._client,
                kwargs,
                record_raw_response=is_rollout,
            )
        else:
            return WrappedMessageStreamManager(
                response,
                span,
                instance._client,
                kwargs,
                record_raw_response=is_rollout,
            )
    elif response:
        try:
            _handle_response(span, response, record_raw_response=is_rollout)
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

    span = safe_start_span(
        name=to_wrap.get("span_name") or "anthropic.chat",
        attributes={"gen_ai.system": "anthropic"},
        span_type="LLM",
    )

    if not span:
        logger.warning("Failed to start span for async anthropic chat")
        return await wrapped(*args, **kwargs)

    stamp_instrumentation_scope(span, to_wrap)
    await _ahandle_input(span, kwargs)

    rollout_wrapper = get_anthropic_rollout_wrapper()
    is_rollout = rollout_wrapper is not None

    try:
        if rollout_wrapper:
            response = await rollout_wrapper.wrap_create(
                wrapped,
                instance,
                args,
                kwargs,
                span=span,
                is_streaming=kwargs.get("stream", False),
                is_async=True,
            )
        else:
            response = await wrapped(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        raise e

    if kwargs.get("stream") or is_streaming_response(response):
        return abuild_from_streaming_response(
            span,
            response,
            instance._client,
            kwargs,
            record_raw_response=is_rollout,
        )
    elif is_stream_manager(response):
        if response.__class__.__name__ == "AsyncMessageStreamManager":
            return WrappedAsyncMessageStreamManager(
                response,
                span,
                instance._client,
                kwargs,
                record_raw_response=is_rollout,
            )
        else:
            return WrappedMessageStreamManager(
                response,
                span,
                instance._client,
                kwargs,
                record_raw_response=is_rollout,
            )
    elif response:
        await _ahandle_response(span, response, record_raw_response=is_rollout)

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


WRAPPED_FUNCTIONS: list[WrappedFunctionSpec] = [
    WrappedFunctionSpec(
        package_name="anthropic.resources.completions",
        object_name="Completions",
        method_name="create",
        span_name="anthropic.completion",
        is_async=False,
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="anthropic.resources.messages",
        object_name="Messages",
        method_name="create",
        span_name="anthropic.chat",
        is_async=False,
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="anthropic.resources.messages",
        object_name="Messages",
        method_name="parse",
        span_name="anthropic.chat",
        is_async=False,
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="anthropic.resources.messages",
        object_name="Messages",
        method_name="stream",
        span_name="anthropic.chat",
        is_async=False,
        wrapper_function=_wrap,
    ),
    # This method is on an async resource, but is meant to be called as
    # an async context manager (async with), which we don't need to await;
    # thus, we wrap it with a sync wrapper
    WrappedFunctionSpec(
        package_name="anthropic.resources.messages",
        object_name="AsyncMessages",
        method_name="stream",
        span_name="anthropic.chat",
        is_async=False,
        wrapper_function=_wrap,
    ),
    # Beta API methods (regular Anthropic SDK)
    WrappedFunctionSpec(
        package_name="anthropic.resources.beta.messages.messages",
        object_name="Messages",
        method_name="create",
        span_name="anthropic.chat",
        is_async=False,
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="anthropic.resources.beta.messages.messages",
        object_name="Messages",
        method_name="parse",
        span_name="anthropic.chat",
        is_async=False,
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="anthropic.resources.beta.messages.messages",
        object_name="Messages",
        method_name="stream",
        span_name="anthropic.chat",
        is_async=False,
        wrapper_function=_wrap,
    ),
    # read note on async with above
    WrappedFunctionSpec(
        package_name="anthropic.resources.beta.messages.messages",
        object_name="AsyncMessages",
        method_name="stream",
        span_name="anthropic.chat",
        is_async=False,
        wrapper_function=_wrap,
    ),
    # Beta API methods (Bedrock SDK)
    WrappedFunctionSpec(
        package_name="anthropic.lib.bedrock._beta_messages",
        object_name="Messages",
        method_name="create",
        span_name="anthropic.chat",
        is_async=False,
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="anthropic.lib.bedrock._beta_messages",
        object_name="Messages",
        method_name="stream",
        span_name="anthropic.chat",
        is_async=False,
        wrapper_function=_wrap,
    ),
    # read note on async with above
    WrappedFunctionSpec(
        package_name="anthropic.lib.bedrock._beta_messages",
        object_name="AsyncMessages",
        method_name="stream",
        span_name="anthropic.chat",
        is_async=False,
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="anthropic.resources.completions",
        object_name="AsyncCompletions",
        method_name="create",
        span_name="anthropic.completion",
        is_async=True,
        wrapper_function=_awrap,
    ),
    WrappedFunctionSpec(
        package_name="anthropic.resources.messages",
        object_name="AsyncMessages",
        method_name="create",
        span_name="anthropic.chat",
        is_async=True,
        wrapper_function=_awrap,
    ),
    WrappedFunctionSpec(
        package_name="anthropic.resources.messages",
        object_name="AsyncMessages",
        method_name="parse",
        span_name="anthropic.chat",
        is_async=True,
        wrapper_function=_awrap,
    ),
    # Beta API async methods (regular Anthropic SDK)
    WrappedFunctionSpec(
        package_name="anthropic.resources.beta.messages.messages",
        object_name="AsyncMessages",
        method_name="create",
        span_name="anthropic.chat",
        is_async=True,
        wrapper_function=_awrap,
    ),
    WrappedFunctionSpec(
        package_name="anthropic.resources.beta.messages.messages",
        object_name="AsyncMessages",
        method_name="parse",
        span_name="anthropic.chat",
        is_async=True,
        wrapper_function=_awrap,
    ),
    # Beta API async methods (Bedrock SDK)
    WrappedFunctionSpec(
        package_name="anthropic.lib.bedrock._beta_messages",
        object_name="AsyncMessages",
        method_name="create",
        span_name="anthropic.chat",
        is_async=True,
        wrapper_function=_awrap,
    ),
]


class AnthropicInstrumentor(BaseLaminarInstrumentor):
    """An instrumentor for Anthropic's client library."""

    _scope: LaminarInstrumentationScopeAttributes | None = None

    def __init__(
        self,
        enrich_token_usage: bool = False,
        exception_logger=None,
        use_legacy_attributes: bool = True,
        get_common_metrics_attributes: Callable[[], dict] = lambda: {},
    ):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.enrich_token_usage = enrich_token_usage
        Config.get_common_metrics_attributes = get_common_metrics_attributes
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
                anthropic_version = version("anthropic")
            except Exception as e:
                logger.debug(f"Failed to get anthropic version {e}")
                anthropic_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="anthropic",
                version=anthropic_version,
            )
        return self._scope

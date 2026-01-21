from inspect import iscoroutine
from typing import Any, Callable, Sequence

from opentelemetry.trace import Status, StatusCode

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    set_span_attribute,
)
from lmnr.opentelemetry_lib.tracing.context import (
    in_litellm_context,
    _in_litellm_context,
)
from lmnr.sdk.log import get_default_logger
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    WrappedFunctionSpec,
)
from .completions import (
    process_completion_inputs,
    process_completion_kwargs,
    process_completion_response,
)
from .completions.streaming import (
    process_completion_streaming_response,
    process_completion_async_streaming_response,
)
from .responses import (
    process_responses_inputs,
    process_responses_kwargs,
    process_responses_response,
)
from .responses.streaming import (
    process_responses_streaming_response,
    process_responses_async_streaming_response,
)

logger = get_default_logger(__name__)


# this relies on users passing everything to `completion` as kwargs. LiteLLM
# does not disallow args, so in theory one could call completion like:
# completion(model, messages, timeout, temperature, top_p, ...).
# We only rely on model being first, and messages being second.
def wrap_completion(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable,
    instance: Any,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
):
    span = Laminar.start_span(
        name=to_wrap["span_name"],
        span_type=to_wrap["span_type"],
        user_id=(kwargs.get("metadata") or {}).get("user_id"),
        session_id=(kwargs.get("metadata") or {}).get("session_id"),
        tags=(kwargs.get("metadata") or {}).get("tags", []),
        metadata=(kwargs.get("metadata") or {}),
    )
    set_span_attribute(
        span,
        "lmnr.span.instrumentation_scope.name",
        to_wrap.get("instrumentation_scope", {}).get("name"),
    )
    set_span_attribute(
        span,
        "lmnr.span.instrumentation_scope.version",
        to_wrap.get("instrumentation_scope", {}).get("version"),
    )
    messages = args[1] if len(args) > 1 else kwargs.get("messages", [])
    process_completion_inputs(span, messages, kwargs.get("tools", []))
    process_completion_kwargs(span, args, kwargs)
    streaming_handled = False
    returned_coroutine = False
    try:
        with in_litellm_context():
            result = wrapped(*args, **kwargs)

        # Handle case where async methods call sync methods internally and return a coroutine
        if iscoroutine(result):
            returned_coroutine = True

            if kwargs.get("stream"):
                # For streaming, we need to return an async generator function
                # that awaits the coroutine and then delegates to the streaming processor
                # We need to maintain the litellm context through the async generator
                async def process_streaming_coroutine():
                    # Set the litellm context flag for the duration of this generator
                    token = _in_litellm_context.set(True)
                    try:
                        actual_result = await result
                        if hasattr(actual_result, "__aiter__"):
                            # Delegate to async streaming processor by yielding from it
                            async for (
                                item
                            ) in process_completion_async_streaming_response(
                                span, actual_result
                            ):
                                yield item
                        elif hasattr(actual_result, "__iter__"):
                            # Sync iterator from async context - yield from sync processor
                            for item in process_completion_streaming_response(
                                span, actual_result
                            ):
                                yield item
                        else:
                            logger.warning(
                                "Result is not an iterator, but stream is True. This is not supported."
                            )
                            # Can't yield a non-iterator result; just return it
                            # This will likely cause issues but matches the original behavior
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.end()
                        raise
                    finally:
                        _in_litellm_context.reset(token)

                return process_streaming_coroutine()
            else:
                # For non-streaming, return a coroutine that processes the result
                # We need to maintain the litellm context through the coroutine execution
                async def process_non_streaming_coroutine():
                    token = _in_litellm_context.set(True)
                    try:
                        actual_result = await result
                        process_completion_response(span, actual_result)
                        return actual_result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
                    finally:
                        _in_litellm_context.reset(token)
                        span.end()

                return process_non_streaming_coroutine()

        if kwargs.get("stream"):
            if hasattr(result, "__iter__"):
                streaming_handled = True
                return process_completion_streaming_response(span, result)
            elif hasattr(result, "__aiter__"):
                streaming_handled = True
                return process_completion_async_streaming_response(span, result)
            else:
                logger.warning(
                    "Result is not an iterator, but stream is True. This is not supported."
                )
                return result
        else:
            process_completion_response(span, result)
            return result
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        if not streaming_handled and not returned_coroutine:
            span.end()


def wrap_responses(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable,
    instance: Any,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
):
    if kwargs is None:
        kwargs = {}
    span = Laminar.start_span(
        name=to_wrap["span_name"],
        span_type=to_wrap["span_type"],
        user_id=(kwargs.get("metadata") or {}).get("user_id"),
        session_id=(kwargs.get("metadata") or {}).get("session_id"),
        tags=(kwargs.get("metadata") or {}).get("tags", []),
        metadata=(kwargs.get("metadata") or {}),
    )
    set_span_attribute(
        span,
        "lmnr.span.instrumentation_scope.name",
        to_wrap.get("instrumentation_scope", {}).get("name"),
    )
    set_span_attribute(
        span,
        "lmnr.span.instrumentation_scope.version",
        to_wrap.get("instrumentation_scope", {}).get("version"),
    )
    # responses() has input as first arg
    input_param = args[0] if args else kwargs.get("input")
    process_responses_inputs(span, input_param, kwargs.get("tools", []))
    process_responses_kwargs(span, args, kwargs)
    streaming_handled = False
    returned_coroutine = False
    try:
        with in_litellm_context():
            result = wrapped(*args, **kwargs)

        # Handle case where async methods call sync methods internally and return a coroutine
        if iscoroutine(result):
            returned_coroutine = True

            if kwargs.get("stream"):
                # For streaming, we need to return an async generator function
                # that awaits the coroutine and then delegates to the streaming processor
                # We need to maintain the litellm context through the async generator
                async def process_streaming_coroutine():
                    token = _in_litellm_context.set(True)
                    try:
                        actual_result = await result
                        if hasattr(actual_result, "__aiter__"):
                            # Delegate to async streaming processor by yielding from it
                            async for (
                                item
                            ) in process_responses_async_streaming_response(
                                span, actual_result
                            ):
                                yield item
                        elif hasattr(actual_result, "__iter__"):
                            # Sync iterator from async context - yield from sync processor
                            for item in process_responses_streaming_response(
                                span, actual_result
                            ):
                                yield item
                        else:
                            logger.warning(
                                "Result is not an iterator, but stream is True. This is not supported."
                            )
                            # Can't yield a non-iterator result; just return it
                            # This will likely cause issues but matches the original behavior
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.end()
                        raise
                    finally:
                        _in_litellm_context.reset(token)

                return process_streaming_coroutine()
            else:
                # For non-streaming, return a coroutine that processes the result
                # We need to maintain the litellm context through the coroutine execution
                async def process_non_streaming_coroutine():
                    token = _in_litellm_context.set(True)
                    try:
                        actual_result = await result
                        process_responses_response(span, actual_result)
                        return actual_result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
                    finally:
                        _in_litellm_context.reset(token)
                        span.end()

                return process_non_streaming_coroutine()

        if kwargs.get("stream"):
            if hasattr(result, "__iter__"):
                streaming_handled = True
                return process_responses_streaming_response(span, result)
            elif hasattr(result, "__aiter__"):
                streaming_handled = True
                return process_responses_async_streaming_response(span, result)
            else:
                logger.warning(
                    "Result is not an iterator, but stream is True. This is not supported."
                )
                return result
        else:
            process_responses_response(span, result)
            return result
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        if not streaming_handled and not returned_coroutine:
            span.end()

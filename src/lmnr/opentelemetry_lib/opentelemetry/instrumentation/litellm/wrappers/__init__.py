from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator, Sequence
from inspect import iscoroutine
from typing import Any, cast

from opentelemetry.context import get_value
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util.types import AttributeValue
from typing_extensions import TypedDict, TypeVar

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.litellm.rollout import (
    DualIteratorWrapper,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.litellm.wrappers.completions import (
    process_completion_inputs,
    process_completion_kwargs,
    process_completion_response,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.litellm.wrappers.completions.streaming import (
    process_completion_async_streaming_response,
    process_completion_streaming_response,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.litellm.wrappers.responses import (
    process_responses_inputs,
    process_responses_kwargs,
    process_responses_response,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.litellm.wrappers.responses.streaming import (
    process_responses_async_streaming_response,
    process_responses_streaming_response,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.context_keys import (
    DISABLE_LITELLM_INSTRUMENTATION_CONTEXT_KEY,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    stamp_instrumentation_scope,
)
from lmnr.opentelemetry_lib.tracing.context import (
    get_current_context,
    in_litellm_context,
    is_in_litellm_context,
    reset_in_litellm_context,
    set_in_litellm_context,
)
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import LaminarSpanType

logger = get_default_logger(__name__)
T = TypeVar("T")


class PassedMetadata(TypedDict):
    user_id: str
    session_id: str
    tags: list[str]


def _instrumentation_disabled() -> bool:
    """Whether an enclosing instrumentation already owns this model call.

    The openai-agents processor sets this around the `generation` span it opens
    for a `LitellmModel` call. That span already records the request, the
    response and the usage, so opening `litellm.completion` underneath it emits
    a second LLM span for the same call - the tokens and the cost of every turn
    end up counted twice in the trace.
    """
    return bool(get_value(DISABLE_LITELLM_INSTRUMENTATION_CONTEXT_KEY, get_current_context()))


def _call_uninstrumented(
    wrapped: Callable[..., T], args: Sequence[Any], kwargs: dict[str, Any]
) -> Any:
    """Run the wrapped call without opening a span for it.

    Unlike the `is_in_litellm_context()` bridge guard, this one still has to
    ENTER the litellm context: `openai/shared/chat_wrappers.py` keys off that
    flag to skip its own span for the request litellm makes underneath, so
    returning a bare `wrapped(...)` would only move the duplicate one level down
    for openai-routed models. Async calls return a coroutine here, so the flag
    is re-set for the duration of the await, the same two-step the instrumented
    path does.
    """
    with in_litellm_context():
        result = wrapped(*args, **kwargs)

    if not iscoroutine(result):
        return result

    async def await_in_litellm_context() -> Any:
        token = set_in_litellm_context(True)
        try:
            return await result
        finally:
            reset_in_litellm_context(token)

    return await_in_litellm_context()


def _get_rollout_wrapper() -> tuple[
    object | None, bool
]:  # object == LiteLLMRolloutWrapper, avoiding circular import
    """Lazy import and get rollout wrapper to avoid circular imports."""
    try:
        from lmnr.sdk.debug.replay import replay_enabled

        if not replay_enabled():
            return None, False

        from ..rollout import get_litellm_rollout_wrapper

        return get_litellm_rollout_wrapper(), True
    except Exception:
        return None, False


# this relies on users passing everything to `completion` as kwargs. LiteLLM
# does not disallow args, so in theory one could call completion like:
# completion(model, messages, timeout, temperature, top_p, ...).
# We only rely on model being first, and messages being second.
def wrap_completion(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    _instance: Any,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> Any:
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = []
    # LiteLLM bridges between its own two public entry points. A `responses()`
    # call for a model that only speaks chat completions is re-entered through
    # `completion()` (`completion_extras/litellm_responses_transformation`), and
    # the reverse bridge sends `completion()` back through `responses()`. Both
    # entry points are instrumented, so without this guard one user-visible call
    # yields two LLM spans carrying identical messages and identical usage —
    # double-counting tokens and cost. The outermost wrapper owns the span; the
    # inner one passes the call straight through.
    if is_in_litellm_context():
        return wrapped(*args, **kwargs)
    if _instrumentation_disabled():
        return _call_uninstrumented(wrapped, args, kwargs)
    meta = cast(PassedMetadata, kwargs.get("metadata") or {})
    span = Laminar.start_span(
        name=to_wrap.get("span_name") or "litellm.completion",
        span_type=cast(LaminarSpanType, to_wrap.get("span_type") or "LLM"),
        user_id=meta.get("user_id"),
        session_id=meta.get("session_id"),
        tags=meta.get("tags", []),
        metadata=cast(dict[str, AttributeValue], meta),  # pyright: ignore[reportInvalidCast]
    )
    stamp_instrumentation_scope(span, to_wrap)
    messages = args[1] if len(args) > 1 else kwargs.get("messages", [])
    process_completion_inputs(span, messages, kwargs.get("tools", []))
    process_completion_kwargs(span, args, kwargs)
    streaming_handled = False
    returned_coroutine = False

    # Check for rollout mode
    rollout_wrapper, is_rollout = _get_rollout_wrapper()

    try:
        # If in rollout mode, delegate to rollout wrapper
        if rollout_wrapper:
            from ..rollout import LiteLLMRolloutWrapper

            rollout_wrapper = cast(LiteLLMRolloutWrapper, rollout_wrapper)
            with Laminar.use_span(span), in_litellm_context():
                result = rollout_wrapper.wrap_completion(
                    wrapped,
                    args,
                    kwargs,
                    is_streaming=cast(bool, kwargs.get("stream", False)),
                )
        else:
            # Activate our span as the current OTel span for the duration of the
            # underlying call. LiteLLM's `langfuse_otel` success callback runs
            # synchronously inside `wrapped()` and resolves its parent via
            # `OpenTelemetry._get_span_context` — Priority 3 of which is
            # `trace.get_current_span()`. With no active span it would latch onto
            # whatever span happens to be current (the user's `@observe` root),
            # fold its hybrid openinference/langfuse attributes onto it, and skip
            # creating its own `litellm_request` span — mis-marking the root as
            # LLM. Making our `litellm.completion` span current keeps litellm's
            # attributes on the LLM span where they belong.
            with Laminar.use_span(span), in_litellm_context():
                result = wrapped(*args, **kwargs)

        # Handle case where async methods call sync methods internally and return a coroutine
        if iscoroutine(result):
            returned_coroutine = True

            if kwargs.get("stream"):
                # For streaming, we need to return an async generator function
                # that awaits the coroutine and then delegates to the streaming processor
                # We need to maintain the litellm context through the async generator
                async def process_streaming_coroutine() -> AsyncGenerator[Any]:
                    # Set the litellm context flag for the duration of this generator
                    token = set_in_litellm_context(True)
                    try:
                        actual_result = await result
                        if hasattr(actual_result, "__aiter__"):
                            # Delegate to async streaming processor by yielding from it
                            processed: AsyncGenerator[Any] = cast(
                                Any,
                                process_completion_async_streaming_response(
                                    span,
                                    actual_result,
                                    record_raw_response=is_rollout,
                                ),
                            )
                            async for item in processed:
                                yield item
                        elif hasattr(actual_result, "__iter__"):
                            # Sync iterator from async context - yield from sync processor
                            for item in process_completion_streaming_response(
                                span,
                                actual_result,
                                record_raw_response=is_rollout,
                            ):
                                yield item
                        else:
                            logger.warning(
                                "Result is not an iterator, but stream is True. This is not supported."
                            )
                            span.end()
                            # Can't yield a non-iterator result; just return it
                            # This will likely cause issues but matches the original behavior
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.end()
                        raise
                    finally:
                        reset_in_litellm_context(token)

                return process_streaming_coroutine()
            else:
                # For non-streaming, return a coroutine that processes the result
                # We need to maintain the litellm context through the coroutine execution
                async def process_non_streaming_coroutine() -> T:
                    token = set_in_litellm_context(True)
                    try:
                        actual_result = await result
                        _processed_response = process_completion_response(
                            span, actual_result, record_raw_response=is_rollout
                        )
                        return actual_result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
                    finally:
                        reset_in_litellm_context(token)
                        span.end()

                return process_non_streaming_coroutine()

        if kwargs.get("stream"):
            # Check if this is our DualIteratorWrapper - if so, set attributes and return directly
            if isinstance(result, DualIteratorWrapper):
                # Set span attributes directly without consuming the iterator
                result.set_span_attributes(span, record_raw_response=is_rollout)
                streaming_handled = True
                span.end()
                return result
            elif hasattr(result, "__iter__"):
                streaming_handled = True
                return process_completion_streaming_response(
                    span,
                    cast(Generator[Any], result),
                    record_raw_response=is_rollout,
                )
            elif hasattr(result, "__aiter__"):
                streaming_handled = True
                return process_completion_async_streaming_response(
                    span,
                    cast(AsyncGenerator[Any], result),
                    record_raw_response=is_rollout,
                )
            else:
                logger.warning(
                    "Result is not an iterator, but stream is True. This is not supported."
                )
                return result
        else:
            _processed_result = process_completion_response(
                span, result, record_raw_response=is_rollout
            )
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
    wrapped: Callable[..., T],
    _instance: Any,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> Any:
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = []
    # See `wrap_completion`: litellm re-enters `responses()` from `completion()`
    # for models that only speak the Responses API. Whichever entry point the
    # user called owns the span; the bridged inner call passes through.
    if is_in_litellm_context():
        return wrapped(*args, **kwargs)
    if _instrumentation_disabled():
        return _call_uninstrumented(wrapped, args, kwargs)
    meta = cast(PassedMetadata, kwargs.get("metadata") or {})
    span = Laminar.start_span(
        name=to_wrap.get("span_name") or "litellm.responses",
        span_type=cast(LaminarSpanType, to_wrap.get("span_type") or "LLM"),
        user_id=meta.get("user_id"),
        session_id=meta.get("session_id"),
        tags=meta.get("tags", []),
        metadata=cast(dict[str, AttributeValue], meta),  # pyright: ignore[reportInvalidCast]
    )
    stamp_instrumentation_scope(span, to_wrap)
    # responses() has input as first arg
    input_param = args[0] if args else kwargs.get("input")
    process_responses_inputs(span, input_param, kwargs.get("tools", []))
    process_responses_kwargs(span, args, kwargs)
    streaming_handled = False
    returned_coroutine = False

    # Check for rollout mode
    rollout_wrapper, is_rollout = _get_rollout_wrapper()

    try:
        # If in rollout mode, delegate to rollout wrapper
        if rollout_wrapper:
            from ..rollout import LiteLLMRolloutWrapper

            rollout_wrapper = cast(LiteLLMRolloutWrapper, rollout_wrapper)
            with Laminar.use_span(span), in_litellm_context():
                result = rollout_wrapper.wrap_responses(
                    wrapped,
                    args,
                    kwargs,
                    is_streaming=cast(bool, kwargs.get("stream", False)),
                )
        else:
            # See `wrap_completion`: activate our span so litellm's
            # `langfuse_otel` callback parents its attributes onto the
            # `litellm.responses` span instead of the user's `@observe` root,
            # and mark the litellm context so a bridged inner `completion()`
            # call — plus the raw provider instrumentors underneath it — know
            # this call is already traced.
            with Laminar.use_span(span), in_litellm_context():
                result = wrapped(*args, **kwargs)

        # Handle case where async methods call sync methods internally and return a coroutine
        if iscoroutine(result):
            returned_coroutine = True

            if kwargs.get("stream"):
                # For streaming, we need to return an async generator function
                # that awaits the coroutine and then delegates to the streaming processor
                # We need to maintain the litellm context through the async generator
                async def process_streaming_coroutine() -> AsyncGenerator[Any]:
                    token = set_in_litellm_context(True)
                    try:
                        actual_result = await result
                        if hasattr(actual_result, "__aiter__"):
                            # Delegate to async streaming processor by yielding from it
                            processed: AsyncIterator[Any] = cast(
                                Any,
                                process_responses_async_streaming_response(
                                    span,
                                    actual_result,
                                    record_raw_response=is_rollout,
                                ),
                            )
                            async for item in processed:
                                yield item
                        elif hasattr(actual_result, "__iter__"):
                            # Sync iterator from async context - yield from sync processor
                            for item in process_responses_streaming_response(
                                span,
                                actual_result,
                                record_raw_response=is_rollout,
                            ):
                                yield item
                        else:
                            logger.warning(
                                "Result is not an iterator, but stream is True. This is not supported."
                            )
                            span.end()
                            # Can't yield a non-iterator result; just return it
                            # This will likely cause issues but matches the original behavior
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.end()
                        raise
                    finally:
                        reset_in_litellm_context(token)

                return process_streaming_coroutine()
            else:
                # For non-streaming, return a coroutine that processes the result
                # We need to maintain the litellm context through the coroutine execution
                async def process_non_streaming_coroutine() -> T:
                    token = set_in_litellm_context(True)
                    try:
                        actual_result = await result
                        process_responses_response(
                            span, actual_result, record_raw_response=is_rollout
                        )
                        return actual_result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
                    finally:
                        reset_in_litellm_context(token)
                        span.end()

                return process_non_streaming_coroutine()

        if kwargs.get("stream"):
            # Check if this is our DualIteratorWrapper - if so, set attributes and return directly
            if isinstance(result, DualIteratorWrapper):
                # Set span attributes directly without consuming the iterator
                result.set_span_attributes(span, record_raw_response=is_rollout)
                streaming_handled = True
                span.end()
                return result
            elif hasattr(result, "__iter__"):
                streaming_handled = True
                return process_responses_streaming_response(
                    span,
                    cast(Generator[Any], result),
                    record_raw_response=is_rollout,
                )
            elif hasattr(result, "__aiter__"):
                streaming_handled = True
                return process_responses_async_streaming_response(
                    span,
                    cast(AsyncGenerator[Any], result),
                    record_raw_response=is_rollout,
                )
            else:
                logger.warning(
                    "Result is not an iterator, but stream is True. This is not supported."
                )
                return result
        else:
            process_responses_response(span, result, record_raw_response=is_rollout)
            return result
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        if not streaming_handled and not returned_coroutine:
            span.end()

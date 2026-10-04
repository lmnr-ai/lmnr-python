from __future__ import annotations

import threading
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence
from types import TracebackType
from typing import Any, TypedDict, cast

from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode
from typing_extensions import Never, NotRequired, Self, TypeVar, override
from wrapt import ObjectProxy

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.shared import (
    is_streaming_response,
    propagate_trace_context,
    set_client_attributes,
    set_request_attributes,
    set_response_attributes,
    set_tools_attributes,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.utils import (
    is_openai_v1,
    should_send_prompts,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    model_as_dict,
    safe_start_span,
    set_span_attribute,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    stamp_instrumentation_scope,
)
from lmnr.opentelemetry_lib.tracing.context import (
    get_event_attributes_from_context,
    is_in_litellm_context,
)
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import JsonValue, json_dumps

SPAN_NAME = "openai.chat"

logger = get_default_logger(__name__)
T = TypeVar("T")


class ChatStreamResponse(TypedDict):
    choices: list[dict[str, Any]]
    model: str
    id: str
    service_tier: str | None
    created: NotRequired[int]
    object: NotRequired[str]
    system_fingerprint: NotRequired[str | None]
    moderation: NotRequired[Any]
    usage: NotRequired[Any]
    prompt_filter_results: NotRequired[Any]


def chat_wrapper(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T | Generator[Any] | ChatStream | None:
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)
    # span needs to be opened and closed manually because the response is a generator

    # LiteLLM calls OpenAI through OpenAI SDK, and to avoid double-instrumentation,
    # we check if we're in a LiteLLM context and return the result directly if so.

    if is_in_litellm_context():
        return wrapped(*args, **kwargs)

    span = safe_start_span(
        name=to_wrap.get("span_name") or SPAN_NAME,
        attributes={"gen_ai.system": "openai"},
        span_type="LLM",
    )

    if span is None:
        return wrapped(*args, **kwargs)

    stamp_instrumentation_scope(span, to_wrap)
    _handle_request(span, kwargs, instance)

    try:
        from lmnr.sdk.debug.replay import replay_enabled

        is_rollout = replay_enabled()
    except Exception:
        is_rollout = False

    try:
        if is_rollout:
            from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.rollout import (
                get_openai_rollout_wrapper,
            )

            rollout_wrapper = get_openai_rollout_wrapper()
            if rollout_wrapper:
                response = rollout_wrapper.wrap_chat_completion(
                    cast(Callable[..., Any], wrapped),  # rollout wrapper is typed stricter than this file
                    instance,
                    args,
                    kwargs,
                    span=span,
                    is_streaming=kwargs.get("stream", False),
                    is_async=False,
                )
            else:
                response = wrapped(*args, **kwargs)
        else:
            response = wrapped(*args, **kwargs)
    except Exception as e:
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        attributes = get_event_attributes_from_context()
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()

        raise

    if is_streaming_response(response):
        if is_openai_v1():
            return ChatStream(
                span,
                response,
                record_raw_response=is_rollout,
            )
        else:
            return _build_from_streaming_response(
                span,
                response,
            )

    _returned_response = _handle_response(
        response,
        span,
        record_raw_response=is_rollout,
    )

    span.end()

    return cast(T, response)


async def achat_wrapper(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[T]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T | AsyncGenerator[Any, Never] | None:
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    if is_in_litellm_context():
        return await wrapped(*args, **kwargs)

    span = safe_start_span(
        name=to_wrap.get("span_name") or SPAN_NAME,
        attributes={"gen_ai.system": "openai"},
        span_type="LLM",
    )

    if span is None:
        return await wrapped(*args, **kwargs)

    stamp_instrumentation_scope(span, to_wrap)
    _handle_request(span, kwargs, instance)

    try:
        from lmnr.sdk.debug.replay import replay_enabled

        is_rollout = replay_enabled()
    except Exception:
        is_rollout = False

    try:
        if is_rollout:
            import inspect

            from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.rollout import (
                get_openai_rollout_wrapper,
            )

            rollout_wrapper = get_openai_rollout_wrapper()
            if rollout_wrapper:
                result = rollout_wrapper.wrap_chat_completion(
                    cast(Callable[..., Any], wrapped),  # rollout wrapper is typed stricter than this file
                    instance,
                    args,
                    kwargs,
                    span=span,
                    is_streaming=kwargs.get("stream", False),
                    is_async=True,
                )
                if inspect.iscoroutine(result):
                    response = await result
                elif inspect.isasyncgen(result):
                    response = result
                else:
                    response = result
            else:
                response = await wrapped(*args, **kwargs)
        else:
            response = await wrapped(*args, **kwargs)
    except Exception as e:
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        attributes = get_event_attributes_from_context()
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()

        raise

    if is_streaming_response(response):
        if is_openai_v1():
            return ChatStream(
                span,
                response,
                record_raw_response=is_rollout,
            )
        else:
            return _abuild_from_streaming_response(
                span,
                response,
            )

    _response = _handle_response(
        response,
        span,
        record_raw_response=is_rollout,
    )

    span.end()

    return cast(T, response)


@dont_throw
def _handle_request(
    span: Span,
    kwargs: dict[str, Any],
    instance: Any,
):
    set_request_attributes(span, kwargs, instance)
    set_client_attributes(span, instance)
    if should_send_prompts():
        _set_prompts(span, kwargs.get("messages"))
        if kwargs.get("functions"):
            set_tools_attributes(span, kwargs.get("functions"))
        elif kwargs.get("tools"):
            set_tools_attributes(span, kwargs.get("tools"))
    propagate_trace_context(span, kwargs)


@dont_throw
def _handle_response(
    response: Any,
    span: Span,
    record_raw_response: bool = False,
):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response

    set_response_attributes(span, response_dict)

    if should_send_prompts():
        _set_completions(span, response_dict.get("choices"))

    if record_raw_response:
        try:
            if hasattr(response, "model_dump_json"):
                set_span_attribute(
                    span, "lmnr.sdk.raw.response", response.model_dump_json()
                )
            else:
                set_span_attribute(
                    span, "lmnr.sdk.raw.response", json_dumps(response_dict)
                )
        except Exception:
            logger.debug("Failed to record raw response", exc_info=True)

    return response


@dont_throw
def _set_prompts(span: Span, messages: list[Any] | None):
    if not span.is_recording() or messages is None:
        return

    processed_messages = []
    for msg in messages:
        msg = msg if isinstance(msg, dict) else model_as_dict(msg)  # pyright: ignore[reportUnknownVariableType]
        processed_msg = dict(msg)  # pyright: ignore[reportUnknownArgumentType]

        if processed_msg.get("tool_calls"):
            processed_msg["tool_calls"] = [
                model_as_dict(tc) for tc in processed_msg["tool_calls"]
            ]

        processed_messages.append(processed_msg)  # pyright: ignore[reportUnknownMemberType]

    set_span_attribute(span, "gen_ai.input.messages", json_dumps(processed_messages))  # pyright: ignore[reportUnknownArgumentType]


def _set_completions(span: Span, choices: list[dict[str, Any]] | None):
    if choices is None:
        return

    set_span_attribute(span, "gen_ai.output.messages", json_dumps(choices))


class ChatStream(ObjectProxy):  # pyright: ignore[reportUntypedBaseClass]
    _record_raw_response: bool = False
    _complete_response: ChatStreamResponse
    _cleanup_completed: bool = False

    @override
    def __init__(
        self,
        span: Span,
        response: Any,
        record_raw_response: bool = False,
    ):
        super().__init__(response)    # pyright: ignore[reportUnknownMemberType]

        self._span: Span = span
        self._record_raw_response = record_raw_response
        self._complete_response = {
            "choices": [],
            "model": "",
            "id": "",
            "service_tier": None,
            "moderation": None,
            "created": 0,
            "system_fingerprint": None,
        }

        self._cleanup_completed = False
        self._cleanup_lock: threading.Lock = threading.Lock()

    def __del__(self):
        """Cleanup when object is garbage collected"""
        if hasattr(self, "_cleanup_completed") and not self._cleanup_completed:
            self._ensure_cleanup()

    def __enter__(self) -> Self:  # pyright: ignore[reportMissingSuperCall]
        return self

    def __exit__(  # pyright: ignore[reportMissingSuperCall]
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        cleanup_exception = None
        try:
            self._ensure_cleanup()
        except Exception as e:
            cleanup_exception = e
            # Don't re-raise to avoid masking original exception

        result: bool | None
        if hasattr(self.__wrapped__, "__exit__"):
            result = self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)
        else:
            result = None

        if cleanup_exception:
            # Log cleanup exception but don't affect context manager behavior
            logger.debug(
                "Error during ChatStream cleanup in __exit__: %s", cleanup_exception
            )

        return result

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if hasattr(self.__wrapped__, "__aexit__"):
            await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)

    def __iter__(self) -> Self:
        return self

    def __aiter__(self) -> Self:
        return self

    def __next__(self) -> Any:
        try:
            chunk = self.__wrapped__.__next__()
        except Exception as e:
            if isinstance(e, StopIteration):
                self._process_complete_response()
            else:
                # Handle cleanup for other exceptions during stream iteration
                self._ensure_cleanup()
                if self._span and self._span.is_recording():
                    self._span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        else:
            self._process_item(chunk)
            return chunk

    async def __anext__(self):
        try:
            chunk = await self.__wrapped__.__anext__()
        except Exception as e:
            if isinstance(e, StopAsyncIteration):
                self._process_complete_response()
            else:
                # Handle cleanup for other exceptions during stream iteration
                self._ensure_cleanup()
                if self._span and self._span.is_recording():
                    self._span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        else:
            self._process_item(chunk)
            return chunk

    def _process_item(self, item: Any):
        self._span.add_event(name="llm.content.completion.chunk")
        self._complete_response["id"] = getattr(item, "id", "")
        self._complete_response["service_tier"] = getattr(item, "service_tier", "")
        self._complete_response["created"] = getattr(item, "created", 0)
        self._complete_response["system_fingerprint"] = getattr(
            item, "system_fingerprint", None
        )
        self._complete_response["moderation"] = getattr(
            item, "moderation", None
        )

        _accumulate_stream_items(item, self._complete_response)

    @dont_throw
    def _process_complete_response(self):
        set_response_attributes(self._span, cast(Any, self._complete_response))
        if should_send_prompts():
            _set_completions(self._span, self._complete_response["choices"])

        if self._record_raw_response:
            try:
                set_span_attribute(
                    self._span,
                    "lmnr.sdk.raw.response",
                    json_dumps(cast(Any, self._complete_response)),
                )
            except Exception:
                logger.debug("Failed to set raw response attribute", exc_info=True)

        self._span.set_status(Status(StatusCode.OK))
        self._span.end()
        self._cleanup_completed = True

    @dont_throw
    def _ensure_cleanup(self):
        """Thread-safe cleanup method that handles different cleanup scenarios"""
        with self._cleanup_lock:
            if self._cleanup_completed:
                logger.debug("ChatStream cleanup already completed, skipping")
                return

            try:
                logger.debug("Starting ChatStream cleanup")

                # Set span status and close it
                if self._span and self._span.is_recording():
                    self._span.set_status(Status(StatusCode.OK))
                    self._span.end()
                    logger.debug("ChatStream span closed successfully")

                self._cleanup_completed = True
                logger.debug("ChatStream cleanup completed successfully")

            except Exception:
                # Log cleanup errors but don't propagate to avoid masking original issues
                logger.debug("Error during ChatStream cleanup", exc_info=True)

                # Still try to close the span even if metrics recording failed
                try:
                    if self._span and self._span.is_recording():
                        self._span.set_status(
                            Status(StatusCode.ERROR, "Cleanup failed")
                        )
                        self._span.end()
                    self._cleanup_completed = True
                except Exception:
                    # Final fallback - just mark as completed to prevent infinite loops
                    self._cleanup_completed = True


# Backward compatibility with OpenAI v0


@dont_throw
def _build_from_streaming_response(
    span: Span,
    response: Any,
) -> Generator[Any]:
    complete_response: ChatStreamResponse = {
        "choices": [],
        "model": "",
        "id": "",
        "service_tier": None,
    }

    for item in response:
        span.add_event(name="llm.content.completion.chunk")

        item_to_yield = item

        _accumulate_stream_items(item, complete_response)

        yield item_to_yield

    set_response_attributes(span, cast(Any, complete_response))
    if should_send_prompts():
        _set_completions(span, complete_response["choices"])

    span.set_status(Status(StatusCode.OK))
    span.end()


@dont_throw
async def _abuild_from_streaming_response(
    span: Span,
    response: Any,
) -> AsyncGenerator[Any]:
    complete_response: ChatStreamResponse = {
        "choices": [],
        "model": "",
        "id": "",
        "service_tier": None,
    }

    async for item in response:
        span.add_event(name="llm.content.completion.chunk")

        item_to_yield = item

        _accumulate_stream_items(item, complete_response)

        yield item_to_yield

    set_response_attributes(span, cast(Any, complete_response))
    if should_send_prompts():
        _set_completions(span, complete_response["choices"])

    span.set_status(Status(StatusCode.OK))
    span.end()


def _accumulate_stream_items(
    item: Any,
    complete_response: ChatStreamResponse,
):
    if is_openai_v1():
        item = model_as_dict(item)

    complete_response["model"] = item.get("model") or ""
    complete_response["id"] = item.get("id") or ""
    complete_response["service_tier"] = item.get("service_tier")
    if item.get("created"):
        complete_response["created"] = item.get("created")
    if "object" not in complete_response:
        complete_response["object"] = "chat.completion"

    # capture usage information from the last stream chunks
    if item.get("usage"):
        complete_response["usage"] = item.get("usage")
    elif item.get("choices") and item["choices"][0].get("usage"):
        # Some LLM providers like moonshot mistakenly place token usage information within choices[0], handle this.
        complete_response["usage"] = item["choices"][0].get("usage")

    # prompt filter results
    if item.get("prompt_filter_results"):
        complete_response["prompt_filter_results"] = item.get("prompt_filter_results")

    for choice in item.get("choices") or []:  # pyright: ignore[reportUnknownVariableType]
        index = choice.get("index")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if len(complete_response["choices"]) <= index:
            complete_response["choices"].append(
                {"index": index, "message": {"content": "", "role": ""}}
            )
        complete_choice = complete_response["choices"][index]  # pyright: ignore[reportUnknownVariableType]
        if choice.get("finish_reason"):  # pyright: ignore[reportUnknownMemberType]
            complete_choice["finish_reason"] = choice.get("finish_reason")  # pyright: ignore[reportUnknownMemberType]
        if choice.get("content_filter_results"):  # pyright: ignore[reportUnknownMemberType]
            complete_choice["content_filter_results"] = choice.get(  # pyright: ignore[reportUnknownMemberType]
                "content_filter_results"
            )

        delta = choice.get("delta")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        if delta and delta.get("content"):  # pyright: ignore[reportUnknownMemberType]
            complete_choice["message"]["content"] += delta.get("content")  # pyright: ignore[reportUnknownMemberType]

        if delta and delta.get("role"):  # pyright: ignore[reportUnknownMemberType]
            complete_choice["message"]["role"] = delta.get("role")  # pyright: ignore[reportUnknownMemberType]
        if delta and delta.get("tool_calls"):  # pyright: ignore[reportUnknownMemberType]
            tool_calls = delta.get("tool_calls")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if not isinstance(tool_calls, list) or len(tool_calls) == 0:   # pyright: ignore[reportUnknownArgumentType]
                continue

            if not complete_choice["message"].get("tool_calls"):  # pyright: ignore[reportUnknownMemberType]
                complete_choice["message"]["tool_calls"] = []

            for tool_call in tool_calls:  # pyright: ignore[reportUnknownVariableType]
                i = int(tool_call["index"])  # pyright: ignore[reportUnknownArgumentType]
                if len(complete_choice["message"]["tool_calls"]) <= i:  # pyright: ignore[reportUnknownArgumentType]
                    complete_choice["message"]["tool_calls"].append(  # pyright: ignore[reportUnknownMemberType]
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    )

                span_tool_call = complete_choice["message"]["tool_calls"][i]  # pyright: ignore[reportUnknownVariableType]
                span_function = span_tool_call["function"]  # pyright: ignore[reportUnknownVariableType]
                tool_call_function = tool_call.get("function")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

                if tool_call.get("id"):  # pyright: ignore[reportUnknownMemberType]
                    span_tool_call["id"] = tool_call.get("id")  # pyright: ignore[reportUnknownMemberType]
                if tool_call_function and tool_call_function.get("name"):  # pyright: ignore[reportUnknownMemberType]
                    span_function["name"] = tool_call_function.get("name")  # pyright: ignore[reportUnknownMemberType]
                if tool_call_function and tool_call_function.get("arguments"):  # pyright: ignore[reportUnknownMemberType]
                    span_function["arguments"] += tool_call_function.get("arguments")  # pyright: ignore[reportUnknownMemberType]

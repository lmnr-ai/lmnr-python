import logging
import threading

from opentelemetry import context as context_api
from ..shared import (
    _set_client_attributes,
    _set_functions_attributes,
    _set_request_attributes,
    _set_response_attributes,
    _set_span_attribute,
    is_streaming_response,
    model_as_dict,
    propagate_trace_context,
    set_tools_attributes,
)
from lmnr.sdk.utils import json_dumps
from ..shared.config import Config
from ..utils import (
    _with_chat_telemetry_wrapper,
    dont_throw,
    is_openai_v1,
    should_send_prompts,
)
from lmnr.opentelemetry_lib.tracing.context import (
    get_event_attributes_from_context,
    is_in_litellm_context,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    safe_start_span,
)
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import ObjectProxy

SPAN_NAME = "openai.chat"

logger = logging.getLogger(__name__)


@_with_chat_telemetry_wrapper
def chat_wrapper(
    tracer: Tracer,
    wrapped,
    instance,
    args,
    kwargs,
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)
    # span needs to be opened and closed manually because the response is a generator

    # LiteLLM calls OpenAI through OpenAI SDK, and to avoid double-instrumentation,
    # we check if we're in a LiteLLM context and return the result directly if so.

    if is_in_litellm_context():
        return wrapped(*args, **kwargs)

    span = safe_start_span(
        name=SPAN_NAME, attributes={"gen_ai.system": "openai"}, span_type="LLM"
    )

    if span is None:
        return wrapped(*args, **kwargs)

    _handle_request(span, kwargs, instance)

    try:
        from lmnr.sdk.rollout_control import is_rollout_mode

        is_rollout = is_rollout_mode()
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

    _handle_response(
        response,
        span,
        record_raw_response=is_rollout,
    )

    span.end()

    return response


@_with_chat_telemetry_wrapper
async def achat_wrapper(
    tracer: Tracer,
    wrapped,
    instance,
    args,
    kwargs,
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    if is_in_litellm_context():
        return await wrapped(*args, **kwargs)

    span = safe_start_span(
        name=SPAN_NAME, attributes={"gen_ai.system": "openai"}, span_type="LLM"
    )

    if span is None:
        return await wrapped(*args, **kwargs)

    _handle_request(span, kwargs, instance)

    try:
        from lmnr.sdk.rollout_control import is_rollout_mode

        is_rollout = is_rollout_mode()
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
                    wrapped,
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

    _handle_response(
        response,
        span,
        record_raw_response=is_rollout,
    )

    span.end()

    return response


@dont_throw
def _handle_request(span, kwargs, instance):
    _set_request_attributes(span, kwargs, instance)
    _set_client_attributes(span, instance)
    if should_send_prompts():
        _set_prompts(span, kwargs.get("messages"))
        if kwargs.get("functions"):
            _set_functions_attributes(span, kwargs.get("functions"))
        elif kwargs.get("tools"):
            set_tools_attributes(span, kwargs.get("tools"))
    if Config.enable_trace_context_propagation:
        propagate_trace_context(span, kwargs)


@dont_throw
def _handle_response(
    response,
    span,
    record_raw_response=False,
):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response

    _set_response_attributes(span, response_dict)

    if should_send_prompts():
        _set_completions(span, response_dict.get("choices"))

    if record_raw_response:
        try:
            if hasattr(response, "model_dump_json"):
                _set_span_attribute(
                    span, "lmnr.sdk.raw.response", response.model_dump_json()
                )
            else:
                _set_span_attribute(
                    span, "lmnr.sdk.raw.response", json_dumps(response_dict)
                )
        except Exception:
            pass

    return response


def _set_choice_counter_metrics(choice_counter, choices, shared_attributes):
    choice_counter.add(len(choices), attributes=shared_attributes)
    for choice in choices:
        attributes_with_reason = {**shared_attributes}
        if choice.get("finish_reason"):
            attributes_with_reason["llm.response.finish_reason"] = choice.get(
                "finish_reason"
            )
        choice_counter.add(1, attributes=attributes_with_reason)


@dont_throw
def _set_prompts(span, messages):
    if not span.is_recording() or messages is None:
        return

    processed_messages = []
    for msg in messages:
        msg = msg if isinstance(msg, dict) else model_as_dict(msg)
        processed_msg = dict(msg)

        if processed_msg.get("tool_calls"):
            processed_msg["tool_calls"] = [
                model_as_dict(tc) for tc in processed_msg["tool_calls"]
            ]

        processed_messages.append(processed_msg)

    _set_span_attribute(span, "gen_ai.input.messages", json_dumps(processed_messages))


def _set_completions(span, choices):
    if choices is None:
        return

    _set_span_attribute(span, "gen_ai.output.messages", json_dumps(choices))


class ChatStream(ObjectProxy):
    _span = None
    _record_raw_response = False
    _complete_response = None
    _cleanup_completed = False
    _cleanup_lock = None

    def __init__(
        self,
        span,
        response,
        record_raw_response=False,
    ):
        super().__init__(response)

        self._span = span
        self._record_raw_response = record_raw_response
        self._complete_response = {
            "choices": [],
            "model": "",
            "id": "",
            "service_tier": None,
        }

        self._cleanup_completed = False
        self._cleanup_lock = threading.Lock()

    def __del__(self):
        """Cleanup when object is garbage collected"""
        if hasattr(self, "_cleanup_completed") and not self._cleanup_completed:
            self._ensure_cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_exception = None
        try:
            self._ensure_cleanup()
        except Exception as e:
            cleanup_exception = e
            # Don't re-raise to avoid masking original exception

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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.__wrapped__, "__aexit__"):
            await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def __next__(self):
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

    def _process_item(self, item):
        self._span.add_event(name="llm.content.completion.chunk")
        self._complete_response["id"] = item.id if hasattr(item, "id") else ""
        self._complete_response["service_tier"] = (
            item.service_tier if hasattr(item, "service_tier") else ""
        )

        _accumulate_stream_items(item, self._complete_response)

    @dont_throw
    def _process_complete_response(self):

        _set_response_attributes(self._span, self._complete_response)
        if should_send_prompts():
            _set_completions(self._span, self._complete_response.get("choices"))

        if self._record_raw_response:
            try:
                _set_span_attribute(
                    self._span,
                    "lmnr.sdk.raw.response",
                    json_dumps(self._complete_response),
                )
            except Exception:
                pass

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

            except Exception as e:
                # Log cleanup errors but don't propagate to avoid masking original issues
                logger.debug("Error during ChatStream cleanup: %s", str(e))

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
    span,
    response,
):
    complete_response = {"choices": [], "model": "", "id": "", "service_tier": None}

    for item in response:
        span.add_event(name="llm.content.completion.chunk")

        item_to_yield = item

        _accumulate_stream_items(item, complete_response)

        yield item_to_yield

    _set_response_attributes(span, complete_response)
    if should_send_prompts():
        _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()


@dont_throw
async def _abuild_from_streaming_response(
    span,
    response,
):
    complete_response = {"choices": [], "model": "", "id": "", "service_tier": None}

    async for item in response:
        span.add_event(name="llm.content.completion.chunk")

        item_to_yield = item

        _accumulate_stream_items(item, complete_response)

        yield item_to_yield

    _set_response_attributes(span, complete_response)
    if should_send_prompts():
        _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()


def _accumulate_stream_items(item, complete_response):
    if is_openai_v1():
        item = model_as_dict(item)

    complete_response["model"] = item.get("model")
    complete_response["id"] = item.get("id")
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

    for choice in item.get("choices") or []:
        index = choice.get("index")
        if len(complete_response.get("choices")) <= index:
            complete_response["choices"].append(
                {"index": index, "message": {"content": "", "role": ""}}
            )
        complete_choice = complete_response.get("choices")[index]
        if choice.get("finish_reason"):
            complete_choice["finish_reason"] = choice.get("finish_reason")
        if choice.get("content_filter_results"):
            complete_choice["content_filter_results"] = choice.get(
                "content_filter_results"
            )

        delta = choice.get("delta")

        if delta and delta.get("content"):
            complete_choice["message"]["content"] += delta.get("content")

        if delta and delta.get("role"):
            complete_choice["message"]["role"] = delta.get("role")
        if delta and delta.get("tool_calls"):
            tool_calls = delta.get("tool_calls")
            if not isinstance(tool_calls, list) or len(tool_calls) == 0:
                continue

            if not complete_choice["message"].get("tool_calls"):
                complete_choice["message"]["tool_calls"] = []

            for tool_call in tool_calls:
                i = int(tool_call["index"])
                if len(complete_choice["message"]["tool_calls"]) <= i:
                    complete_choice["message"]["tool_calls"].append(
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    )

                span_tool_call = complete_choice["message"]["tool_calls"][i]
                span_function = span_tool_call["function"]
                tool_call_function = tool_call.get("function")

                if tool_call.get("id"):
                    span_tool_call["id"] = tool_call.get("id")
                if tool_call_function and tool_call_function.get("name"):
                    span_function["name"] = tool_call_function.get("name")
                if tool_call_function and tool_call_function.get("arguments"):
                    span_function["arguments"] += tool_call_function.get("arguments")

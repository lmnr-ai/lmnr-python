import logging

from .config import Config
from .span_utils import (
    set_streaming_response_attributes,
)
from .utils import (
    dont_throw,
    set_span_attribute,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_RESPONSE_ID,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import Status, StatusCode

logger = logging.getLogger(__name__)


@dont_throw
def _process_response_item(item, complete_response):
    if item.type == "message_start":
        complete_response["model"] = item.message.model
        usage = dict(item.message.usage)
        complete_response["usage"] = usage
        complete_response["service_tier"] = usage.get("service_tier") or None
        complete_response["id"] = item.message.id
    elif item.type == "content_block_start":
        index = item.index
        if len(complete_response.get("events")) <= index:
            complete_response["events"].append(
                {"index": index, "text": "", "type": item.content_block.type}
            )
            if item.content_block.type == "tool_use":
                complete_response["events"][index]["id"] = item.content_block.id
                complete_response["events"][index]["name"] = item.content_block.name
                complete_response["events"][index]["input"] = ""

    elif item.type == "content_block_delta":
        index = item.index
        if item.delta.type == "thinking_delta":
            complete_response["events"][index]["text"] += item.delta.thinking or ""
        elif item.delta.type == "text_delta":
            complete_response["events"][index]["text"] += item.delta.text or ""
        elif item.delta.type == "input_json_delta":
            complete_response["events"][index]["input"] += item.delta.partial_json
    elif item.type == "message_delta":
        for event in complete_response.get("events", []):
            event["finish_reason"] = item.delta.stop_reason
        if item.usage:
            if "usage" in complete_response:
                item_output_tokens = dict(item.usage).get("output_tokens", 0)
                existing_output_tokens = complete_response["usage"].get(
                    "output_tokens", 0
                )
                complete_response["usage"]["output_tokens"] = (
                    item_output_tokens + existing_output_tokens
                )
            else:
                complete_response["usage"] = dict(item.usage)
    elif item.type in ["message_stop", "message_start"]:
        # raw stream returns the service_tier in the message_start event
        # messages.stream returns the service_tier in the message_stop event
        usage = dict(item.message.usage or {})
        complete_response["service_tier"] = usage.get("service_tier")


def _set_token_usage(
    span,
    complete_response,
    prompt_tokens,
    completion_tokens,
):
    cache_read_tokens = (
        complete_response.get("usage", {}).get("cache_read_input_tokens", 0) or 0
    )
    cache_creation_tokens = (
        complete_response.get("usage", {}).get("cache_creation_input_tokens", 0) or 0
    )

    input_tokens = prompt_tokens + cache_read_tokens + cache_creation_tokens
    total_tokens = input_tokens + completion_tokens

    set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)
    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens
    )
    set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    set_span_attribute(
        span, SpanAttributes.LLM_RESPONSE_MODEL, complete_response.get("model")
    )
    set_span_attribute(
        span, SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cache_read_tokens
    )
    set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
        cache_creation_tokens,
    )


def _handle_streaming_response(span, complete_response):
    if not span.is_recording():
        return
    set_streaming_response_attributes(span, complete_response.get("events"))


@dont_throw
def build_from_streaming_response(
    span,
    response,
    instance,
    kwargs: dict = {},
):
    complete_response = {
        "events": [],
        "model": "",
        "usage": {},
        "id": "",
        "service_tier": None,
    }

    for item in response:
        try:
            yield item
        except Exception as e:
            raise e
        _process_response_item(item, complete_response)

    set_span_attribute(span, GEN_AI_RESPONSE_ID, complete_response.get("id"))
    set_span_attribute(
        span,
        "anthropic.response.service_tier",
        complete_response.get("service_tier"),
    )
    # calculate token usage
    if Config.enrich_token_usage:
        try:
            completion_tokens = -1
            # prompt_usage
            if usage := complete_response.get("usage"):
                prompt_tokens = usage.get("input_tokens", 0) or 0
            else:
                prompt_tokens = 0

            # completion_usage
            if usage := complete_response.get("usage"):
                completion_tokens = usage.get("output_tokens", 0) or 0
            else:
                completion_content = ""
                if complete_response.get("events"):
                    model_name = complete_response.get("model") or None
                    for event in complete_response.get("events"):
                        if event.get("text"):
                            completion_content += event.get("text")

                    if model_name and hasattr(instance, "count_tokens"):
                        completion_tokens = instance.count_tokens(completion_content)

            _set_token_usage(
                span,
                complete_response,
                prompt_tokens,
                completion_tokens,
            )
        except Exception as e:
            logger.warning("Failed to set token usage, error: %s", e)

    _handle_streaming_response(span, complete_response)

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))
        span.end()


@dont_throw
async def abuild_from_streaming_response(
    span,
    response,
    instance,
    kwargs: dict = {},
):
    complete_response = {
        "events": [],
        "model": "",
        "usage": {},
        "id": "",
        "service_tier": None,
    }
    async for item in response:
        try:
            yield item
        except Exception as e:
            raise e
        _process_response_item(item, complete_response)

    set_span_attribute(span, GEN_AI_RESPONSE_ID, complete_response.get("id"))
    set_span_attribute(
        span,
        "anthropic.response.service_tier",
        complete_response.get("service_tier"),
    )

    # calculate token usage
    if Config.enrich_token_usage:
        try:
            # prompt_usage
            if usage := complete_response.get("usage"):
                prompt_tokens = usage.get("input_tokens", 0)
            else:
                prompt_tokens = 0

            # completion_usage
            if usage := complete_response.get("usage"):
                completion_tokens = usage.get("output_tokens", 0)
            else:
                completion_content = ""
                if complete_response.get("events"):
                    model_name = complete_response.get("model") or None
                    for event in complete_response.get("events"):
                        if event.get("text"):
                            completion_content += event.get("text")

                    if model_name and hasattr(instance, "count_tokens"):
                        completion_tokens = instance.count_tokens(completion_content)

            _set_token_usage(
                span,
                complete_response,
                prompt_tokens,
                completion_tokens,
            )
        except Exception as e:
            logger.warning("Failed to set token usage, error: %s", str(e))

    _handle_streaming_response(span, complete_response)

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))
        span.end()


class WrappedMessageStreamManager:
    """Wrapper for MessageStreamManager that handles instrumentation"""

    def __init__(
        self,
        stream_manager,
        span,
        instance,
        kwargs,
    ):
        self._stream_manager = stream_manager
        self._span = span
        self._instance = instance
        self._kwargs = kwargs

    def __enter__(self):
        # Call the original stream manager's __enter__ to get the actual stream
        stream = self._stream_manager.__enter__()
        # Return the wrapped stream
        return build_from_streaming_response(
            self._span,
            stream,
            self._instance,
            self._kwargs,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._stream_manager.__exit__(exc_type, exc_val, exc_tb)


class WrappedAsyncMessageStreamManager:
    """Wrapper for AsyncMessageStreamManager that handles instrumentation"""

    def __init__(
        self,
        stream_manager,
        span,
        instance,
        kwargs,
    ):
        self._stream_manager = stream_manager
        self._span = span
        self._instance = instance
        self._kwargs = kwargs

    async def __aenter__(self):
        # Call the original stream manager's __aenter__ to get the actual stream
        stream = await self._stream_manager.__aenter__()
        # Return the wrapped stream
        return abuild_from_streaming_response(
            self._span,
            stream,
            self._instance,
            self._kwargs,
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._stream_manager.__aexit__(exc_type, exc_val, exc_tb)

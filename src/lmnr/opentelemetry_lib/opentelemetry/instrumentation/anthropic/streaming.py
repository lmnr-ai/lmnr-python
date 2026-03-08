import logging

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.trace.status import Status, StatusCode

from lmnr.sdk.utils import json_dumps

from .config import Config
from .span_utils import (
    set_streaming_response_attributes,
)
from .utils import (
    dont_throw,
    set_span_attribute,
)

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
            # message_delta usage values are cumulative (per Anthropic docs),
            # so we update/replace rather than add to existing values.
            # Filter out None values to avoid overwriting message_start data
            # (e.g. cache_creation_input_tokens) with None from message_delta.
            usage_update = {
                k: v for k, v in dict(item.usage).items() if v is not None
            }
            if "usage" in complete_response:
                complete_response["usage"].update(usage_update)
            else:
                complete_response["usage"] = usage_update
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

    set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)

    set_span_attribute(span, GEN_AI_RESPONSE_MODEL, complete_response.get("model"))
    set_span_attribute(span, "gen_ai.usage.cache_read_input_tokens", cache_read_tokens)
    set_span_attribute(
        span,
        "gen_ai.usage.cache_creation_input_tokens",
        cache_creation_tokens,
    )


def _handle_streaming_response(span, complete_response, record_raw_response=False):
    if not span.is_recording():
        return
    result = set_streaming_response_attributes(span, complete_response.get("events"))

    if record_raw_response and result:
        try:
            # Enrich the result with static attributes
            result["id"] = complete_response.get("id")
            result["model"] = complete_response.get("model")
            result["type"] = "message"
            result["usage"] = complete_response.get("usage") or {
                "input_tokens": 0,
                "output_tokens": 0,
            }

            set_span_attribute(span, "lmnr.sdk.raw.response", json_dumps(result))
        except Exception:
            pass


@dont_throw
def build_from_streaming_response(
    span,
    response,
    instance,
    kwargs: dict = {},
    record_raw_response: bool = False,
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

    try:
        usage = complete_response.get("usage")
        prompt_tokens = (usage.get("input_tokens", 0) or 0) if usage else 0
        completion_tokens = (usage.get("output_tokens", 0) or 0) if usage else 0

        if not usage and Config.enrich_token_usage:
            completion_content = ""
            if complete_response.get("events"):
                model_name = complete_response.get("model") or None
                for event in complete_response.get("events") or []:
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

    _handle_streaming_response(span, complete_response, record_raw_response)

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))
        span.end()


@dont_throw
async def abuild_from_streaming_response(
    span,
    response,
    instance,
    kwargs: dict = {},
    record_raw_response: bool = False,
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

    try:
        usage = complete_response.get("usage")
        prompt_tokens = (usage.get("input_tokens", 0) or 0) if usage else 0
        completion_tokens = (usage.get("output_tokens", 0) or 0) if usage else 0

        if not usage and Config.enrich_token_usage:
            completion_content = ""
            if complete_response.get("events"):
                model_name = complete_response.get("model") or None
                for event in complete_response.get("events") or []:
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

    _handle_streaming_response(span, complete_response, record_raw_response)

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
        record_raw_response: bool = False,
    ):
        self._stream_manager = stream_manager
        self._span = span
        self._instance = instance
        self._kwargs = kwargs
        self._record_raw_response = record_raw_response

    def __enter__(self):
        # Call the original stream manager's __enter__ to get the actual stream
        stream = self._stream_manager.__enter__()
        # Return the wrapped stream
        return build_from_streaming_response(
            self._span,
            stream,
            self._instance,
            self._kwargs,
            record_raw_response=self._record_raw_response,
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
        record_raw_response: bool = False,
    ):
        self._stream_manager = stream_manager
        self._span = span
        self._instance = instance
        self._kwargs = kwargs
        self._record_raw_response = record_raw_response

    async def __aenter__(self):
        # Call the original stream manager's __aenter__ to get the actual stream
        stream = await self._stream_manager.__aenter__()
        # Return the wrapped stream
        return abuild_from_streaming_response(
            self._span,
            stream,
            self._instance,
            self._kwargs,
            record_raw_response=self._record_raw_response,
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._stream_manager.__aexit__(exc_type, exc_val, exc_tb)

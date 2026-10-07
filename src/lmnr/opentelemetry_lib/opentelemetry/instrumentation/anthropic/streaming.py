from collections.abc import AsyncGenerator, Generator
from types import TracebackType
from typing import Any, cast

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.trace.span import Span
from opentelemetry.trace.status import Status, StatusCode

from anthropic.lib.streaming import (
    AsyncMessageStream,
    AsyncMessageStreamManager,
    MessageStream,
    MessageStreamManager,
    ParsedMessageStreamEvent,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic.event_models import (
    AnthropicUsage,
    CompleteResponse,
    StreamUsage,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic.span_utils import (
    set_streaming_response_attributes,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    set_span_attribute,
)
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import json_dumps

logger = get_default_logger(__name__)


def _new_complete_response() -> CompleteResponse:
    return {
        "events": [],
        "model": "",
        "usage": {},
        "id": "",
        "service_tier": None,
    }


@dont_throw
def _process_response_item(
    item: Any,
    complete_response: CompleteResponse,
):
    if item.type == "message_start":
        complete_response["model"] = item.message.model
        usage = cast(StreamUsage, cast(object, dict(item.message.usage)))
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
            complete_response["usage"].update(cast(StreamUsage, cast(object, usage_update)))
    elif item.type in ["message_stop", "message_start"]:
        # raw stream returns the service_tier in the message_start event
        # messages.stream returns the service_tier in the message_stop event
        usage = dict(item.message.usage or {})
        complete_response["service_tier"] = usage.get("service_tier")


def _set_token_usage(
    span: Span,
    complete_response: CompleteResponse,
    prompt_tokens: int,
    completion_tokens: int,
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


def _handle_streaming_response(
    span: Span,
    complete_response: CompleteResponse,
    record_raw_response: bool = False
):
    if not span.is_recording():
        return
    result = set_streaming_response_attributes(span, complete_response.get("events"))

    if record_raw_response and result:
        try:
            # Enrich the result with static attributes
            result["id"] = complete_response.get("id")
            result["model"] = complete_response.get("model")
            result["type"] = "message"
            result["usage"] = cast(
                AnthropicUsage,
                cast(
                    object,
                    complete_response.get("usage")
                    or {"input_tokens": 0, "output_tokens": 0},
                ),
            )

            set_span_attribute(
                span, "lmnr.sdk.raw.response", json_dumps(dict(result))
            )
        except Exception:
            logger.debug("Failed to record raw response", exc_info=True)


@dont_throw
def build_from_streaming_response(
    span: Span,
    response: MessageStream,
    _instance: Any,
    _kwargs: dict[str, Any],
    record_raw_response: bool = False,
) -> Generator[ParsedMessageStreamEvent]:
    complete_response = _new_complete_response()

    for item in response:
        yield item
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

        _set_token_usage(
            span,
            complete_response,
            prompt_tokens,
            completion_tokens,
        )
    except Exception:
        logger.warning("Failed to set token usage", exc_info=True)

    _handle_streaming_response(span, complete_response, record_raw_response)

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))
        span.end()


@dont_throw
async def abuild_from_streaming_response(
    span: Span,
    response: AsyncMessageStream,
    _instance: Any,
    _kwargs: dict[str, Any],
    record_raw_response: bool = False,
) -> AsyncGenerator[ParsedMessageStreamEvent]:
    complete_response = _new_complete_response()
    async for item in response:
        yield item
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

        _set_token_usage(
            span,
            complete_response,
            prompt_tokens,
            completion_tokens,
        )
    except Exception:
        logger.warning("Failed to set token usage", exc_info=True)

    _handle_streaming_response(span, complete_response, record_raw_response)

    if span.is_recording():
        span.set_status(Status(StatusCode.OK))
        span.end()


class WrappedMessageStreamManager:
    """Wrapper for MessageStreamManager that handles instrumentation"""

    def __init__(
        self,
        stream_manager: MessageStreamManager,
        span: Span,
        instance: Any,
        kwargs: dict[str, Any],
        record_raw_response: bool = False,
    ):
        self._stream_manager: MessageStreamManager = stream_manager
        self._span: Span = span
        self._instance: Any = instance
        self._kwargs: dict[str, Any] = kwargs
        self._record_raw_response: bool = record_raw_response

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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        return self._stream_manager.__exit__(exc_type, exc_val, exc_tb)


class WrappedAsyncMessageStreamManager:
    """Wrapper for AsyncMessageStreamManager that handles instrumentation"""

    def __init__(
        self,
        stream_manager: AsyncMessageStreamManager,
        span: Span,
        instance: Any,
        kwargs: dict[str, Any],
        record_raw_response: bool = False,
    ):
        self._stream_manager: AsyncMessageStreamManager = stream_manager
        self._span: Span = span
        self._instance: Any = instance
        self._kwargs: dict[str, Any] = kwargs
        self._record_raw_response: bool = record_raw_response

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

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        return await self._stream_manager.__aexit__(exc_type, exc_val, exc_tb)

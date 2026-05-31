"""
Debug replay wrapper for Anthropic chat completions instrumentation.

On a debug run with replay enabled, this serves the first N spine LLM calls from
the in-process `ReplayCache` (reconstructing an Anthropic `Message` from cached
attributes) and runs the rest live. Override application and the old HTTP cache
server are gone (§2 non-goals, §H).
"""

import json
from typing import Any, AsyncGenerator, Generator

from opentelemetry.trace import Span

from anthropic.types import (
    Message,
    RawMessageStreamEvent,
    InputJSONDelta,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextDelta,
    ThinkingDelta,
    Usage,
)

from anthropic.types.raw_message_delta_event import Delta
from lmnr.sdk.debug.replay import (
    cached_payload_for,
    mark_span_cached,
    replay_enabled,
    span_path_from_span,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class AnthropicRolloutWrapper:
    """Serves cached Anthropic responses on a debug run; runs live otherwise."""

    def cached_response_to_anthropic(
        self, cached_span: dict[str, Any]
    ) -> Message | None:
        """
        Convert cached span data to Anthropic Message response.
        Tries raw response first, then falls back to reconstruction.
        """
        attributes = cached_span.get("attributes", {})
        if raw_response := attributes.get("lmnr.sdk.raw.response"):
            try:
                response_dict = (
                    raw_response
                    if isinstance(raw_response, dict)
                    else json.loads(raw_response)
                )
                if response_dict:
                    # Reset usage to 0 tokens/cost for cached responses
                    if "usage" in response_dict:
                        response_dict["usage"] = {
                            "input_tokens": 0,
                            "output_tokens": 0,
                        }
                    return Message.model_validate(response_dict)
            except Exception as e:
                logger.debug(f"Failed to parse raw response: {e}")

        try:
            output_str = attributes.get("gen_ai.output.messages")
            if not output_str:
                logger.warning("Cached span has no output messages")
                return None

            output = json.loads(output_str)
            if not isinstance(output, list) or not output:
                return None

            # Anthropic instrumentation saves output as a list of choices, usually just one
            first_choice = output[0]
            content_blocks = first_choice.get("content", [])

            return Message(
                id=attributes.get("gen_ai.response.id", "cached"),
                model=attributes.get("gen_ai.response.model", "unknown"),
                content=content_blocks,
                role="assistant",
                stop_reason=first_choice.get("stop_reason", "end_turn"),
                type="message",
                usage=Usage(input_tokens=0, output_tokens=0),
            )

        except Exception as e:
            logger.debug(
                f"Failed to convert cached response to Anthropic format: {e}",
                exc_info=True,
            )
            return None

    def wrap_create(
        self,
        wrapped,
        instance,
        args,
        kwargs,
        span: Span | None = None,
        is_streaming: bool = False,
        is_async: bool = False,
    ) -> Any:
        span_path = span_path_from_span(span)
        cached = cached_payload_for(span_path)
        if cached is not None:
            response = self.cached_response_to_anthropic(cached)
            if response is not None:
                logger.debug("Replaying cached Anthropic response at %s", span_path)
                mark_span_cached(span)
                if is_streaming:
                    if is_async:
                        return self._as_coroutine(
                            self._create_async_cached_stream(response)
                        )
                    return self._create_cached_stream(response)
                if is_async:
                    return self._as_coroutine(response)
                return response

        return wrapped(*args, **kwargs)

    def _create_cached_stream(
        self, response: Message
    ) -> Generator[RawMessageStreamEvent, None, None]:
        """Yield a cached response as a sequence of streaming events."""

        # 1. Message Start
        yield RawMessageStartEvent(
            message=Message(
                id=response.id,
                model=response.model,
                role=response.role,
                content=[],  # Content is empty in message_start
                type="message",
                usage=Usage(input_tokens=0, output_tokens=0),
            ),
            type="message_start",
        )

        # 2. Content Blocks
        for i, block in enumerate(response.content):
            # Content Block Start
            yield RawContentBlockStartEvent(
                content_block=block,
                index=i,
                type="content_block_start",
            )

            # Content Block Delta (send entire content as one delta)
            delta_obj = None
            if block.type == "text":
                delta_obj = TextDelta(text=block.text, type="text_delta")
            elif block.type == "tool_use":
                # For tool use, we send the whole input as one delta
                input_json = (
                    json.dumps(block.input)
                    if not isinstance(block.input, str)
                    else block.input
                )
                delta_obj = InputJSONDelta(
                    partial_json=input_json, type="input_json_delta"
                )
            elif block.type == "thinking":
                delta_obj = ThinkingDelta(
                    thinking=block.thinking, type="thinking_delta"
                )

            if delta_obj:
                yield RawContentBlockDeltaEvent(
                    delta=delta_obj,
                    index=i,
                    type="content_block_delta",
                )

            # Content Block Stop
            yield RawContentBlockStopEvent(
                index=i,
                type="content_block_stop",
            )

        # 3. Message Delta
        yield RawMessageDeltaEvent(
            delta=Delta(
                stop_reason=response.stop_reason,
                stop_sequence=response.stop_sequence,
            ),
            type="message_delta",
            usage=MessageDeltaUsage(output_tokens=0),
        )

        # 4. Message Stop
        yield RawMessageStopEvent(type="message_stop")

    async def _create_async_cached_stream(
        self, response: Message
    ) -> AsyncGenerator[RawMessageStreamEvent, None]:
        """Yield a cached response as a sequence of async streaming events."""
        for event in self._create_cached_stream(response):
            yield event

    async def _as_coroutine(self, response: Any) -> Any:
        """Return a response as a coroutine."""
        return response


_anthropic_rollout_wrapper: AnthropicRolloutWrapper | None = None


def get_anthropic_rollout_wrapper() -> AnthropicRolloutWrapper | None:
    global _anthropic_rollout_wrapper

    if not replay_enabled():
        return None

    if _anthropic_rollout_wrapper is None:
        try:
            _anthropic_rollout_wrapper = AnthropicRolloutWrapper()
        except Exception as e:
            logger.error(f"Failed to create Anthropic replay wrapper: {e}")
            return None

    return _anthropic_rollout_wrapper

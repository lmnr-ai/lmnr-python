"""
Debug replay wrapper for Anthropic chat completions instrumentation.

On a debug run with replay configured, each live LLM call asks the server-side
cache (shared spec §7) what to do: on a HIT it reconstructs an Anthropic
`Message` from the cached span and serves it; on MISS / LIVE it runs the call
live. The decision is made by `cache_outcome_for` (sync) / `acache_outcome_for`
(async); there is no in-process cache anymore.
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
    acache_outcome_for,
    cache_outcome_for,
    mark_span_cached,
    replay_enabled,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class AnthropicRolloutWrapper:
    """Serves cached Anthropic responses on a debug run; runs live otherwise."""

    def cached_response_to_anthropic(
        self, cached_span: dict[str, Any]
    ) -> Message | None:
        """Convert cached span envelope to Anthropic Message response."""
        envelope_type = cached_span.get("type")
        if envelope_type not in ("raw", "genAi"):
            logger.warning(f"Unknown cached span type: {envelope_type!r}")
            return None

        if envelope_type == "raw":
            try:
                raw = cached_span.get("response")
                if not raw:
                    logger.warning("Cached span type='raw' has no response field")
                    return None
                response_dict = raw if isinstance(raw, dict) else json.loads(raw)
                if "usage" in response_dict:
                    response_dict["usage"] = {"input_tokens": 0, "output_tokens": 0}
                return Message.model_validate(response_dict)
            except Exception as e:
                logger.debug(f"Failed to parse raw Anthropic response: {e}", exc_info=True)
                return None

        # envelope_type == "genAi"
        try:
            messages = cached_span.get("messages")
            if not isinstance(messages, list) or not messages:
                logger.warning("Cached span type='genAi' has no messages")
                return None
            model = cached_span.get("model", "unknown")
            finish_reasons = cached_span.get("finishReasons", [])
            stop_reason = finish_reasons[0] if finish_reasons else "end_turn"
            content_blocks = messages[0].get("content", [])
            return Message(
                id="cached",
                model=model,
                content=content_blocks,
                role="assistant",
                stop_reason=stop_reason,
                type="message",
                usage=Usage(input_tokens=0, output_tokens=0),
            )
        except Exception as e:
            logger.debug(
                f"Failed to convert genAi response to Anthropic format: {e}",
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
        # The async call site always awaits the result, so on the async path
        # return the coroutine directly (it does its own cache lookup + await).
        if is_async:
            return self._awrap_create(wrapped, args, kwargs, span, is_streaming)

        outcome = cache_outcome_for(span)
        if outcome is not None and outcome.kind == "hit":
            response = self.cached_response_to_anthropic(outcome.cached)
            if response is not None:
                logger.debug("Serving cached Anthropic response from replay cache")
                mark_span_cached(span)
                if is_streaming:
                    return self._create_cached_stream(response)
                return response

        return wrapped(*args, **kwargs)

    async def _awrap_create(self, wrapped, args, kwargs, span, is_streaming) -> Any:
        """Async cache lookup; on a HIT serve cached, else run the call live."""
        outcome = await acache_outcome_for(span)
        if outcome is not None and outcome.kind == "hit":
            response = self.cached_response_to_anthropic(outcome.cached)
            if response is not None:
                logger.debug("Serving cached Anthropic response from replay cache")
                mark_span_cached(span)
                if is_streaming:
                    return self._create_async_cached_stream(response)
                return response

        return await wrapped(*args, **kwargs)

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

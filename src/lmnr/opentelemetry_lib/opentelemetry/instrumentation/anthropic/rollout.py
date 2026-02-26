"""
Rollout wrapper for Anthropic chat completions instrumentation.

This module adds caching and override capabilities to Anthropic chat completions
instrumentation during rollout sessions.
"""

import json
from typing import Any, AsyncGenerator, Generator, Optional

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
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.rollout.instrumentation import RolloutInstrumentationWrapper
from lmnr.sdk.rollout_control import is_rollout_mode

logger = get_default_logger(__name__)


class AnthropicRolloutWrapper(RolloutInstrumentationWrapper):
    """
    Rollout wrapper specific to Anthropic chat completions instrumentation.

    Handles:
    - Converting cached responses to Anthropic Message format
    - Applying overrides to system prompts and tool definitions
    - Setting rollout-specific span attributes
    """

    def apply_anthropic_overrides(
        self, kwargs: dict[str, Any], path: str, span: Span | None = None
    ) -> dict[str, Any]:
        overrides = self.get_overrides(path)
        if not overrides:
            return kwargs

        modified_kwargs = kwargs.copy()

        if "system" in overrides:
            system_override = overrides["system"]
            logger.debug(f"Applying system override for {path}")

            if span and span.is_recording():
                try:
                    # For Anthropic, we store system prompt separately in gen_ai.input.messages if present
                    # or it might be passed as a 'system' parameter in kwargs.
                    # We update the span attribute to reflect the override.
                    if input_messages_raw := span.attributes.get(
                        "gen_ai.input.messages"
                    ):
                        input_messages = json.loads(input_messages_raw)
                        if input_messages and input_messages[0].get("role") == "system":
                            input_messages[0]["content"] = system_override
                        else:
                            input_messages.insert(
                                0, {"role": "system", "content": system_override}
                            )

                        span.set_attribute(
                            "gen_ai.input.messages",
                            json.dumps(input_messages),
                        )
                except Exception:
                    pass

            modified_kwargs["system"] = system_override

        if "tools" in overrides:
            # Anthropic tools are passed as a list of dicts
            modified_kwargs["tools"] = overrides["tools"]

        return modified_kwargs

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
        if not self.should_use_rollout():
            return wrapped(*args, **kwargs)

        span_path = self._get_span_path_from_span(span) if span else None
        if not span_path:
            return wrapped(*args, **kwargs)

        current_index = self.get_current_index_for_path(span_path)
        logger.debug(f"Anthropic create call at {span_path}:{current_index}")

        if self.should_use_cache(span_path, current_index):
            cached_span = self.get_cached_response(span_path, current_index)
            if cached_span:
                logger.debug(
                    f"Using cached response for Anthropic at {span_path}:{current_index}"
                )
                response = self.cached_response_to_anthropic(cached_span)
                if response:
                    try:
                        if span and span.is_recording():
                            span.set_attributes(
                                {
                                    "lmnr.span.type": "CACHED",
                                    "lmnr.rollout.cache_index": current_index,
                                    "lmnr.span.original_type": "LLM",
                                }
                            )
                    except Exception:
                        pass

                    if is_streaming:
                        if is_async:
                            return self._as_coroutine(
                                self._create_async_cached_stream(response)
                            )
                        else:
                            return self._create_cached_stream(response)

                    if is_async:
                        return self._as_coroutine(response)
                    return response

        modified_kwargs = self.apply_anthropic_overrides(kwargs, span_path, span)
        logger.debug(f"Executing live Anthropic call for {span_path}:{current_index}")
        return wrapped(*args, **modified_kwargs)

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

    def _get_span_path_from_span(self, span: Span) -> Optional[str]:
        """Get the span path from the span's attributes."""
        try:
            path_list = span.attributes.get("lmnr.span.path")
            if path_list:
                return ".".join(path_list)
        except Exception:
            pass
        return None


_anthropic_rollout_wrapper: AnthropicRolloutWrapper | None = None


def get_anthropic_rollout_wrapper() -> AnthropicRolloutWrapper | None:
    global _anthropic_rollout_wrapper

    if not is_rollout_mode():
        return None

    if _anthropic_rollout_wrapper is None:
        try:
            _anthropic_rollout_wrapper = AnthropicRolloutWrapper()
        except Exception as e:
            logger.error(f"Failed to create Anthropic rollout wrapper: {e}")
            return None

    return _anthropic_rollout_wrapper

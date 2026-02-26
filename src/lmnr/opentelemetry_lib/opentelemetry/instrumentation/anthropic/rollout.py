"""
Rollout wrapper for Anthropic instrumentation.

This module adds caching and override capabilities to Anthropic instrumentation
during rollout sessions.
"""

import json
from typing import Any, AsyncGenerator, Generator

from anthropic.types import (
    Message,
    TextBlock,
    ToolUseBlock,
    Usage,
)
from opentelemetry.trace import Span

from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.rollout.instrumentation import RolloutInstrumentationWrapper
from lmnr.sdk.rollout_control import is_rollout_mode

logger = get_default_logger(__name__)


class AnthropicRolloutWrapper(RolloutInstrumentationWrapper):
    """
    Rollout wrapper specific to Anthropic messages instrumentation.

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

            # Update the span's input messages attribute to reflect the override
            if span and span.is_recording():
                try:
                    if input_messages_raw := span.attributes.get(
                        "gen_ai.input.messages"
                    ):
                        input_messages = json.loads(input_messages_raw)
                        if input_messages and input_messages[0].get("role") == "system":
                            input_messages[0]["content"] = system_override
                            span.set_attribute(
                                "gen_ai.input.messages",
                                json.dumps(input_messages),
                            )
                except Exception:
                    pass

            # Anthropic uses a top-level "system" kwarg (string or list of blocks)
            modified_kwargs["system"] = system_override

        if "tools" in overrides:
            tool_overrides = overrides["tools"]
            logger.debug(
                f"Applying tool overrides for {path}: {len(tool_overrides)} tools"
            )

            existing_tools = modified_kwargs.get("tools", [])
            updated_tools = self._apply_tool_overrides(existing_tools, tool_overrides)
            if updated_tools is not None:
                modified_kwargs["tools"] = updated_tools

        return modified_kwargs

    def _apply_tool_overrides(
        self,
        existing_tools: list[dict[str, Any]] | None,
        tool_overrides: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        if not tool_overrides:
            return existing_tools

        # Anthropic tools use: {"name": ..., "description": ..., "input_schema": ...}
        tools_by_name: dict[str, dict[str, Any]] = {}
        if existing_tools:
            for tool in existing_tools:
                if name := tool.get("name"):
                    tools_by_name[name] = tool

        for override in tool_overrides:
            name = override.get("name")
            if not name:
                continue

            if name in tools_by_name:
                existing_tool = tools_by_name[name]
                tools_by_name[name] = {
                    "name": name,
                    "description": override.get(
                        "description", existing_tool.get("description")
                    ),
                    "input_schema": override.get(
                        "parameters", existing_tool.get("input_schema")
                    ),
                }
            elif "parameters" in override:
                tools_by_name[name] = {
                    "name": name,
                    "description": override.get("description"),
                    "input_schema": override["parameters"],
                }

        if tools_by_name:
            return list(tools_by_name.values())
        return None

    def cached_response_to_anthropic(
        self, cached_span: dict[str, Any]
    ) -> Message | None:
        """
        Convert cached span data to Anthropic Message response.
        Tries raw response first, then falls back to legacy parsing.
        """
        # Try raw response first
        if raw_response := cached_span.get("attributes", {}).get(
            "lmnr.sdk.raw.response"
        ):
            try:
                if isinstance(raw_response, dict):
                    return Message.model_validate(raw_response)
                elif isinstance(raw_response, str):
                    return Message.model_validate_json(raw_response)
            except Exception as e:
                logger.debug(f"Failed to parse raw response: {e}")

        # Legacy parsing: reconstruct Message from input/output attributes
        try:
            attributes = cached_span.get("attributes", {})
            output_str = cached_span.get("output", "")
            if not output_str:
                logger.warning("Cached span has no output")
                return None

            output = json.loads(output_str)
            if not isinstance(output, list) or len(output) == 0:
                logger.warning(f"Unexpected output format: {type(output)}")
                return None

            # Expected format from Anthropic instrumentation:
            # [{"role": "assistant", "content": [...], "stop_reason": "..."}]
            message_data = output[0]
            content_blocks = message_data.get("content", [])

            content = []
            for block in content_blocks:
                block_type = block.get("type")

                if block_type == "text":
                    content.append(
                        TextBlock(
                            type="text",
                            text=block.get("text", ""),
                        )
                    )
                elif block_type == "tool_use":
                    tool_input = block.get("input", {})
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    content.append(
                        ToolUseBlock(
                            type="tool_use",
                            id=block.get("id", ""),
                            name=block.get("name", ""),
                            input=tool_input,
                        )
                    )
                elif block_type == "thinking":
                    try:
                        from anthropic.types import ThinkingBlock

                        content.append(
                            ThinkingBlock(
                                type="thinking",
                                thinking=block.get("thinking", ""),
                                signature=block.get("signature", ""),
                            )
                        )
                    except ImportError:
                        logger.debug("ThinkingBlock not available in this SDK version")

            stop_reason = message_data.get("stop_reason", "end_turn")

            return Message(
                id=attributes.get("gen_ai.response.id", "cached"),
                model=attributes.get("gen_ai.response.model", "unknown"),
                role="assistant",
                content=content,
                stop_reason=stop_reason,
                stop_sequence=None,
                type="message",
                usage=Usage(
                    input_tokens=0,
                    output_tokens=0,
                ),
            )

        except Exception as e:
            logger.debug(
                f"Failed to convert cached response to Anthropic format: {e}",
                exc_info=True,
            )
            return None

    def wrap_messages_create(
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

        span_path = self.get_span_path()
        if not span_path:
            return wrapped(*args, **kwargs)

        current_index = self.get_current_index_for_path(span_path)
        logger.debug(f"Anthropic messages call at {span_path}:{current_index}")

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
                            return self._create_async_cached_stream(response)
                        else:
                            return self._create_cached_stream(response)
                    return response

        modified_kwargs = self.apply_anthropic_overrides(kwargs, span_path, span)
        logger.debug(f"Executing live Anthropic call for {span_path}:{current_index}")
        return wrapped(*args, **modified_kwargs)

    def _create_cached_stream(self, response: Message) -> Generator[Any, None, None]:
        """Create a generator that yields cached response as streaming events."""
        from anthropic.types import (
            MessageDeltaUsage,
            RawMessageStartEvent,
            RawMessageDeltaEvent,
            RawMessageStopEvent,
            RawContentBlockStartEvent,
            RawContentBlockStopEvent,
            RawContentBlockDeltaEvent,
            TextDelta,
            InputJSONDelta,
        )
        from anthropic.types.raw_message_delta_event import Delta as MessageDelta

        # message_start event
        yield RawMessageStartEvent(
            type="message_start",
            message=response,
        )

        # Content block events
        for i, block in enumerate(response.content):
            yield RawContentBlockStartEvent(
                type="content_block_start",
                index=i,
                content_block=block,
            )

            # Emit a delta for text blocks
            if block.type == "text":
                yield RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=i,
                    delta=TextDelta(type="text_delta", text=block.text),
                )
            elif block.type == "tool_use":
                yield RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=i,
                    delta=InputJSONDelta(
                        type="input_json_delta",
                        partial_json=json.dumps(block.input),
                    ),
                )

            yield RawContentBlockStopEvent(
                type="content_block_stop",
                index=i,
            )

        # message_delta event (with stop reason and usage)
        delta_usage = MessageDeltaUsage(
            output_tokens=getattr(response.usage, "output_tokens", 0)
            if response.usage
            else 0,
        )
        yield RawMessageDeltaEvent(
            type="message_delta",
            delta=MessageDelta(
                stop_reason=response.stop_reason,
                stop_sequence=response.stop_sequence,
            ),
            usage=delta_usage,
        )

        # message_stop event
        yield RawMessageStopEvent(type="message_stop")

    async def _create_async_cached_stream(
        self, response: Message
    ) -> AsyncGenerator[Any, None]:
        """Yield cached response as async streaming events."""
        for event in self._create_cached_stream(response):
            yield event


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

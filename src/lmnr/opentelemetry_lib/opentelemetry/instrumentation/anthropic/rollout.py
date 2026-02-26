"""
Rollout wrapper for Anthropic messages instrumentation.

This module adds caching and override capabilities to Anthropic messages
instrumentation during rollout sessions.
"""

import json
from typing import Any, AsyncGenerator, Generator

from anthropic.types import (
    InputJSONDelta,
    Message,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    Usage,
    RawMessageStartEvent,
    RawContentBlockStartEvent,
    RawContentBlockDeltaEvent,
    RawMessageDeltaEvent,
    RawMessageStopEvent,
)
from anthropic.types.raw_message_delta_event import Delta as MessageDelta
from opentelemetry.trace import Span

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

            # Anthropic uses a separate "system" kwarg, not embedded in messages
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
        if raw_response := cached_span.get("attributes", {}).get(
            "lmnr.sdk.raw.response"
        ):
            try:
                response_dict = (
                    raw_response
                    if isinstance(raw_response, dict)
                    else json.loads(raw_response)
                )
                if response_dict:
                    return Message.model_validate(response_dict)
            except Exception as e:
                logger.debug(f"Failed to parse raw response: {e}")

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

            # Output format: [{"role": "assistant", "content": [...], "stop_reason": ...}]
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
                            id=block.get("id", "cached_tool_use"),
                            name=block.get("name", ""),
                            input=tool_input,
                        )
                    )
                elif block_type == "thinking":
                    # Thinking blocks are not returned as content in cached responses
                    pass
                else:
                    logger.debug(f"Unknown content block type: {block_type}")

            stop_reason = message_data.get("stop_reason", "end_turn")

            return Message(
                id=attributes.get("gen_ai.response.id", "cached_msg"),
                model=attributes.get("gen_ai.response.model", "unknown"),
                content=content,
                role="assistant",
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

    def _get_span_path_from_span(self, span: Span) -> str | None:
        """Get the span path from the span's attributes (set by the processor)."""
        try:
            path_list = span.attributes.get("lmnr.span.path")
            if path_list:
                return ".".join(path_list)
        except Exception:
            pass
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

        span_path = self._get_span_path_from_span(span) if span else None
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

    def _create_cached_stream(
        self, response: Message
    ) -> Generator[Any, None, None]:
        """Yield a cached response as streaming events."""
        # message_start event
        yield RawMessageStartEvent(
            message=Message(
                id=response.id,
                model=response.model,
                content=[],
                role="assistant",
                stop_reason=None,
                stop_sequence=None,
                type="message",
                usage=Usage(
                    input_tokens=0,
                    output_tokens=0,
                ),
            ),
            type="message_start",
        )

        # content_block_start + content_block_delta for each block
        for index, block in enumerate(response.content):
            if isinstance(block, TextBlock):
                yield RawContentBlockStartEvent(
                    content_block=TextBlock(type="text", text=""),
                    index=index,
                    type="content_block_start",
                )
                yield RawContentBlockDeltaEvent(
                    delta=TextDelta(type="text_delta", text=block.text),
                    index=index,
                    type="content_block_delta",
                )
            elif isinstance(block, ToolUseBlock):
                yield RawContentBlockStartEvent(
                    content_block=ToolUseBlock(
                        type="tool_use",
                        id=block.id,
                        name=block.name,
                        input={},
                    ),
                    index=index,
                    type="content_block_start",
                )
                yield RawContentBlockDeltaEvent(
                    delta=InputJSONDelta(
                        type="input_json_delta",
                        partial_json=json.dumps(block.input),
                    ),
                    index=index,
                    type="content_block_delta",
                )

        # message_delta event
        yield RawMessageDeltaEvent(
            delta=MessageDelta(
                stop_reason=response.stop_reason,
                stop_sequence=None,
            ),
            type="message_delta",
            usage={"output_tokens": 0},
        )

        # message_stop event
        yield RawMessageStopEvent(
            type="message_stop",
            message=Message(
                id=response.id,
                model=response.model,
                content=[],
                role="assistant",
                stop_reason=response.stop_reason,
                stop_sequence=None,
                type="message",
                usage=Usage(
                    input_tokens=0,
                    output_tokens=0,
                ),
            ),
        )

    async def _create_async_cached_stream(
        self, response: Message
    ) -> AsyncGenerator[Any, None]:
        """Yield a cached response as async streaming events."""
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

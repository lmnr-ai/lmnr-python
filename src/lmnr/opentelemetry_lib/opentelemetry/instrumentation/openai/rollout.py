"""
Rollout wrapper for OpenAI chat completions instrumentation.

This module adds caching and override capabilities to OpenAI chat completions
instrumentation during rollout sessions.
"""

import json
from typing import Any, AsyncGenerator, Generator

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
)
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function as ToolCallFunction,
)
from opentelemetry.trace import Span

from lmnr.sdk.log import get_default_logger
from lmnr.sdk.rollout.instrumentation import RolloutInstrumentationWrapper
from lmnr.sdk.rollout_control import is_rollout_mode

logger = get_default_logger(__name__)


class OpenAIRolloutWrapper(RolloutInstrumentationWrapper):
    """
    Rollout wrapper specific to OpenAI chat completions instrumentation.

    Handles:
    - Converting cached responses to OpenAI ChatCompletion format
    - Applying overrides to system prompts and tool definitions
    - Setting rollout-specific span attributes
    """

    def apply_openai_overrides(
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

            messages = modified_kwargs.get("messages", [])
            if messages and isinstance(messages, list):
                modified_kwargs["messages"] = list(messages)
                if modified_kwargs["messages"][0].get("role") == "system":
                    modified_kwargs["messages"][0] = {
                        **modified_kwargs["messages"][0],
                        "content": system_override,
                    }
                else:
                    modified_kwargs["messages"].insert(
                        0, {"role": "system", "content": system_override}
                    )

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
                func = tool.get("function", {})
                if name := func.get("name"):
                    tools_by_name[name] = tool

        for override in tool_overrides:
            name = override.get("name")
            if not name:
                continue

            if name in tools_by_name:
                existing_func = tools_by_name[name].get("function", {})
                tools_by_name[name] = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": override.get(
                            "description", existing_func.get("description")
                        ),
                        "parameters": override.get(
                            "parameters", existing_func.get("parameters")
                        ),
                    },
                }
            elif "parameters" in override:
                tools_by_name[name] = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": override.get("description"),
                        "parameters": override["parameters"],
                    },
                }

        if tools_by_name:
            return list(tools_by_name.values())
        return None

    def cached_response_to_openai(
        self, cached_span: dict[str, Any]
    ) -> ChatCompletion | None:
        """
        Convert cached span data to OpenAI ChatCompletion response.
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
                    return ChatCompletion.model_validate(response_dict)
            except Exception as e:
                logger.debug(f"Failed to parse raw response: {e}")

        try:
            attributes = cached_span.get("attributes", {})
            output_str = cached_span.get("output", "")
            if not output_str:
                logger.warning("Cached span has no output")
                return None

            # NOTE: This is assuming the latest openai instrumentation where we save the output as
            # a list of choices. Not compatible with older versions.
            output = json.loads(output_str)
            if not isinstance(output, list):
                logger.warning(f"Unexpected output format: {type(output)}")
                return None

            choices = []
            for choice_data in output:
                msg = choice_data.get("message", {})

                tool_calls = None
                if raw_tool_calls := msg.get("tool_calls"):
                    tool_calls = [
                        ChatCompletionMessageFunctionToolCall(
                            id=tc.get("id", ""),
                            type="function",
                            function=ToolCallFunction(
                                name=tc.get("function", {}).get("name", ""),
                                arguments=tc.get("function", {}).get("arguments", ""),
                            ),
                        )
                        for tc in raw_tool_calls
                    ]

                function_call = None
                if raw_fc := msg.get("function_call"):
                    function_call = FunctionCall(
                        name=raw_fc.get("name", ""),
                        arguments=raw_fc.get("arguments", ""),
                    )

                choices.append(
                    Choice(
                        index=choice_data.get("index", len(choices)),
                        finish_reason=choice_data.get("finish_reason", "stop"),
                        message=ChatCompletionMessage(
                            role=msg.get("role", "assistant"),
                            content=msg.get("content"),
                            refusal=msg.get("refusal"),
                            tool_calls=tool_calls,
                            function_call=function_call,
                        ),
                    )
                )

            return ChatCompletion(
                id=attributes.get("gen_ai.response.id", "cached"),
                model=attributes.get("gen_ai.response.model", "unknown"),
                choices=choices,
                created=int(cached_span.get("start_time", 0) / 1_000_000_000),
                object="chat.completion",
                usage=None,
            )

        except Exception as e:
            logger.debug(
                f"Failed to convert cached response to OpenAI format: {e}",
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

    def wrap_chat_completion(
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

        # Read path from the span's attributes rather than Laminar.get_current_span(),
        # because OpenAI spans are created with tracer.start_span() and are not
        # registered in Laminar's context.
        span_path = self._get_span_path_from_span(span) if span else None
        if not span_path:
            return wrapped(*args, **kwargs)

        current_index = self.get_current_index_for_path(span_path)
        logger.debug(f"OpenAI chat completion call at {span_path}:{current_index}")

        if self.should_use_cache(span_path, current_index):
            cached_span = self.get_cached_response(span_path, current_index)
            if cached_span:
                logger.debug(
                    f"Using cached response for OpenAI at {span_path}:{current_index}"
                )
                response = self.cached_response_to_openai(cached_span)
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

        modified_kwargs = self.apply_openai_overrides(kwargs, span_path, span)
        logger.debug(f"Executing live OpenAI call for {span_path}:{current_index}")
        return wrapped(*args, **modified_kwargs)

    def _create_cached_stream(
        self, response: ChatCompletion
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Yield a cached response as a single streaming chunk."""
        response_dict = response.model_dump()

        choices = []
        for choice in response_dict.get("choices", []):
            message = dict(choice.get("message", {}))
            # Streaming tool_calls require an `index` field that the message
            # format doesn't have. Add it before passing to ChoiceDelta.
            if message.get("tool_calls"):
                message["tool_calls"] = [
                    {**tc, "index": i} for i, tc in enumerate(message["tool_calls"])
                ]
            choices.append(
                ChunkChoice(
                    index=choice.get("index", 0),
                    delta=ChoiceDelta(**message),
                    finish_reason=choice.get("finish_reason"),
                )
            )

        yield ChatCompletionChunk(
            id=response_dict.get("id", "cached"),
            model=response_dict.get("model", "unknown"),
            object="chat.completion.chunk",
            created=response_dict.get("created", 0),
            choices=choices,
        )

    async def _create_async_cached_stream(
        self, response: ChatCompletion
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Yield a cached response as a single async streaming chunk."""
        for chunk in self._create_cached_stream(response):
            yield chunk


_openai_rollout_wrapper: OpenAIRolloutWrapper | None = None


def get_openai_rollout_wrapper() -> OpenAIRolloutWrapper | None:
    global _openai_rollout_wrapper

    if not is_rollout_mode():
        return None

    if _openai_rollout_wrapper is None:
        try:
            _openai_rollout_wrapper = OpenAIRolloutWrapper()
        except Exception as e:
            logger.error(f"Failed to create OpenAI debugger wrapper: {e}")
            return None

    return _openai_rollout_wrapper

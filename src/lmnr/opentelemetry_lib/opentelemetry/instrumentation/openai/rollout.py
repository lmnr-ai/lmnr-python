"""
Debug replay wrapper for OpenAI chat completions instrumentation.

On a debug run with replay enabled, serves the first N spine LLM calls from the
in-process `ReplayCache` (reconstructing an OpenAI `ChatCompletion` from cached
attributes) and runs the rest live. Overrides + HTTP cache server are gone (§H).
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

from lmnr.sdk.debug.replay import (
    cached_payload_for,
    mark_span_cached,
    replay_enabled,
    span_path_from_span,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class OpenAIRolloutWrapper:
    """Serves cached OpenAI responses on a debug run; runs live otherwise."""

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

    def wrap_chat_completion(
        self,
        wrapped,
        instance,
        args,
        kwargs,
        span: Any = None,
        is_streaming: bool = False,
        is_async: bool = False,
    ) -> Any:
        # Read path from the span's attributes rather than Laminar.get_current_span(),
        # because OpenAI spans are created with tracer.start_span() and are not
        # registered in Laminar's context.
        span_path = span_path_from_span(span)
        cached = cached_payload_for(span_path)
        if cached is not None:
            response = self.cached_response_to_openai(cached)
            if response is not None:
                logger.debug("Replaying cached OpenAI response at %s", span_path)
                mark_span_cached(span)
                if is_streaming:
                    if is_async:
                        return self._create_async_cached_stream(response)
                    return self._create_cached_stream(response)
                return response

        return wrapped(*args, **kwargs)

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

    if not replay_enabled():
        return None

    if _openai_rollout_wrapper is None:
        try:
            _openai_rollout_wrapper = OpenAIRolloutWrapper()
        except Exception as e:
            logger.error(f"Failed to create OpenAI debugger wrapper: {e}")
            return None

    return _openai_rollout_wrapper

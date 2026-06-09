"""
Debug replay wrapper for OpenAI chat completions and Responses API
instrumentation.

On a debug run with replay configured, each live LLM call asks the server-side
cache (shared spec §7) what to do: on a HIT it reconstructs an OpenAI
`ChatCompletion` / `Response` from the cached span and serves it; on MISS / LIVE
it runs the call live. The decision is made by `cache_outcome_for` (sync) /
`acache_outcome_for` (async) — there is no in-process cache anymore.

The chat path probes the cache inside `wrap_chat_completion`; the Responses API
path keeps its orchestration in `responses_wrappers.py` (its instrumentation
opens the span and builds `TracedData` post-hoc) and only borrows
`cached_response_to_responses` from here to turn a cached envelope into an
`openai.types.responses.Response`.
"""

import json
import uuid
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

try:
    from openai.types.responses import Response

    _RESPONSES_AVAILABLE = True
except ImportError:
    Response = Any
    _RESPONSES_AVAILABLE = False

from lmnr.sdk.debug.replay import (
    acache_outcome_for,
    cache_outcome_for,
    mark_span_cached,
    replay_enabled,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class OpenAIRolloutWrapper:
    """Serves cached OpenAI responses on a debug run; runs live otherwise."""

    def cached_response_to_openai(
        self, cached_span: dict[str, Any]
    ) -> ChatCompletion | None:
        """Convert cached span envelope to OpenAI ChatCompletion response."""
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
                return ChatCompletion.model_validate(response_dict)
            except Exception as e:
                logger.debug(f"Failed to parse raw OpenAI response: {e}", exc_info=True)
                return None

        # envelope_type == "genAi"
        try:
            messages = cached_span.get("messages")
            if not isinstance(messages, list):
                logger.warning("Cached span type='genAi' has no messages list")
                return None
            model = cached_span.get("model", "unknown")
            finish_reasons = cached_span.get("finishReasons", [])

            choices = []
            for i, choice_data in enumerate(messages):
                msg = choice_data.get("message", {})
                finish_reason = finish_reasons[i] if i < len(finish_reasons) else "stop"

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
                        index=choice_data.get("index", i),
                        finish_reason=finish_reason,
                        message=ChatCompletionMessage(
                            role=msg.get("role", "assistant"),
                            content=msg.get("content"),
                            refusal=msg.get("refusal"),
                            tool_calls=tool_calls,
                            function_call=function_call,
                            annotations=msg.get("annotations", []),
                        ),
                    )
                )

            return ChatCompletion(
                id="cached",
                model=model,
                choices=choices,
                created=0,
                object="chat.completion",
                usage=None,
            )

        except Exception as e:
            logger.debug(
                f"Failed to convert genAi response to OpenAI format: {e}",
                exc_info=True,
            )
            return None

    def cached_response_to_responses(
        self, cached_span: dict[str, Any]
    ) -> "Response | None":
        """Convert a cached span envelope to an OpenAI Responses-API `Response`.

        Mirrors `cached_response_to_openai` but targets
        `openai.types.responses.Response`. `type="raw"` validates the recorded
        provider response directly; `type="genAi"` rebuilds the response from the
        stored OTel output items (`gen_ai.output.messages`) — each item already
        has the `{type, ...}` shape `Response.output` expects (message /
        function_call / reasoning / *_call blocks). Streaming is out of scope.
        """
        if not _RESPONSES_AVAILABLE:
            logger.warning(
                "openai.types.responses.Response unavailable; cannot serve cached "
                "Responses-API response"
            )
            return None

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
                return Response.model_validate(response_dict)
            except Exception as e:
                logger.debug(
                    f"Failed to parse raw OpenAI Responses response: {e}",
                    exc_info=True,
                )
                return None

        # envelope_type == "genAi"
        try:
            messages = cached_span.get("messages")
            if not isinstance(messages, list):
                logger.warning("Cached span type='genAi' has no messages list")
                return None
            model = cached_span.get("model", "unknown")

            # The recorder (responses_wrappers.set_data_attributes) stores the
            # response's output blocks — optionally preceded by a reasoning item —
            # as the OTel output messages. They already carry the discriminated
            # `type` field Response.output validates against, so pass them through.
            output_items = [item for item in messages if isinstance(item, dict)]

            response_dict = {
                # Unique per reconstruction: the Responses instrumentation keys
                # its module-level `responses` accumulator by response id, so a
                # constant id collides across sequential cached HITs and stamps
                # the first call's (stale, shorter) input on every later span.
                "id": f"cached_{uuid.uuid4().hex}",
                "created_at": 0,
                "model": model,
                "object": "response",
                "output": output_items,
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "status": "completed",
                "usage": None,
            }
            return Response.model_validate(response_dict)

        except Exception as e:
            logger.debug(
                f"Failed to convert genAi response to OpenAI Responses format: {e}",
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
        # Async path: hand back the coroutine so the call site (which checks
        # inspect.iscoroutine) awaits the cache lookup off the event loop.
        if is_async:
            return self._awrap_chat_completion(
                wrapped, args, kwargs, span, is_streaming
            )

        outcome = cache_outcome_for(span)
        if outcome is not None and outcome.kind == "hit":
            response = self.cached_response_to_openai(outcome.cached)
            if response is not None:
                logger.debug("Serving cached OpenAI response from replay cache")
                mark_span_cached(span)
                if is_streaming:
                    return self._create_cached_stream(response)
                return response

        return wrapped(*args, **kwargs)

    async def _awrap_chat_completion(
        self, wrapped, args, kwargs, span, is_streaming
    ) -> Any:
        """Async cache lookup; on a HIT serve cached, else run the call live."""
        outcome = await acache_outcome_for(span)
        if outcome is not None and outcome.kind == "hit":
            response = self.cached_response_to_openai(outcome.cached)
            if response is not None:
                logger.debug("Serving cached OpenAI response from replay cache")
                mark_span_cached(span)
                if is_streaming:
                    return self._create_async_cached_stream(response)
                return response

        return await wrapped(*args, **kwargs)

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

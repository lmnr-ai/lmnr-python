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
`openai.types.responses.Response`, plus, for `stream=True`,
`cached_response_to_responses_events` and the `(Async)CachedResponsesEventStream`
that replay that response as a stream of `ResponseStreamEvent`s.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, cast

from opentelemetry.sdk.trace import Span as SDKSpan
from opentelemetry.trace import Span
from typing_extensions import TypeVar

from lmnr.opentelemetry_lib.tracing.span import LaminarSpan
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
)
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    Function as ToolCallFunction,
)

if TYPE_CHECKING:
    from openai.types.responses import Response, ResponseStreamEvent

from lmnr.sdk.debug.replay import (
    acache_outcome_for,
    cache_outcome_for,
    mark_span_cached,
    replay_enabled,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)
T = TypeVar("T")


class OpenAIRolloutWrapper:
    """Serves cached OpenAI responses on a debug run; runs live otherwise."""

    def cached_response_to_openai(self, cached_span: dict[str, Any]) -> ChatCompletion | None:
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
                response_dict = raw if isinstance(raw, dict) else json.loads(raw)  # pyright: ignore[reportUnknownVariableType]
                return ChatCompletion.model_validate(response_dict)
            except Exception:
                logger.debug("Failed to parse raw OpenAI response", exc_info=True)
                return None

        # envelope_type == "genAi"
        try:
            messages: list[dict[str, Any]] | None = cached_span.get("messages")
            if not isinstance(messages, list):
                logger.warning("Cached span type='genAi' has no messages list")
                return None
            model = cached_span.get("model", "unknown")
            finish_reasons = cached_span.get("finishReasons", [])

            choices: list[Choice] = []
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
                    raw_fc = cast(dict[str, Any], raw_fc)
                    function_call = FunctionCall(
                        name=cast(str, raw_fc.get("name", "")),
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
                            tool_calls=cast(list[Any], tool_calls),
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

        except Exception:
            logger.debug(
                "Failed to convert genAi response to OpenAI format",
                exc_info=True,
            )
            return None

    def cached_response_to_responses(self, cached_span: dict[str, Any]) -> Response | None:
        """Convert a cached span envelope to an OpenAI Responses-API `Response`.

        Mirrors `cached_response_to_openai` but targets
        `openai.types.responses.Response`. `type="raw"` validates the recorded
        provider response directly; `type="genAi"` rebuilds the response from the
        stored OTel output items (`gen_ai.output.messages`) — each item already
        has the `{type, ...}` shape `Response.output` expects (message /
        function_call / reasoning / *_call blocks). For `stream=True`, feed the
        result to `cached_response_to_responses_events`.
        """
        try:
            from openai.types.responses import Response
        except ImportError:
            logger.warning(
                "openai.types.responses.Response unavailable; cannot serve cached "
                + "Responses-API response"
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
                response_dict = raw if isinstance(raw, dict) else json.loads(raw)  # pyright: ignore[reportUnknownVariableType]
                return Response.model_validate(response_dict)
            except Exception:
                logger.debug(
                    "Failed to parse raw OpenAI Responses response",
                    exc_info=True,
                )
                return None

        # envelope_type == "genAi"
        try:
            messages: list[dict[str, Any]] | None = cached_span.get("messages")
            if not isinstance(messages, list):
                logger.warning("Cached span type='genAi' has no messages list")
                return None
            model = cached_span.get("model", "unknown")

            # The recorder (responses_wrappers.set_data_attributes) stores the
            # response's output blocks — optionally preceded by a reasoning item —
            # as the OTel output messages. They already carry the discriminated
            # `type` field Response.output validates against, so pass them through.
            output_items = [item for item in messages if isinstance(item, dict)]  # pyright: ignore[reportUnnecessaryIsInstance]

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

        except Exception:
            logger.debug(
                "Failed to convert genAi response to OpenAI Responses format",
                exc_info=True,
            )
            return None

    def cached_response_to_responses_events(
        self, response: Response
    ) -> list[ResponseStreamEvent] | None:
        """Synthesize the Responses-API event sequence that streams `response`.

        The sequence is what the API itself emits, collapsed to one delta per
        text / refusal / arguments part: `response.created` and
        `response.in_progress` (with an empty `output`), then per output item
        `output_item.added`, its content parts (`content_part.added`, one delta,
        `*.done`, `content_part.done`) or arguments (one delta, `done`), and
        `output_item.done`, then the terminal `response.completed` /
        `response.incomplete` / `response.failed` carrying the full response
        (none for `cancelled`, which the API has no event for).
        That's the order the SDK's `ResponseStreamState` (behind
        `client.responses.stream()`) accumulates, so `get_final_response()` on a
        replayed stream returns `response`. Returns None if synthesis fails.
        """
        try:
            from openai.types.responses import (
                ResponseCompletedEvent,
                ResponseContentPartAddedEvent,
                ResponseContentPartDoneEvent,
                ResponseCreatedEvent,
                ResponseFailedEvent,
                ResponseFunctionCallArgumentsDeltaEvent,
                ResponseFunctionCallArgumentsDoneEvent,
                ResponseFunctionToolCall,
                ResponseIncompleteEvent,
                ResponseInProgressEvent,
                ResponseOutputItemAddedEvent,
                ResponseOutputItemDoneEvent,
                ResponseOutputMessage,
                ResponseOutputText,
                ResponseRefusalDeltaEvent,
                ResponseRefusalDoneEvent,
                ResponseTextDeltaEvent,
                ResponseTextDoneEvent,
            )
        except ImportError:
            logger.debug("Responses-API stream event types unavailable", exc_info=True)
            return None

        try:
            events: list[ResponseStreamEvent] = []

            def seq() -> int:
                return len(events)

            pending = response.model_copy(update={"status": "in_progress", "output": []})
            events.append(
                ResponseCreatedEvent(
                    response=pending, sequence_number=seq(), type="response.created"
                )
            )
            events.append(
                ResponseInProgressEvent(
                    response=pending, sequence_number=seq(), type="response.in_progress"
                )
            )

            for output_index, item in enumerate(response.output):
                if isinstance(item, ResponseOutputMessage):
                    added_item = item.model_copy(update={"content": [], "status": "in_progress"})
                elif isinstance(item, ResponseFunctionToolCall):
                    added_item = item.model_copy(update={"arguments": ""})
                else:
                    added_item = item
                events.append(
                    ResponseOutputItemAddedEvent(
                        item=added_item,
                        output_index=output_index,
                        sequence_number=seq(),
                        type="response.output_item.added",
                    )
                )

                if isinstance(item, ResponseOutputMessage):
                    for content_index, part in enumerate(item.content):
                        if isinstance(part, ResponseOutputText):
                            events.append(
                                ResponseContentPartAddedEvent(
                                    item_id=item.id,
                                    output_index=output_index,
                                    content_index=content_index,
                                    part=part.model_copy(update={"text": "", "annotations": []}),
                                    sequence_number=seq(),
                                    type="response.content_part.added",
                                )
                            )
                            events.append(
                                ResponseTextDeltaEvent(
                                    item_id=item.id,
                                    output_index=output_index,
                                    content_index=content_index,
                                    delta=part.text,
                                    logprobs=[],
                                    sequence_number=seq(),
                                    type="response.output_text.delta",
                                )
                            )
                            events.append(
                                ResponseTextDoneEvent(
                                    item_id=item.id,
                                    output_index=output_index,
                                    content_index=content_index,
                                    text=part.text,
                                    logprobs=[],
                                    sequence_number=seq(),
                                    type="response.output_text.done",
                                )
                            )
                        else:
                            events.append(
                                ResponseContentPartAddedEvent(
                                    item_id=item.id,
                                    output_index=output_index,
                                    content_index=content_index,
                                    part=part.model_copy(update={"refusal": ""}),
                                    sequence_number=seq(),
                                    type="response.content_part.added",
                                )
                            )
                            events.append(
                                ResponseRefusalDeltaEvent(
                                    item_id=item.id,
                                    output_index=output_index,
                                    content_index=content_index,
                                    delta=part.refusal,
                                    sequence_number=seq(),
                                    type="response.refusal.delta",
                                )
                            )
                            events.append(
                                ResponseRefusalDoneEvent(
                                    item_id=item.id,
                                    output_index=output_index,
                                    content_index=content_index,
                                    refusal=part.refusal,
                                    sequence_number=seq(),
                                    type="response.refusal.done",
                                )
                            )
                        events.append(
                            ResponseContentPartDoneEvent(
                                item_id=item.id,
                                output_index=output_index,
                                content_index=content_index,
                                part=part,
                                sequence_number=seq(),
                                type="response.content_part.done",
                            )
                        )
                elif isinstance(item, ResponseFunctionToolCall):
                    item_id = item.id or ""
                    events.append(
                        ResponseFunctionCallArgumentsDeltaEvent(
                            item_id=item_id,
                            output_index=output_index,
                            delta=item.arguments,
                            sequence_number=seq(),
                            type="response.function_call_arguments.delta",
                        )
                    )
                    events.append(
                        ResponseFunctionCallArgumentsDoneEvent(
                            item_id=item_id,
                            output_index=output_index,
                            arguments=item.arguments,
                            name=item.name,
                            sequence_number=seq(),
                            type="response.function_call_arguments.done",
                        )
                    )

                events.append(
                    ResponseOutputItemDoneEvent(
                        item=item,
                        output_index=output_index,
                        sequence_number=seq(),
                        type="response.output_item.done",
                    )
                )

            if response.status == "incomplete":
                events.append(
                    ResponseIncompleteEvent(
                        response=response, sequence_number=seq(), type="response.incomplete"
                    )
                )
            elif response.status == "failed":
                events.append(
                    ResponseFailedEvent(
                        response=response, sequence_number=seq(), type="response.failed"
                    )
                )
            elif response.status == "completed":
                events.append(
                    ResponseCompletedEvent(
                        response=response, sequence_number=seq(), type="response.completed"
                    )
                )
            # Any other status (`cancelled`) has no terminal event: the live
            # stream just ends, and `get_final_response()` raises.
            return events
        except Exception:
            logger.debug("Failed to synthesize Responses-API stream events", exc_info=True)
            return None

    def wrap_chat_completion(
        self,
        wrapped: Callable[..., ChatCompletion | Awaitable[ChatCompletion]],
        _instance: Any,
        args: Sequence[Any],
        kwargs: dict[str, Any],
        span: Span,
        is_streaming: bool = False,
        is_async: bool = False,
    ) -> (
        ChatCompletion
        | Generator[ChatCompletionChunk]
        | Awaitable[ChatCompletion | AsyncGenerator[ChatCompletionChunk]]
    ):
        # Async path: hand back the coroutine so the call site (which checks
        # inspect.iscoroutine) awaits the cache lookup off the event loop.
        if is_async:
            return self._awrap_chat_completion(
                cast(Callable[..., Awaitable[ChatCompletion]], wrapped),
                args,
                kwargs,
                span,
                is_streaming,
            )

        if not isinstance(span, LaminarSpan):
            span = LaminarSpan(cast(SDKSpan, span))
        outcome = cache_outcome_for(span)
        if outcome is not None and outcome.kind == "hit":
            response = self.cached_response_to_openai(outcome.cached or {})
            if response is not None:
                logger.debug("Serving cached OpenAI response from replay cache")
                mark_span_cached(span)
                if is_streaming:
                    return self._create_cached_stream(response)
                return response

        return wrapped(*args, **kwargs)

    async def _awrap_chat_completion(
        self,
        wrapped: Callable[..., Awaitable[ChatCompletion]],
        args: Sequence[Any],
        kwargs: dict[str, Any],
        span: Span,
        is_streaming: bool = False,
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk]:
        """Async cache lookup; on a HIT serve cached, else run the call live."""
        if not isinstance(span, LaminarSpan):
            span = LaminarSpan(cast(SDKSpan, span))
        outcome = await acache_outcome_for(span)
        if outcome is not None and outcome.kind == "hit":
            response = self.cached_response_to_openai(outcome.cached or {})
            if response is not None:
                logger.debug("Serving cached OpenAI response from replay cache")
                mark_span_cached(span)
                if is_streaming:
                    return self._create_async_cached_stream(response)
                return response

        return await wrapped(*args, **kwargs)

    def _create_cached_stream(self, response: ChatCompletion) -> Generator[ChatCompletionChunk]:
        """Yield a cached response as a single streaming chunk."""
        response_dict = response.model_dump()

        choices: list[ChunkChoice] = []
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
    ) -> AsyncGenerator[ChatCompletionChunk]:
        """Yield a cached response as a single async streaming chunk."""
        for chunk in self._create_cached_stream(response):
            yield chunk


class _CachedHTTPResponse:
    """Stand-in for the `httpx.Response` behind an SDK `Stream`. The SDK's
    `ResponseStream` helpers read `.response` off the stream they wrap and close
    it; nothing was fetched, so there is nothing to close."""

    status_code: int = 200
    headers: dict[str, str] = {}  # noqa: RUF012

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class CachedResponsesEventStream:
    """Sync replay of a cached Responses-API response, served in place of the
    SDK's `Stream[ResponseStreamEvent]` on a debug-run cache HIT. Supports what
    callers (and `client.responses.stream()`) use: iteration, the context
    manager, `close()` and `.response`."""

    def __init__(self, events: list[ResponseStreamEvent]):
        self._events: list[ResponseStreamEvent] = events
        self._index: int = 0
        self.response: _CachedHTTPResponse = _CachedHTTPResponse()

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> ResponseStreamEvent:
        if self._index >= len(self._events):
            raise StopIteration
        event = self._events[self._index]
        self._index += 1
        return event

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._index = len(self._events)


class AsyncCachedResponsesEventStream:
    """Async counterpart of `CachedResponsesEventStream`, replacing the SDK's
    `AsyncStream[ResponseStreamEvent]` (whose `close()` is a coroutine)."""

    def __init__(self, events: list[ResponseStreamEvent]):
        self._events: list[ResponseStreamEvent] = events
        self._index: int = 0
        self.response: _CachedHTTPResponse = _CachedHTTPResponse()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> ResponseStreamEvent:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        self._index = len(self._events)


_openai_rollout_wrapper: OpenAIRolloutWrapper | None = None


def get_openai_rollout_wrapper() -> OpenAIRolloutWrapper | None:
    global _openai_rollout_wrapper

    if not replay_enabled():
        return None

    if _openai_rollout_wrapper is None:
        try:
            _openai_rollout_wrapper = OpenAIRolloutWrapper()
        except Exception:
            logger.exception("Failed to create OpenAI debugger wrapper")
            return None

    return _openai_rollout_wrapper

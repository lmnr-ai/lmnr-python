from collections import defaultdict
from collections.abc import AsyncGenerator, Generator
from typing import Any, TypedDict, TypeVar, cast

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.trace import Span, Status, StatusCode

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    set_span_attribute,
    to_dict,
)
from lmnr.sdk.utils import json_dumps


class _ToolCallFunction(TypedDict):
    name: str | None
    arguments: str


class _ToolCall(TypedDict):
    index: int
    id: str | None
    type: str | None
    function: _ToolCallFunction


class _Choice(TypedDict):
    index: int | None
    content: str
    role: str
    reasoning_content: str
    finish_reason: str | None
    tool_calls: dict[int, _ToolCall]


class _Usage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, int]
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


class _Accumulated(TypedDict):
    id: str | None
    model: str | None
    usage: _Usage | None
    choices: defaultdict[int, _Choice]


T = TypeVar("T")


def _new_choice() -> _Choice:
    return {
        "index": None,
        "content": "",
        "role": "assistant",
        "reasoning_content": "",
        "finish_reason": None,
        "tool_calls": {},
    }


def _new_accumulated() -> _Accumulated:
    return {
        "id": None,
        "model": None,
        "usage": None,
        "choices": defaultdict(_new_choice),
    }


@dont_throw
def _accumulate_chunk(accumulated: _Accumulated, chunk: Any):  # pyright: ignore[reportAny, reportExplicitAny]
    chunk_dict = to_dict(chunk)
    if accumulated["id"] is None and chunk_dict.get("id"):
        accumulated["id"] = chunk_dict.get("id")
    if accumulated["model"] is None and chunk_dict.get("model"):
        accumulated["model"] = chunk_dict.get("model")
    if chunk_dict.get("usage") is not None:
        accumulated["usage"] = cast(_Usage, cast(object, to_dict(chunk_dict["usage"])))
    for i, choice in enumerate(chunk_dict.get("choices", [])):  # pyright: ignore[reportAny]
        idx = cast(int, choice.get("index", i))  # pyright: ignore[reportAny]
        accumulated["choices"][idx]["content"] += choice.get("content", "")  # pyright: ignore[reportAny]
        accumulated["choices"][idx]["index"] = idx
        if choice.get("finish_reason"):  # pyright: ignore[reportAny]
            accumulated["choices"][idx]["finish_reason"] = choice.get("finish_reason")  # pyright: ignore[reportAny]
        delta = choice.get("delta", {})  # pyright: ignore[reportAny]
        if delta.get("role"):  # pyright: ignore[reportAny]
            accumulated["choices"][idx]["role"] = delta.get("role")  # pyright: ignore[reportAny]
        if delta.get("content"):  # pyright: ignore[reportAny]
            accumulated["choices"][idx]["content"] += delta.get("content")  # pyright: ignore[reportAny]
        reasoning = next(
            (
                delta.get(key)  # pyright: ignore[reportAny]
                for key in ("reasoning_content", "reasoning", "thinking")
                if delta.get(key)  # pyright: ignore[reportAny]
            ),
            None,
        )
        if reasoning:
            accumulated["choices"][idx]["reasoning_content"] += reasoning
        if delta.get("tool_calls"):  # pyright: ignore[reportAny]
            tool_calls_acc = accumulated["choices"][idx]["tool_calls"]
            for tc_chunk in delta.get("tool_calls"):  # pyright: ignore[reportAny]
                tc_idx = tc_chunk.get("index", 0)  # pyright: ignore[reportAny]
                if tc_idx not in tool_calls_acc:
                    tool_calls_acc[tc_idx] = {
                        "index": tc_idx,
                        "id": None,
                        "type": None,
                        "function": {"name": None, "arguments": ""},
                    }
                tc = tool_calls_acc[tc_idx]
                if tc_chunk.get("id"):  # pyright: ignore[reportAny]
                    tc["id"] = tc_chunk["id"]
                if tc_chunk.get("type"):  # pyright: ignore[reportAny]
                    tc["type"] = tc_chunk["type"]
                func = tc_chunk.get("function") or {}  # pyright: ignore[reportAny, reportUnknownVariableType]
                if func.get("name"):  # pyright: ignore[reportUnknownMemberType]
                    tc["function"]["name"] = func["name"]
                if func.get("arguments"):  # pyright: ignore[reportUnknownMemberType]
                    tc["function"]["arguments"] += func["arguments"]


@dont_throw
def _set_accumulated_attributes(
    span: Span, accumulated: _Accumulated, record_raw_response: bool = False
):
    try:
        set_span_attribute(span, "gen_ai.response.id", accumulated["id"])
        set_span_attribute(span, "gen_ai.response.model", accumulated["model"])
        formatted_choices = []
        for choice in accumulated["choices"].values():
            formatted_choices.append(  # pyright: ignore[reportUnknownMemberType]
                {
                    "index": choice["index"],
                    # if the content is empty, set it to None
                    "content": (
                        choice["content"] if len(choice["content"]) > 0 else None
                    ),
                    "role": choice["role"],
                    "reasoning_content": choice["reasoning_content"] or None,
                    "finish_reason": (
                        choice["finish_reason"] if choice["finish_reason"] else None
                    ),
                    "tool_calls": (
                        list(choice["tool_calls"].values())
                        if choice["tool_calls"]
                        else None
                    ),
                }
            )

        set_span_attribute(
            span, "gen_ai.output.messages", json_dumps(formatted_choices)  # pyright: ignore[reportUnknownArgumentType]
        )

        if usage := accumulated.get("usage"):
            input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            output_tokens = usage.get(
                "completion_tokens", usage.get("output_tokens", 0)
            )
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
            set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
            set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
            set_span_attribute(span, "llm.usage.total_tokens", total_tokens)

            input_details = to_dict(usage.get("prompt_tokens_details", {}))
            if "cached_tokens" in input_details:
                set_span_attribute(
                    span,
                    "gen_ai.usage.cache_read_input_tokens",
                    input_details["cached_tokens"],  # pyright: ignore[reportAny]
                )
            elif "cache_read_input_tokens" in usage:
                set_span_attribute(
                    span,
                    "gen_ai.usage.cache_read_input_tokens",
                    usage["cache_read_input_tokens"],
                )
            if "cache_creation_tokens" in input_details:
                set_span_attribute(
                    span,
                    "gen_ai.usage.cache_creation_input_tokens",
                    input_details["cache_creation_tokens"],  # pyright: ignore[reportAny]
                )
            elif "cache_creation_input_tokens" in usage:
                set_span_attribute(
                    span,
                    "gen_ai.usage.cache_creation_input_tokens",
                    usage["cache_creation_input_tokens"],
                )

        # Record raw response in rollout mode
        if record_raw_response:
            # Reconstruct full response from accumulated data
            raw_response: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
                "id": accumulated["id"],
                "model": accumulated["model"],
                "object": "chat.completion",
                "choices": [],
                "usage": accumulated.get("usage"),
            }
            for choice in accumulated["choices"].values():
                raw_response["choices"].append(  # pyright: ignore[reportAny]
                    {
                        "index": choice["index"],
                        "message": {
                            "role": choice["role"],
                            "content": (
                                choice["content"]
                                if len(choice["content"]) > 0
                                else None
                            ),
                            "reasoning_content": (choice["reasoning_content"] or None),
                            "tool_calls": (
                                list(choice["tool_calls"].values())
                                if choice["tool_calls"]
                                else None
                            ),
                        },
                        "finish_reason": (
                            choice["finish_reason"] if choice["finish_reason"] else None
                        ),
                    }
                )
            set_span_attribute(span, "lmnr.sdk.raw.response", json_dumps(raw_response))
    finally:
        span.end()


def process_completion_streaming_response(
    span: Span,
    response: Generator[T, None, None],
    record_raw_response: bool = False,
) -> Generator[T, None, None]:
    accumulated = _new_accumulated()
    try:
        for item in response:
            _accumulate_chunk(accumulated, item)
            yield item
        _set_accumulated_attributes(span, accumulated, record_raw_response)
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise


async def process_completion_async_streaming_response(
    span: Span,
    response: AsyncGenerator[T, None],
    record_raw_response: bool = False,
) -> AsyncGenerator[T, None]:
    accumulated = _new_accumulated()
    try:
        async for item in response:
            _accumulate_chunk(accumulated, item)
            yield item
        _set_accumulated_attributes(span, accumulated, record_raw_response)
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise

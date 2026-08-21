from collections import defaultdict
from collections.abc import AsyncGenerator, Generator
from typing import Any

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


@dont_throw
def _accumulate_chunk(accumulated: dict, chunk: dict):
    chunk_dict = to_dict(chunk)
    if accumulated["id"] is None and chunk_dict.get("id"):
        accumulated["id"] = chunk_dict.get("id")
    if accumulated["model"] is None and chunk_dict.get("model"):
        accumulated["model"] = chunk_dict.get("model")
    if chunk_dict.get("usage") is not None:
        accumulated["usage"] = to_dict(chunk_dict["usage"])
    for i, choice in enumerate(chunk_dict.get("choices", [])):
        idx = choice.get("index", i)
        accumulated["choices"][idx]["content"] += choice.get("content", "")
        accumulated["choices"][idx]["index"] = idx
        if choice.get("finish_reason"):
            accumulated["choices"][idx]["finish_reason"] = choice.get("finish_reason")
        delta = choice.get("delta", {})
        if delta.get("role"):
            accumulated["choices"][idx]["role"] = delta.get("role")
        if delta.get("content"):
            accumulated["choices"][idx]["content"] += delta.get("content")
        if delta.get("tool_calls"):
            tool_calls_acc = accumulated["choices"][idx]["tool_calls"]
            for tc_chunk in delta.get("tool_calls"):
                tc_idx = tc_chunk.get("index", 0)
                if tc_idx not in tool_calls_acc:
                    tool_calls_acc[tc_idx] = {
                        "index": tc_idx,
                        "id": None,
                        "type": None,
                        "function": {"name": None, "arguments": ""},
                    }
                tc = tool_calls_acc[tc_idx]
                if tc_chunk.get("id"):
                    tc["id"] = tc_chunk["id"]
                if tc_chunk.get("type"):
                    tc["type"] = tc_chunk["type"]
                func = tc_chunk.get("function") or {}
                if func.get("name"):
                    tc["function"]["name"] = func["name"]
                if func.get("arguments"):
                    tc["function"]["arguments"] += func["arguments"]


@dont_throw
def _set_accumulated_attributes(
    span: Span, accumulated: dict, record_raw_response: bool = False
):
    try:
        set_span_attribute(span, "gen_ai.response.id", accumulated["id"])
        set_span_attribute(span, "gen_ai.response.model", accumulated["model"])
        formatted_choices = []
        for choice in accumulated["choices"].values():
            formatted_choices.append(
                {
                    "index": choice["index"],
                    # if the content is empty, set it to None
                    "content": (
                        choice["content"] if len(choice["content"]) > 0 else None
                    ),
                    "role": choice["role"],
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
            span, "gen_ai.output.messages", json_dumps(formatted_choices)
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
            cache_read_tokens = input_details.get(
                "cached_tokens", usage.get("cache_read_input_tokens", 0)
            )
            cache_creation_tokens = input_details.get(
                "cache_creation_tokens",
                usage.get("cache_creation_input_tokens", 0),
            )
            set_span_attribute(
                span, "gen_ai.usage.cache_read_input_tokens", cache_read_tokens
            )
            set_span_attribute(
                span,
                "gen_ai.usage.cache_creation_input_tokens",
                cache_creation_tokens,
            )

        # Record raw response in rollout mode
        if record_raw_response:
            # Reconstruct full response from accumulated data
            raw_response = {
                "id": accumulated["id"],
                "model": accumulated["model"],
                "object": "chat.completion",
                "choices": [],
                "usage": accumulated.get("usage"),
            }
            for choice in accumulated["choices"].values():
                raw_response["choices"].append(
                    {
                        "index": choice["index"],
                        "message": {
                            "role": choice["role"],
                            "content": (
                                choice["content"]
                                if len(choice["content"]) > 0
                                else None
                            ),
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
    response: Generator[Any, None, None],
    record_raw_response: bool = False,
) -> Generator[Any, None, None]:
    accumulated = {
        "id": None,
        "model": None,
        "usage": None,
        "choices": defaultdict(
            lambda: {
                "index": None,
                "content": "",
                "role": "assistant",
                "finish_reason": None,
                "tool_calls": {},
            }
        ),
    }
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
    response: AsyncGenerator[Any, None],
    record_raw_response: bool = False,
) -> AsyncGenerator[Any, None]:
    accumulated = {
        "id": None,
        "model": None,
        "usage": None,
        "choices": defaultdict(
            lambda: {
                "index": None,
                "content": "",
                "role": "assistant",
                "finish_reason": None,
                "tool_calls": {},
            }
        ),
    }
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

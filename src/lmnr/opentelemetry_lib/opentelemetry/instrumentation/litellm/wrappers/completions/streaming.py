from collections import defaultdict
from typing import Any, AsyncGenerator, Generator

from opentelemetry.trace import Span

from lmnr.sdk.utils import json_dumps
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    set_span_attribute,
    to_dict,
)


@dont_throw
def _accumulate_chunk(accumulated: dict, chunk: dict):
    chunk_dict = to_dict(chunk)
    if accumulated["id"] is None and chunk_dict.get("id"):
        accumulated["id"] = chunk_dict.get("id")
    if accumulated["model"] is None and chunk_dict.get("model"):
        accumulated["model"] = chunk_dict.get("model")
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
            accumulated["choices"][idx]["tool_calls"] = delta.get("tool_calls")


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
                        choice["tool_calls"] if choice["tool_calls"] else None
                    ),
                }
            )

        set_span_attribute(
            span, "gen_ai.output.messages", json_dumps(formatted_choices)
        )

        # Record raw response in rollout mode
        if record_raw_response:
            # Reconstruct full response from accumulated data
            raw_response = {
                "id": accumulated["id"],
                "model": accumulated["model"],
                "object": "chat.completion",
                "choices": [],
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
                                choice["tool_calls"] if choice["tool_calls"] else None
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
        "choices": defaultdict(
            lambda: {
                "index": None,
                "content": "",
                "role": "assistant",
                "finish_reason": None,
                "tool_calls": [],
            }
        ),
    }
    for item in response:
        _accumulate_chunk(accumulated, item)
        yield item
    _set_accumulated_attributes(span, accumulated, record_raw_response)


async def process_completion_async_streaming_response(
    span: Span,
    response: AsyncGenerator[Any, None],
    record_raw_response: bool = False,
) -> AsyncGenerator[Any, None]:
    accumulated = {
        "id": None,
        "model": None,
        "choices": defaultdict(
            lambda: {
                "index": None,
                "content": "",
                "role": "assistant",
                "finish_reason": None,
                "tool_calls": [],
            }
        ),
    }
    async for item in response:
        _accumulate_chunk(accumulated, item)
        yield item
    _set_accumulated_attributes(span, accumulated, record_raw_response)

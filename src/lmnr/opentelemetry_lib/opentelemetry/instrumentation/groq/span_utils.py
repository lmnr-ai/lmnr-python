from __future__ import annotations

from typing import Any, cast

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.trace import Span

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.reasoning import (
    set_reasoning_attributes,
    set_reasoning_usage,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    model_as_dict,
    set_span_attribute,
    should_send_prompts,
)
from lmnr.sdk.utils import json_dumps


@dont_throw
def set_input_attributes(span: Span, kwargs: dict[str, Any]):
    if not span.is_recording() or not should_send_prompts():
        return

    # `messages` and `tools` are typed `Iterable`; only record materialized
    # sequences, since reading a generator would consume what the SDK sends.
    messages = kwargs.get("messages")
    if isinstance(messages, (list, tuple)):
        set_span_attribute(
            span, "gen_ai.input.messages", json_dumps(list(cast(list[Any], messages)))
        )

    tools = kwargs.get("tools")
    if isinstance(tools, (list, tuple)) and tools:
        set_span_attribute(
            span, "gen_ai.tool.definitions", json_dumps(list(cast(list[Any], tools)))
        )


@dont_throw
def set_model_input_attributes(span: Span, kwargs: dict[str, Any]):
    if not span.is_recording():
        return

    set_span_attribute(span, GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    # `max_tokens` is deprecated in favor of `max_completion_tokens`.
    set_span_attribute(
        span,
        GEN_AI_REQUEST_MAX_TOKENS,
        kwargs.get("max_completion_tokens") or kwargs.get("max_tokens"),
    )
    set_span_attribute(span, GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature"))
    set_span_attribute(span, GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(span, GEN_AI_REQUEST_FREQUENCY_PENALTY, kwargs.get("frequency_penalty"))
    set_span_attribute(span, GEN_AI_REQUEST_PRESENCE_PENALTY, kwargs.get("presence_penalty"))
    set_span_attribute(span, "llm.is_streaming", kwargs.get("stream") or False)
    set_reasoning_attributes(span, kwargs)


def _set_usage_attributes(span: Span, usage: dict[str, Any]):
    set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, usage.get("prompt_tokens"))
    set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, usage.get("completion_tokens"))
    set_span_attribute(span, "llm.usage.total_tokens", usage.get("total_tokens"))

    prompt_details = cast(dict[str, int], usage.get("prompt_tokens_details") or {})
    set_span_attribute(
        span, "gen_ai.usage.cache_read.input_tokens", prompt_details.get("cached_tokens")
    )
    set_reasoning_usage(span, usage)


@dont_throw
def set_model_response_attributes(span: Span, response: Any):
    if not span.is_recording():
        return
    response = model_as_dict(response)
    set_span_attribute(span, GEN_AI_RESPONSE_MODEL, response.get("model"))
    set_span_attribute(span, GEN_AI_RESPONSE_ID, response.get("id"))

    # Streams report usage on the last chunk's `x_groq.usage` (newer API
    # versions also put it on the top-level `usage`).
    x_groq = cast(dict[str, Any], response.get("x_groq") or {})
    usage = response.get("usage") or x_groq.get("usage")
    if usage:
        _set_usage_attributes(span, cast(dict[str, Any], usage))


@dont_throw
def set_response_attributes(span: Span, response: Any):
    if not span.is_recording() or not should_send_prompts():
        return
    choices = model_as_dict(response).get("choices")
    if choices:
        set_span_attribute(span, "gen_ai.output.messages", json_dumps(choices))


class StreamAccumulator:
    """Folds `ChatCompletionChunk`s into a `ChatCompletion`-shaped dict."""

    def __init__(self):
        self._id: str | None = None
        self._model: str | None = None
        self._usage: dict[str, Any] | None = None
        self._choices: dict[int, dict[str, Any]] = {}
        self._tool_calls: dict[int, dict[int, dict[str, Any]]] = {}

    def add(self, chunk: Any):
        chunk = model_as_dict(chunk)
        self._id = self._id or chunk.get("id")
        self._model = self._model or chunk.get("model")
        x_groq = cast(dict[str, Any], chunk.get("x_groq") or {})
        if usage := chunk.get("usage") or x_groq.get("usage"):
            self._usage = cast(dict[str, Any], usage)

        for choice in cast(list[dict[str, Any]], chunk.get("choices") or []):
            index = cast(int, choice.get("index") or 0)
            accumulated = self._choices.setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": None,
                },
            )
            message = cast(dict[str, Any], accumulated["message"])
            delta = cast(dict[str, Any], choice.get("delta") or {})
            if delta.get("role"):
                message["role"] = delta["role"]
            if delta.get("content"):
                message["content"] = (message["content"] or "") + delta["content"]
            if delta.get("reasoning"):
                message["reasoning"] = message.get("reasoning", "") + delta["reasoning"]
            for call in cast(list[dict[str, Any]], delta.get("tool_calls") or []):
                slot = self._tool_calls.setdefault(index, {}).setdefault(
                    cast(int, call.get("index") or 0),
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                slot["id"] = slot["id"] or call.get("id")
                function = cast(dict[str, str | None], call.get("function") or {})
                slot["function"]["name"] += function.get("name") or ""
                slot["function"]["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                accumulated["finish_reason"] = choice["finish_reason"]

    def response(self) -> dict[str, Any]:
        for index, calls in self._tool_calls.items():
            self._choices[index]["message"]["tool_calls"] = [calls[i] for i in sorted(calls)]
        return {
            "id": self._id,
            "model": self._model,
            "usage": self._usage,
            "choices": [self._choices[i] for i in sorted(self._choices)],
        }

import os
from collections.abc import Iterable
from typing import TypedDict, cast

from opentelemetry.trace import Span
from typing_extensions import Any, NotRequired

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    set_span_attribute,
    to_dict,
)
from lmnr.sdk.utils import json_dumps

TERMINAL_RESPONSE_EVENTS = (
    "response.completed",
    "response.incomplete",
    "response.failed",
)

# A `responses` result in one of these states carries no usable completion.
ERROR_RESPONSE_STATUSES = ("failed", "incomplete")


def should_send_prompts() -> bool:
    return (os.getenv("LMNR_TRACE_CONTENT") or "true").lower() == "true"


def _to_dicts(items: list[Any]) -> list[dict[str, Any]]:
    return [item if isinstance(item, dict) else to_dict(item) for item in items]


def _aliased(d: dict[str, Any], key: str) -> Any:
    """Speakeasy models dump `schema`/`format` as `schema_`/`format_`."""
    return d.get(key, d.get(f"{key}_"))


def _set_structured_output_schema(span: Span, schema: dict[str, Any] | None):
    if schema:
        set_span_attribute(
            span, "gen_ai.request.structured_output_schema", json_dumps(schema)
        )


def _set_common_request_attributes(span: Span, kwargs: dict[str, Any]):
    set_span_attribute(span, "gen_ai.request.model", kwargs.get("model"))
    set_span_attribute(span, "gen_ai.request.temperature", kwargs.get("temperature"))
    set_span_attribute(span, "gen_ai.request.top_p", kwargs.get("top_p"))
    set_span_attribute(
        span, "gen_ai.request.frequency_penalty", kwargs.get("frequency_penalty")
    )
    set_span_attribute(
        span, "gen_ai.request.presence_penalty", kwargs.get("presence_penalty")
    )
    set_span_attribute(span, "llm.user", kwargs.get("user"))
    # `chat` takes a flat `reasoning_effort`, `responses` nests it under `reasoning`.
    reasoning_effort = kwargs.get("reasoning_effort") or to_dict(
        kwargs.get("reasoning")
    ).get("effort")
    set_span_attribute(span, "gen_ai.request.reasoning_effort", reasoning_effort)
    if kwargs.get("stream"):
        set_span_attribute(span, "llm.is_streaming", True)
    if kwargs.get("tools") and should_send_prompts():
        set_span_attribute(
            span, "gen_ai.tool.definitions", json_dumps(_to_dicts(kwargs["tools"]))
        )


@dont_throw
def set_chat_request_attributes(span: Span, kwargs: dict[str, Any]):
    _set_common_request_attributes(span, kwargs)
    set_span_attribute(span, "gen_ai.request.max_tokens", kwargs.get("max_tokens"))

    response_format = to_dict(kwargs.get("response_format"))
    if response_format.get("type") == "json_schema":
        json_schema = to_dict(response_format.get("json_schema"))
        _set_structured_output_schema(span, _aliased(json_schema, "schema"))

    if kwargs.get("messages") and should_send_prompts():
        set_span_attribute(
            span, "gen_ai.input.messages", json_dumps(_to_dicts(kwargs["messages"]))
        )


@dont_throw
def set_responses_request_attributes(span: Span, kwargs: dict[str, Any]):
    _set_common_request_attributes(span, kwargs)
    set_span_attribute(
        span, "gen_ai.request.max_tokens", kwargs.get("max_output_tokens")
    )

    text_format = to_dict(_aliased(to_dict(kwargs.get("text")), "format"))
    if text_format.get("type") == "json_schema":
        _set_structured_output_schema(span, _aliased(text_format, "schema"))

    if not should_send_prompts():
        return
    messages: list[dict[str, Any]] = []
    if kwargs.get("instructions"):
        messages.append({"role": "system", "content": kwargs["instructions"]})
    input_value = kwargs.get("input")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
    elif isinstance(input_value, list):
        messages.extend(_to_dicts(input_value))
    if messages:
        set_span_attribute(span, "gen_ai.input.messages", json_dumps(messages))


@dont_throw
def _embeddings_input_messages(input_value: Any) -> list[dict[str, Any]]:
    """`input` is either one document or a batch of them."""
    if not isinstance(input_value, list):
        return [{"content": input_value}]
    # A flat list of numbers is one token-id sequence, not a batch of documents.
    if isinstance(input_value[0], (int, float)):
        return [{"content": input_value}]
    return [{"content": document} for document in cast(Iterable[Any], input_value)]


def set_embeddings_request_attributes(span: Span, kwargs: dict[str, Any]):
    set_span_attribute(span, "gen_ai.request.model", kwargs.get("model"))
    set_span_attribute(span, "llm.user", kwargs.get("user"))
    input_value = kwargs.get("input")
    if input_value and should_send_prompts():
        set_span_attribute(
            span,
            "gen_ai.input.messages",
            json_dumps(_embeddings_input_messages(input_value)),
        )


def _set_usage_attributes(
    span: Span,
    usage: dict[str, int | float | dict[str, int | float]] | None,
    input_key: str,
    output_key: str,
    input_cost_key: str,
    output_cost_key: str,
):
    if not usage:
        return
    set_span_attribute(span, "gen_ai.usage.input_tokens", cast(int, usage.get(input_key)))
    set_span_attribute(span, "gen_ai.usage.output_tokens", cast(int, usage.get(output_key)))
    set_span_attribute(span, "llm.usage.total_tokens", cast(int, usage.get("total_tokens")))

    input_details = cast(dict[str, int], usage.get(f"{input_key}_details") or {})
    set_span_attribute(
        span, "gen_ai.usage.cache_read_input_tokens", input_details.get("cached_tokens")
    )
    set_span_attribute(
        span,
        "gen_ai.usage.cache_creation_input_tokens",
        input_details.get("cache_write_tokens"),
    )
    output_details = cast(dict[str, int], usage.get(f"{output_key}_details") or {})
    set_span_attribute(
        span, "gen_ai.usage.reasoning_tokens", output_details.get("reasoning_tokens")
    )

    set_span_attribute(span, "gen_ai.usage.cost", cast(float, usage.get("cost")))
    cost_details = cast(dict[str, float], usage.get("cost_details") or {})
    set_span_attribute(
        span, "gen_ai.usage.input_cost", cost_details.get(input_cost_key)
    )
    set_span_attribute(
        span, "gen_ai.usage.output_cost", cost_details.get(output_cost_key)
    )


@dont_throw
def set_chat_response_attributes(span: Span, response: dict[str, Any]):
    set_span_attribute(span, "gen_ai.response.id", response.get("id"))
    set_span_attribute(span, "gen_ai.response.model", response.get("model"))
    _set_usage_attributes(
        span,
        response.get("usage"),
        "prompt_tokens",
        "completion_tokens",
        "upstream_inference_prompt_cost",
        "upstream_inference_completions_cost",
    )
    if response.get("choices") and should_send_prompts():
        set_span_attribute(
            span, "gen_ai.output.messages", json_dumps(response["choices"])
        )


@dont_throw
def set_embeddings_response_attributes(span: Span, response: dict[str, Any]):
    set_span_attribute(span, "gen_ai.response.id", response.get("id"))
    set_span_attribute(span, "gen_ai.response.model", response.get("model"))
    _set_usage_attributes(
        span,
        response.get("usage"),
        "prompt_tokens",
        "completion_tokens",
        "upstream_inference_prompt_cost",
        "upstream_inference_completions_cost",
    )


@dont_throw
def set_responses_response_attributes(span: Span, response: dict[str, Any]):
    set_span_attribute(span, "gen_ai.response.id", response.get("id"))
    set_span_attribute(span, "gen_ai.response.model", response.get("model"))
    _set_usage_attributes(
        span,
        response.get("usage"),
        "input_tokens",
        "output_tokens",
        "upstream_inference_input_cost",
        "upstream_inference_output_cost",
    )
    if response.get("output") and should_send_prompts():
        set_span_attribute(
            span, "gen_ai.output.messages", json_dumps(response["output"])
        )


def responses_error_message(response: dict[str, Any]) -> str | None:
    """Message to fail the span with, or `None` if the response succeeded."""
    status = response.get("status")
    if status not in ERROR_RESPONSE_STATUSES:
        return None
    error = cast(dict[str, str], response.get("error") or {})
    incomplete_details = cast(dict[str, str], response.get("incomplete_details") or {})
    return error.get("message") or incomplete_details.get("reason") or status


class _AggregatedFunction(TypedDict):
    name: str
    arguments: str


class _AggregatedToolCall(TypedDict):
    id: str | None
    type: str
    function: _AggregatedFunction


class _AggregatedMessage(TypedDict):
    role: str
    content: str
    tool_calls: NotRequired[list[_AggregatedToolCall]]  # only when tools were called


class _AggregatedChoice(TypedDict):
    index: int
    message: _AggregatedMessage
    finish_reason: str | None


class _AggregatedChatCompletion(TypedDict):
    id: str | None
    model: str | None
    usage: dict[str, Any] | None  # taken as-is from the last chunk carrying it
    choices: list[_AggregatedChoice]


def aggregate_chat_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    result: _AggregatedChatCompletion = {
        "id": None,
        "model": None,
        "usage": None,
        "choices": [],
    }
    choices: dict[int, _AggregatedChoice] = {}
    tool_calls: dict[int, dict[int, _AggregatedToolCall]] = {}

    for chunk in chunks:
        result["id"] = result["id"] or chunk.get("id")
        result["model"] = result["model"] or chunk.get("model")
        if chunk.get("usage"):
            result["usage"] = chunk["usage"]

        for choice in cast(list[dict[str, Any]], chunk.get("choices") or []):
            index = choice.get("index") or 0
            accumulated = choices.setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                },
            )
            delta = cast(dict[str, Any], choice.get("delta") or {})
            if delta.get("role"):
                accumulated["message"]["role"] = delta["role"]
            if delta.get("content"):
                accumulated["message"]["content"] += delta["content"]
            for call in cast(list[dict[str, Any]], delta.get("tool_calls") or []):
                slot = tool_calls.setdefault(index, {}).setdefault(
                    call.get("index") or 0,
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                slot["id"] = slot["id"] or call.get("id")
                function = cast(dict[str, str], call.get("function") or {})
                slot["function"]["name"] += function.get("name") or ""
                slot["function"]["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                accumulated["finish_reason"] = choice["finish_reason"]

    for index, calls in tool_calls.items():
        choices[index]["message"]["tool_calls"] = [calls[i] for i in sorted(calls)]
    result["choices"] = [choices[i] for i in sorted(choices)]
    # Callers treat the response as a plain dict (same shape as a parsed response).
    return cast("dict[str, Any]", cast(object, result))


def response_from_stream_events(
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("type") in TERMINAL_RESPONSE_EVENTS:
            return event.get("response")
    return None

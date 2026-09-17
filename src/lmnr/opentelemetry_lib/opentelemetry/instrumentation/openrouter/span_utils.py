from opentelemetry.trace import Span

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


def _to_dicts(items: list) -> list:
    return [item if isinstance(item, dict) else to_dict(item) for item in items]


def _set_common_request_attributes(span: Span, kwargs: dict):
    set_span_attribute(span, "gen_ai.request.model", kwargs.get("model"))
    set_span_attribute(span, "gen_ai.request.temperature", kwargs.get("temperature"))
    set_span_attribute(span, "gen_ai.request.top_p", kwargs.get("top_p"))
    if kwargs.get("stream"):
        set_span_attribute(span, "llm.is_streaming", True)
    if kwargs.get("tools"):
        set_span_attribute(
            span, "gen_ai.tool.definitions", json_dumps(_to_dicts(kwargs["tools"]))
        )


@dont_throw
def set_chat_request_attributes(span: Span, kwargs: dict):
    _set_common_request_attributes(span, kwargs)
    set_span_attribute(span, "gen_ai.request.max_tokens", kwargs.get("max_tokens"))
    if kwargs.get("messages"):
        set_span_attribute(
            span, "gen_ai.input.messages", json_dumps(_to_dicts(kwargs["messages"]))
        )


@dont_throw
def set_responses_request_attributes(span: Span, kwargs: dict):
    _set_common_request_attributes(span, kwargs)
    set_span_attribute(
        span, "gen_ai.request.max_tokens", kwargs.get("max_output_tokens")
    )

    messages = []
    if kwargs.get("instructions"):
        messages.append({"role": "system", "content": kwargs["instructions"]})
    input_value = kwargs.get("input")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
    elif isinstance(input_value, list):
        messages.extend(_to_dicts(input_value))
    if messages:
        set_span_attribute(span, "gen_ai.input.messages", json_dumps(messages))


def _set_usage_attributes(
    span: Span,
    usage: dict | None,
    input_key: str,
    output_key: str,
    input_cost_key: str,
    output_cost_key: str,
):
    if not usage:
        return
    set_span_attribute(span, "gen_ai.usage.input_tokens", usage.get(input_key))
    set_span_attribute(span, "gen_ai.usage.output_tokens", usage.get(output_key))
    set_span_attribute(span, "llm.usage.total_tokens", usage.get("total_tokens"))

    input_details = usage.get(f"{input_key}_details") or {}
    set_span_attribute(
        span, "gen_ai.usage.cache_read_input_tokens", input_details.get("cached_tokens")
    )
    set_span_attribute(
        span,
        "gen_ai.usage.cache_creation_input_tokens",
        input_details.get("cache_write_tokens"),
    )
    output_details = usage.get(f"{output_key}_details") or {}
    set_span_attribute(
        span, "gen_ai.usage.reasoning_tokens", output_details.get("reasoning_tokens")
    )

    set_span_attribute(span, "gen_ai.usage.cost", usage.get("cost"))
    cost_details = usage.get("cost_details") or {}
    set_span_attribute(
        span, "gen_ai.usage.input_cost", cost_details.get(input_cost_key)
    )
    set_span_attribute(
        span, "gen_ai.usage.output_cost", cost_details.get(output_cost_key)
    )


@dont_throw
def set_chat_response_attributes(span: Span, response: dict):
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
    if response.get("choices"):
        set_span_attribute(
            span, "gen_ai.output.messages", json_dumps(response["choices"])
        )


@dont_throw
def set_responses_response_attributes(span: Span, response: dict):
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
    if response.get("output"):
        set_span_attribute(
            span, "gen_ai.output.messages", json_dumps(response["output"])
        )


def aggregate_chat_chunks(chunks: list[dict]) -> dict:
    result: dict = {"id": None, "model": None, "usage": None}
    choices: dict[int, dict] = {}
    tool_calls: dict[int, dict[int, dict]] = {}

    for chunk in chunks:
        result["id"] = result["id"] or chunk.get("id")
        result["model"] = result["model"] or chunk.get("model")
        if chunk.get("usage"):
            result["usage"] = chunk["usage"]

        for choice in chunk.get("choices") or []:
            index = choice.get("index") or 0
            accumulated = choices.setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                },
            )
            delta = choice.get("delta") or {}
            if delta.get("role"):
                accumulated["message"]["role"] = delta["role"]
            if delta.get("content"):
                accumulated["message"]["content"] += delta["content"]
            for call in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(index, {}).setdefault(
                    call.get("index") or 0,
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                slot["id"] = slot["id"] or call.get("id")
                function = call.get("function") or {}
                slot["function"]["name"] += function.get("name") or ""
                slot["function"]["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                accumulated["finish_reason"] = choice["finish_reason"]

    for index, calls in tool_calls.items():
        choices[index]["message"]["tool_calls"] = [calls[i] for i in sorted(calls)]
    result["choices"] = [choices[i] for i in sorted(choices)]
    return result


def response_from_stream_events(events: list[dict]) -> dict | None:
    for event in reversed(events):
        if event.get("type") in TERMINAL_RESPONSE_EVENTS:
            return event.get("response")
    return None

import os

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

# A `responses` result in one of these states carries no usable completion.
ERROR_RESPONSE_STATUSES = ("failed", "incomplete")


def should_send_prompts() -> bool:
    return (os.getenv("LMNR_TRACE_CONTENT") or "true").lower() == "true"


def _to_dicts(items: list) -> list:
    return [item if isinstance(item, dict) else to_dict(item) for item in items]


def _aliased(d: dict, key: str):
    """Speakeasy models dump `schema`/`format` as `schema_`/`format_`."""
    return d.get(key, d.get(f"{key}_"))


def _set_structured_output_schema(span: Span, schema: dict | None):
    if schema:
        set_span_attribute(
            span, "gen_ai.request.structured_output_schema", json_dumps(schema)
        )


def _set_common_request_attributes(span: Span, kwargs: dict):
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
def set_chat_request_attributes(span: Span, kwargs: dict):
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
def set_responses_request_attributes(span: Span, kwargs: dict):
    _set_common_request_attributes(span, kwargs)
    set_span_attribute(
        span, "gen_ai.request.max_tokens", kwargs.get("max_output_tokens")
    )

    text_format = to_dict(_aliased(to_dict(kwargs.get("text")), "format"))
    if text_format.get("type") == "json_schema":
        _set_structured_output_schema(span, _aliased(text_format, "schema"))

    if not should_send_prompts():
        return
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


@dont_throw
def set_embeddings_request_attributes(span: Span, kwargs: dict):
    set_span_attribute(span, "gen_ai.request.model", kwargs.get("model"))
    set_span_attribute(span, "llm.user", kwargs.get("user"))
    input_value = kwargs.get("input")
    if input_value and should_send_prompts():
        items = input_value if isinstance(input_value, list) else [input_value]
        set_span_attribute(
            span, "gen_ai.input.messages", json_dumps([{"content": i} for i in items])
        )


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
    if response.get("choices") and should_send_prompts():
        set_span_attribute(
            span, "gen_ai.output.messages", json_dumps(response["choices"])
        )


@dont_throw
def set_embeddings_response_attributes(span: Span, response: dict):
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
    if response.get("output") and should_send_prompts():
        set_span_attribute(
            span, "gen_ai.output.messages", json_dumps(response["output"])
        )


def responses_error_message(response: dict) -> str | None:
    """Message to fail the span with, or `None` if the response succeeded."""
    status = response.get("status")
    if status not in ERROR_RESPONSE_STATUSES:
        return None
    error = response.get("error") or {}
    incomplete_details = response.get("incomplete_details") or {}
    return error.get("message") or incomplete_details.get("reason") or status


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

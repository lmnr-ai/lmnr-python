from typing import Any, Sequence

from opentelemetry.trace import Span
from pydantic import BaseModel

from lmnr.sdk.utils import json_dumps
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    extract_json_schema,
    set_span_attribute,
    to_dict,
)
from ...utils import infer_provider


@dont_throw
def process_completion_kwargs(
    span: Span,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
):
    if kwargs is None:
        kwargs = {}
    model = args[0] if args else kwargs.get("model")
    set_span_attribute(span, "gen_ai.request.model", model)
    set_span_attribute(span, "gen_ai.system", infer_provider(model))
    set_span_attribute(span, "gen_ai.request.temperature", kwargs.get("temperature"))
    set_span_attribute(span, "gen_ai.request.top_p", kwargs.get("top_p"))
    set_span_attribute(span, "gen_ai.request.top_k", kwargs.get("top_logprobs"))
    set_span_attribute(
        span,
        "gen_ai.request.max_tokens",
        kwargs.get("max_tokens", kwargs.get("max_completion_tokens")),
    )
    set_span_attribute(span, "gen_ai.request.stop_sequences", kwargs.get("stop"))
    set_span_attribute(
        span, "gen_ai.request.frequency_penalty", kwargs.get("frequency_penalty")
    )
    set_span_attribute(
        span, "gen_ai.request.presence_penalty", kwargs.get("presence_penalty")
    )
    set_span_attribute(
        span, "gen_ai.request.reasoning_effort", kwargs.get("reasoning_effort")
    )
    set_span_attribute(
        span,
        "gen_ai.request.thinking_budget",
        (kwargs.get("thinking") or {}).get("budget_tokens"),
    )
    set_span_attribute(
        span,
        "gen_ai.request.base_url",
        kwargs.get("base_url"),
    )
    if kwargs.get("response_format"):
        set_span_attribute(
            span,
            "gen_ai.request.structured_output_schema",
            json_dumps(extract_json_schema(kwargs.get("response_format"))),
        )


@dont_throw
def process_completion_inputs(
    span: Span,
    messages: list[dict],
    tools: list[dict] | None = None,
):
    # this for loop replicates `litellm.utils.validate_and_fix_openai_messages`
    attr_messages = []
    for msg in messages:
        message = to_dict(msg)
        if not message.get("role"):
            message["role"] = "assistant"
        if isinstance(message.get("tool_calls"), list):
            message["tool_calls"] = [
                to_dict(tool_call) for tool_call in message.get("tool_calls")
            ]
        attr_messages.append(message)

    span.set_attribute("gen_ai.input.messages", json_dumps(attr_messages))

    if tools:
        attr_tools = [to_dict(tool) for tool in tools]
        span.set_attribute("gen_ai.tool.definitions", json_dumps(attr_tools))


@dont_throw
def process_completion_response(
    span: Span,
    response: BaseModel,
    record_raw_response: bool = False,
):
    response_dict = to_dict(response)
    set_span_attribute(span, "gen_ai.response.id", response_dict.get("id"))
    set_span_attribute(span, "gen_ai.response.model", response_dict.get("model"))
    set_span_attribute(
        span,
        "gen_ai.response.system_fingerprint",
        response_dict.get("system_fingerprint"),
    )
    choices = response_dict.get("choices", [])
    messages = [choice.get("message", {}) for choice in choices]
    span.set_attribute("gen_ai.output.messages", json_dumps(messages))
    usage = to_dict(response_dict.get("usage", {}))
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
    set_span_attribute(span, "gen_ai.usage.input_tokens", input_tokens)
    set_span_attribute(span, "gen_ai.usage.output_tokens", output_tokens)
    set_span_attribute(span, "llm.usage.total_tokens", total_tokens)
    if input_details := usage.get("prompt_tokens_details"):
        details = to_dict(input_details)
        cache_read_tokens = details.get(
            "cached_tokens", usage.get("cache_read_input_tokens", 0)
        )
        cache_creation_tokens = details.get(
            "cache_creation_tokens", usage.get("cache_creation_input_tokens", 0)
        )
        set_span_attribute(
            span, "gen_ai.usage.cache_read_input_tokens", cache_read_tokens
        )
        set_span_attribute(
            span, "gen_ai.usage.cache_creation_input_tokens", cache_creation_tokens
        )

    # Record raw response in rollout mode
    if record_raw_response:
        set_span_attribute(span, "lmnr.sdk.raw.response", json_dumps(response_dict))

    return response

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
def process_responses_kwargs(
    span: Span,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
):
    """Process responses kwargs and set span attributes."""
    if kwargs is None:
        kwargs = {}

    # Get model - responses() has model as second arg or model kwarg
    model = args[1] if len(args) > 1 else kwargs.get("model")
    set_span_attribute(span, "gen_ai.request.model", model)
    set_span_attribute(span, "gen_ai.system", infer_provider(model))

    # Common parameters
    set_span_attribute(span, "gen_ai.request.temperature", kwargs.get("temperature"))
    set_span_attribute(span, "gen_ai.request.top_p", kwargs.get("top_p"))

    # responses uses max_output_tokens instead of max_tokens
    set_span_attribute(
        span,
        "gen_ai.request.max_tokens",
        kwargs.get("max_output_tokens"),
    )

    # reasoning parameter for o1-style models
    if reasoning := kwargs.get("reasoning"):
        set_span_attribute(
            span,
            "gen_ai.request.reasoning_effort",
            reasoning.get("effort") if isinstance(reasoning, dict) else None,
        )

    # Structured output via text_format
    if text_format := kwargs.get("text_format"):
        set_span_attribute(
            span,
            "gen_ai.request.structured_output_schema",
            json_dumps(extract_json_schema(text_format)),
        )

    # Other responses-specific parameters
    set_span_attribute(span, "gen_ai.request.instructions", kwargs.get("instructions"))
    set_span_attribute(span, "gen_ai.request.store", kwargs.get("store"))
    set_span_attribute(span, "gen_ai.request.truncation", kwargs.get("truncation"))
    set_span_attribute(span, "gen_ai.request.service_tier", kwargs.get("service_tier"))
    set_span_attribute(
        span, "gen_ai.request.previous_response_id", kwargs.get("previous_response_id")
    )


@dont_throw
def process_responses_inputs(
    span: Span,
    input_param: Any,
    tools: list[dict] | None = None,
):
    """Process responses input and tools."""
    # this for loop replicates `litellm.utils.validate_and_fix_openai_messages`
    if tools and isinstance(tools, list):
        attr_tools = [to_dict(tool) for tool in tools]
        span.set_attribute("gen_ai.tool.definitions", json_dumps(attr_tools))

    attr_input_items = []
    if not isinstance(input_param, list):
        return
    for item in input_param:
        item = to_dict(item)
        attr_input_items.append(to_dict(item))

    span.set_attribute("gen_ai.input.messages", json_dumps(attr_input_items))


@dont_throw
def process_responses_response(
    span: Span,
    response: BaseModel,
):
    """Process responses response."""
    response_dict = to_dict(response)
    set_span_attribute(span, "gen_ai.response.id", response_dict.get("id"))
    set_span_attribute(span, "gen_ai.response.model", response_dict.get("model"))

    if usage := response_dict.get("usage"):
        usage_dict = to_dict(usage)
        input_tokens = usage_dict.get(
            "input_tokens", usage_dict.get("prompt_tokens", 0)
        )
        output_tokens = usage_dict.get(
            "output_tokens", usage_dict.get("completion_tokens", 0)
        )
        total_tokens = usage_dict.get("total_tokens", input_tokens + output_tokens)
        set_span_attribute(span, "gen_ai.usage.input_tokens", input_tokens)
        set_span_attribute(span, "gen_ai.usage.output_tokens", output_tokens)
        set_span_attribute(span, "llm.usage.total_tokens", total_tokens)
        if input_details := usage_dict.get("input_tokens_details"):
            details = to_dict(input_details)
            cache_read_tokens = details.get("cached_tokens", 0)
            set_span_attribute(
                span, "gen_ai.usage.cache_read_input_tokens", cache_read_tokens
            )

    final_items = []
    if reasoning := response_dict.get("reasoning"):
        reasoning_dict = to_dict(reasoning)
        if reasoning_dict.get("summary") or reasoning_dict.get("effort"):
            final_items.append(reasoning_dict)
    if isinstance(response_dict.get("output"), list):
        for item in response_dict.get("output"):
            item = to_dict(item)
            final_items.append(item)

    span.set_attribute("gen_ai.output.messages", json_dumps(final_items))

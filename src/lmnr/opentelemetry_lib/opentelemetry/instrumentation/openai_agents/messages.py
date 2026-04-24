"""gen_ai message helpers, tool definitions, and LLM attribute helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmnr.opentelemetry_lib.tracing.span import LaminarSpan

from lmnr.opentelemetry_lib.tracing.attributes import Attributes
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import json_dumps

from .helpers import (
    get_attr_not_none,
    get_first_not_none,
    model_as_dict,
    normalize_messages,
    to_dict,
)

logger = get_default_logger(__name__)

# ---------------------------------------------------------------------------
# gen_ai.input.messages / gen_ai.output.messages helpers
# ---------------------------------------------------------------------------


def set_lmnr_span_io(
    lmnr_span: LaminarSpan,
    input_data: Any,
    output_data: Any,
) -> None:
    """Set gen_ai.input.messages and gen_ai.output.messages on the span."""
    if input_data is not None:
        lmnr_span.set_attribute("lmnr.span.input", json_dumps(input_data))
    if output_data is not None:
        lmnr_span.set_attribute("lmnr.span.output", json_dumps(output_data))


def set_gen_ai_input_messages(
    lmnr_span: LaminarSpan,
    input_data: Any,
    system_instructions: str | None = None,
) -> None:
    """Set gen_ai.input.messages on the span.

    If system_instructions is provided, prepend a system message so the span
    reflects the full prompt actually sent to the model.
    """
    if input_data is None and not system_instructions:
        return
    messages = normalize_messages(input_data) if input_data is not None else []
    if system_instructions:
        messages.insert(
            0,
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_instructions}],
            },
        )
    if messages:
        lmnr_span.set_attribute("gen_ai.input.messages", json_dumps(messages))


def set_gen_ai_output_messages(lmnr_span: LaminarSpan, output_data: Any) -> None:
    """Set gen_ai.output.messages on the span."""
    if output_data is None:
        return
    messages = normalize_messages(output_data, role="assistant")
    if messages:
        lmnr_span.set_attribute("gen_ai.output.messages", json_dumps(messages))


def set_gen_ai_output_messages_from_response(
    lmnr_span: LaminarSpan, response: Any
) -> None:
    """Extract and set gen_ai.output.messages from a Response object."""
    if response is None:
        return

    output_items = getattr(response, "output", None)
    if not output_items:
        return

    id = getattr(response, "id", None)

    if not isinstance(output_items, list):
        logger.debug(
            "Laminar OpenAI agents instrumentation, failed to parse output items. Expected array"
        )
        output_items = []

    result = {
        "id": id,
        "object": "response",
        "output": output_items,
    }

    lmnr_span.set_attribute("gen_ai.output.messages", json_dumps(result))


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def set_tool_definitions_from_response(lmnr_span: LaminarSpan, response: Any) -> None:
    """Extract gen_ai.tool.definitions from a Response object's tools field."""
    tools = getattr(response, "tools", None)
    if not tools:
        return

    tool_defs = []
    for tool in tools:
        tool_dict = model_as_dict(tool)
        if not tool_dict:
            continue
        tool_type = tool_dict.get("type")
        if tool_type == "function":
            func_def = {}
            func_def["type"] = "function"
            function_info = tool_dict.get("function")
            if function_info is None:
                function_info = tool_dict
            func_def["function"] = {
                "name": function_info.get("name", ""),
            }
            desc = function_info.get("description")
            if desc:
                func_def["function"]["description"] = desc
            params = function_info.get("parameters")
            if params:
                func_def["function"]["parameters"] = params
            strict = function_info.get("strict")
            if strict is not None:
                func_def["function"]["strict"] = strict
            tool_defs.append(func_def)
        else:
            tool_defs.append(tool_dict)

    if tool_defs:
        lmnr_span.set_attribute("gen_ai.tool.definitions", json_dumps(tool_defs))


# ---------------------------------------------------------------------------
# LLM attributes (model, usage, response_id)
# ---------------------------------------------------------------------------


def apply_llm_attributes(lmnr_span: LaminarSpan, data: dict[str, Any]) -> None:
    if not data:
        return

    model = data.get("model")
    if model:
        lmnr_span.set_attribute(Attributes.REQUEST_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.RESPONSE_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.PROVIDER.value, "openai")

    usage = data.get("usage")
    if usage is not None:
        _apply_usage(lmnr_span, usage)

    response_id = data.get("response_id")
    if response_id is None:
        response_id = data.get("id")
    if response_id is not None:
        lmnr_span.set_attribute(Attributes.RESPONSE_ID.value, response_id)


def _apply_usage(lmnr_span: LaminarSpan, usage: Any) -> None:
    """Extract token usage from a usage object or dict, handling zero correctly."""
    if usage is None:
        return
    cached_input_tokens = 0
    reasoning_output_tokens = 0
    input_tokens_details = None
    output_tokens_details = None

    if isinstance(usage, dict):
        input_tokens = get_first_not_none(
            usage, "input_tokens", "prompt_tokens", "input"
        )
        output_tokens = get_first_not_none(
            usage, "output_tokens", "completion_tokens", "output"
        )
        total_tokens = get_first_not_none(usage, "total_tokens", "total")
        input_tokens_details = get_first_not_none(
            usage, "input_tokens_details", "prompt_tokens_details"
        )
        output_tokens_details = get_first_not_none(
            usage, "output_tokens_details", "completion_tokens_details"
        )
    else:
        # Object with attributes (e.g. ResponseUsage)
        input_tokens = get_attr_not_none(usage, "input_tokens", "prompt_tokens")
        output_tokens = get_attr_not_none(usage, "output_tokens", "completion_tokens")
        total_tokens = get_attr_not_none(usage, "total_tokens")
        input_tokens_details = get_attr_not_none(
            usage, "input_tokens_details", "prompt_tokens_details"
        )
        output_tokens_details = get_attr_not_none(
            usage, "output_tokens_details", "completion_tokens_details"
        )

    if input_tokens_details:
        cached_input_tokens = to_dict(input_tokens_details).get("cached_tokens", 0)
    if output_tokens_details:
        reasoning_output_tokens = to_dict(output_tokens_details).get("reasoning_tokens")
    if input_tokens is not None:
        lmnr_span.set_attribute(Attributes.INPUT_TOKEN_COUNT.value, input_tokens)
    if cached_input_tokens:
        lmnr_span.set_attribute(
            "gen_ai.usage.cache_read_input_tokens", cached_input_tokens
        )
    if output_tokens is not None:
        lmnr_span.set_attribute(Attributes.OUTPUT_TOKEN_COUNT.value, output_tokens)
    if reasoning_output_tokens:
        lmnr_span.set_attribute(
            "gen_ai.usage.reasoning_output_tokens", reasoning_output_tokens
        )
    if total_tokens is not None:
        lmnr_span.set_attribute(Attributes.TOTAL_TOKEN_COUNT.value, total_tokens)
    elif input_tokens is not None and output_tokens is not None:
        lmnr_span.set_attribute(
            Attributes.TOTAL_TOKEN_COUNT.value, input_tokens + output_tokens
        )


def response_to_llm_data(response: Any) -> dict[str, Any]:
    if response is None:
        return {}
    return {
        "model": getattr(response, "model", None),
        "usage": getattr(response, "usage", None),
        "response_id": getattr(response, "id", None),
    }

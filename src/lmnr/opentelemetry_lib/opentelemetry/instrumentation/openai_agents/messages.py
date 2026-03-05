"""gen_ai message helpers, tool definitions, and LLM attribute helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from lmnr.opentelemetry_lib.tracing.attributes import Attributes
from lmnr.sdk.utils import json_dumps

from .helpers import (
    get_attr_not_none,
    get_first_not_none,
    model_as_dict,
    normalize_messages,
)


# ---------------------------------------------------------------------------
# gen_ai.input.messages / gen_ai.output.messages helpers
# ---------------------------------------------------------------------------


def _set_gen_ai_messages(
    lmnr_span: Any,
    input_data: Any,
    output_data: Any,
) -> None:
    """Set gen_ai.input.messages and gen_ai.output.messages on the span."""
    if input_data is not None:
        _set_gen_ai_input_messages(lmnr_span, input_data)
    if output_data is not None:
        _set_gen_ai_output_messages(lmnr_span, output_data)


def _set_gen_ai_input_messages(lmnr_span: Any, input_data: Any) -> None:
    """Set gen_ai.input.messages on the span."""
    if input_data is None:
        return
    if not hasattr(lmnr_span, "set_attribute"):
        return

    messages = normalize_messages(input_data)
    if messages:
        lmnr_span.set_attribute("gen_ai.input.messages", json_dumps(messages))


def _set_gen_ai_output_messages(lmnr_span: Any, output_data: Any) -> None:
    """Set gen_ai.output.messages on the span."""
    if output_data is None:
        return
    if not hasattr(lmnr_span, "set_attribute"):
        return

    messages = normalize_messages(output_data, role="assistant")
    if messages:
        lmnr_span.set_attribute("gen_ai.output.messages", json_dumps(messages))


def _set_gen_ai_output_messages_from_response(lmnr_span: Any, response: Any) -> None:
    """Extract and set gen_ai.output.messages from a Response object."""
    if response is None:
        return
    if not hasattr(lmnr_span, "set_attribute"):
        return

    output_items = getattr(response, "output", None)
    if not output_items:
        return

    messages: List[Dict[str, Any]] = []
    for item in output_items:
        item_dict = model_as_dict(item)
        if not item_dict:
            continue
        item_type = item_dict.get("type")
        if item_type == "message":
            content_list = item_dict.get("content", [])
            text_parts = []
            for content in content_list if isinstance(content_list, list) else []:
                if isinstance(content, dict):
                    ct = content.get("type", "")
                    if ct in ("output_text", "text"):
                        text_parts.append(content.get("text", ""))
                else:
                    ct = getattr(content, "type", "")
                    if ct in ("output_text", "text"):
                        text_parts.append(getattr(content, "text", ""))
            if text_parts:
                messages.append(
                    {
                        "role": item_dict.get("role", "assistant"),
                        "content": "".join(text_parts),
                    }
                )
        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item_dict.get("call_id", item_dict.get("id", "")),
                            "type": "function",
                            "function": {
                                "name": item_dict.get("name", ""),
                                "arguments": item_dict.get("arguments", ""),
                            },
                        }
                    ],
                }
            )

    if messages:
        lmnr_span.set_attribute("gen_ai.output.messages", json_dumps(messages))


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def _set_tool_definitions_from_response(lmnr_span: Any, response: Any) -> None:
    """Extract gen_ai.tool.definitions from a Response object's tools field."""
    if not hasattr(lmnr_span, "set_attribute"):
        return

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
            function_info = tool_dict.get("function") or tool_dict
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


def _apply_llm_attributes(lmnr_span: Any, data: Optional[Dict[str, Any]]) -> None:
    if not data or not hasattr(lmnr_span, "set_attribute"):
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


def _apply_usage(lmnr_span: Any, usage: Any) -> None:
    """Extract token usage from a usage object or dict, handling zero correctly."""
    if usage is None:
        return

    if isinstance(usage, dict):
        input_tokens = get_first_not_none(
            usage, "input_tokens", "prompt_tokens", "input"
        )
        output_tokens = get_first_not_none(
            usage, "output_tokens", "completion_tokens", "output"
        )
        total_tokens = get_first_not_none(usage, "total_tokens", "total")
    else:
        # Object with attributes (e.g. ResponseUsage)
        input_tokens = get_attr_not_none(usage, "input_tokens", "prompt_tokens")
        output_tokens = get_attr_not_none(usage, "output_tokens", "completion_tokens")
        total_tokens = get_attr_not_none(usage, "total_tokens")

    if input_tokens is not None:
        lmnr_span.set_attribute(Attributes.INPUT_TOKEN_COUNT.value, input_tokens)
    if output_tokens is not None:
        lmnr_span.set_attribute(Attributes.OUTPUT_TOKEN_COUNT.value, output_tokens)
    if total_tokens is not None:
        lmnr_span.set_attribute(Attributes.TOTAL_TOKEN_COUNT.value, total_tokens)
    elif input_tokens is not None and output_tokens is not None:
        lmnr_span.set_attribute(
            Attributes.TOTAL_TOKEN_COUNT.value, input_tokens + output_tokens
        )


def _response_to_llm_data(response: Any) -> Dict[str, Any]:
    if response is None:
        return {}
    return {
        "model": getattr(response, "model", None),
        "usage": getattr(response, "usage", None),
        "response_id": getattr(response, "id", None),
    }

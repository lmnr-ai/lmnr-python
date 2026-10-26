"""Translate LangChain callback payloads into span attributes.

LLM spans carry the OTel GenAI message shape (`{role, parts: [...]}` arrays in
`gen_ai.input.messages` / `gen_ai.output.messages`), which the Laminar backend
preserves end-to-end. Chain, tool and retriever spans carry their raw inputs and
outputs in `lmnr.span.input` / `lmnr.span.output`.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any, cast
from uuid import UUID

from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry.trace import Span

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.reasoning import (
    set_reasoning_attributes,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    set_span_attribute,
    should_send_prompts,
)
from lmnr.sdk.utils import json_dumps

_ROLE_BY_MESSAGE_TYPE = {
    "human": "user",
    "system": "system",
    "ai": "assistant",
    "tool": "tool",
    "function": "tool",
}

_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_call",
    "function_call": "tool_call",
    "content_filter": "content_filter",
    # Anthropic, surfaced through langchain-anthropic
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_call",
    "max_tokens": "length",
}

_MODEL_KEYS = ("model", "model_name", "model_id")


def message_role(message_type: str) -> str:
    return _ROLE_BY_MESSAGE_TYPE.get(message_type, message_type or "unknown")


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def content_to_parts(content: Any) -> list[dict[str, Any]]:
    """Convert LangChain message content (str or list of blocks) into GenAI parts."""
    if isinstance(content, str):
        return [{"type": "text", "content": content}] if content else []
    if not isinstance(content, list):
        return [{"type": "text", "content": str(content)}] if content else []

    parts: list[dict[str, Any]] = []
    for block in cast(list[Any], content):
        if isinstance(block, str):
            if block:
                parts.append({"type": "text", "content": block})
            continue
        if not isinstance(block, dict):
            parts.append({"type": "text", "content": str(block)})
            continue
        block = cast(dict[str, Any], block)
        block_type = block.get("type")
        if block_type == "text":
            if text := block.get("text"):
                parts.append({"type": "text", "content": text})
        elif block_type == "image_url":
            image_url = block.get("image_url")
            url = (
                cast(dict[str, Any], image_url).get("url")
                if isinstance(image_url, dict)
                else image_url
            )
            if url:
                parts.append({"type": "uri", "modality": "image", "uri": url})
        elif block_type == "image" and block.get("url"):
            parts.append({"type": "uri", "modality": "image", "uri": block["url"]})
        elif block_type == "image":
            parts.append(
                {
                    "type": "blob",
                    "modality": "image",
                    "mime_type": block.get("mime_type") or block.get("media_type") or "",
                    "content": block.get("data") or block.get("base64") or "",
                }
            )
        else:
            # Provider-specific block (reasoning, tool_use, ...): keep it whole
            # rather than guessing at a mapping.
            parts.append(block)
    return parts


def tool_calls_to_parts(tool_calls: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if not isinstance(tool_calls, list):
        return parts
    for tool_call in cast(list[Any], tool_calls):
        if not isinstance(tool_call, Mapping):
            continue
        tool_call = cast(Mapping[str, Any], tool_call)
        raw_function = tool_call.get("function")
        function: Mapping[str, Any] = (
            cast(Mapping[str, Any], raw_function) if isinstance(raw_function, Mapping) else {}
        )
        part: dict[str, Any] = {
            "type": "tool_call",
            "id": tool_call.get("id") or "",
            "name": tool_call.get("name") or function.get("name") or "",
        }
        arguments = tool_call.get("args", function.get("arguments"))
        if arguments is not None:
            part["arguments"] = _maybe_json(arguments)
        parts.append(part)
    return parts


def _message_tool_calls(message: BaseMessage) -> Any:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return tool_calls
    return message.additional_kwargs.get("tool_calls")


def message_to_dict(message: BaseMessage) -> dict[str, Any]:
    role = message_role(message.type)
    tool_call_id = getattr(message, "tool_call_id", None)
    if role == "tool" and tool_call_id:
        parts: list[dict[str, Any]] = [
            {"type": "tool_call_response", "id": tool_call_id, "response": message.content}
        ]
    else:
        parts = content_to_parts(message.content)
        parts.extend(tool_calls_to_parts(_message_tool_calls(message)))
        function_call = message.additional_kwargs.get("function_call")
        if isinstance(function_call, Mapping):
            # Legacy OpenAI single function-call format
            function_call = cast(Mapping[str, Any], function_call)
            parts.append(
                {
                    "type": "tool_call",
                    "id": "",
                    "name": function_call.get("name") or "",
                    "arguments": _maybe_json(function_call.get("arguments")),
                }
            )
    return {"role": role, "parts": parts}


def _first_model(*sources: Mapping[str, Any] | None) -> str | None:
    for source in sources:
        if not source:
            continue
        for key in _MODEL_KEYS:
            model = source.get(key)
            if isinstance(model, str) and model:
                return model
    return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def set_request_attributes(
    span: Span,
    serialized: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> str | None:
    """Record the request model and sampling params. Returns the request model."""
    invocation_params = _as_mapping(kwargs.get("invocation_params"))
    # Some integrations nest the provider params one level deeper
    params = _as_mapping(invocation_params.get("params")) or invocation_params
    serialized_kwargs = _as_mapping((serialized or {}).get("kwargs"))

    model = _first_model(params, invocation_params, serialized_kwargs)
    if model is None and metadata:
        ls_model_name = metadata.get("ls_model_name")
        model = ls_model_name if isinstance(ls_model_name, str) and ls_model_name else None
    set_span_attribute(span, "gen_ai.request.model", model)

    def param(*keys: str) -> Any:
        for source in (params, serialized_kwargs):
            for key in keys:
                if (value := source.get(key)) is not None:
                    return value
        return None

    max_tokens = param("max_tokens", "max_completion_tokens", "max_new_tokens")
    temperature = param("temperature")
    top_p = param("top_p")
    if isinstance(max_tokens, int):
        set_span_attribute(span, "gen_ai.request.max_tokens", max_tokens)
    if isinstance(temperature, (int, float)):
        set_span_attribute(span, "gen_ai.request.temperature", temperature)
    if isinstance(top_p, (int, float)):
        set_span_attribute(span, "gen_ai.request.top_p", top_p)
    # Chat model fields as the integrations report them: `reasoning_effort` /
    # `reasoning` (OpenAI, Groq), `thinking` / `output_config` (Anthropic),
    # `thinking_budget` / `thinking_level` / `thinking_config` (Google GenAI).
    # Like `param`, a `None` invocation param falls through to the constructor kwargs
    reasoning_params = {
        key: value
        for source in (serialized_kwargs, params)
        for key, value in source.items()
        if value is not None
    }
    set_reasoning_attributes(span, reasoning_params)

    tools = params.get("tools") or params.get("functions")
    if should_send_prompts() and isinstance(tools, list) and tools:
        tool_definitions: list[dict[str, Any]] = []
        for tool in cast(list[Any], tools):
            tool = _as_mapping(tool)
            function = _as_mapping(tool.get("function")) or tool
            definition: dict[str, Any] = {"name": function.get("name")}
            if description := function.get("description"):
                definition["description"] = description
            if parameters := function.get("parameters") or tool.get("input_schema"):
                definition["parameters"] = parameters
            tool_definitions.append(definition)
        set_span_attribute(span, "gen_ai.tool.definitions", json_dumps(tool_definitions))

    return model


def set_chat_request(span: Span, messages: Sequence[Sequence[BaseMessage]]) -> None:
    if not should_send_prompts():
        return
    input_messages = [message_to_dict(message) for batch in messages for message in batch]
    if input_messages:
        set_span_attribute(span, "gen_ai.input.messages", json_dumps(input_messages))


def set_completion_request(span: Span, prompts: Sequence[str]) -> None:
    if not should_send_prompts():
        return
    input_messages = [
        {"role": "user", "parts": [{"type": "text", "content": prompt}]} for prompt in prompts
    ]
    if input_messages:
        set_span_attribute(span, "gen_ai.input.messages", json_dumps(input_messages))


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _set_usage(span: Span, response: LLMResult) -> None:
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    found = False

    for generations in response.generations:
        for generation in generations:
            message = getattr(generation, "message", None)
            usage = _as_mapping(getattr(message, "usage_metadata", None))
            if not usage:
                continue
            found = True
            # LangChain's `input_tokens` already includes cached tokens; the
            # details only break that total down.
            input_tokens += _int(usage.get("input_tokens")) or 0
            output_tokens += _int(usage.get("output_tokens")) or 0
            details = _as_mapping(usage.get("input_token_details"))
            if (cache_read := _int(details.get("cache_read"))) is not None:
                cache_read_tokens = (cache_read_tokens or 0) + cache_read
            if (cache_creation := _int(details.get("cache_creation"))) is not None:
                cache_creation_tokens = (cache_creation_tokens or 0) + cache_creation

    if not found:
        # Fall back to the provider-reported totals in `llm_output`, which
        # completion (non-chat) models and older integrations still use.
        llm_output = _as_mapping(response.llm_output)
        token_usage = _as_mapping(llm_output.get("token_usage") or llm_output.get("usage"))
        if not token_usage:
            return
        input_tokens = (
            _int(token_usage.get("prompt_tokens"))
            or _int(token_usage.get("input_tokens"))
            or _int(token_usage.get("input_token_count"))
            or 0
        )
        output_tokens = (
            _int(token_usage.get("completion_tokens"))
            or _int(token_usage.get("output_tokens"))
            or _int(token_usage.get("generated_token_count"))
            or 0
        )

    set_span_attribute(span, "gen_ai.usage.input_tokens", input_tokens)
    set_span_attribute(span, "gen_ai.usage.output_tokens", output_tokens)
    set_span_attribute(span, "llm.usage.total_tokens", input_tokens + output_tokens)
    set_span_attribute(span, "gen_ai.usage.cache_read_input_tokens", cache_read_tokens)
    set_span_attribute(span, "gen_ai.usage.cache_creation_input_tokens", cache_creation_tokens)


def _response_model_and_id(response: LLMResult) -> tuple[str | None, str | None]:
    llm_output = _as_mapping(response.llm_output)
    model = _first_model(llm_output)
    response_id = llm_output.get("id")
    for generations in response.generations:
        for generation in generations:
            message = getattr(generation, "message", None)
            if model is None:
                model = _first_model(_as_mapping(getattr(message, "response_metadata", None)))
            if response_id is None:
                response_id = getattr(message, "id", None)
    return model, response_id if isinstance(response_id, str) and response_id else None


def set_response_attributes(span: Span, response: LLMResult, request_model: str | None) -> None:
    model, response_id = _response_model_and_id(response)
    set_span_attribute(span, "gen_ai.response.model", model or request_model)
    if request_model is None:
        set_span_attribute(span, "gen_ai.request.model", model)
    set_span_attribute(span, "gen_ai.response.id", response_id)
    _set_usage(span, response)

    send_prompts = should_send_prompts()
    output_messages: list[dict[str, Any]] = []
    finish_reasons: list[str] = []
    for generations in response.generations:
        for generation in generations:
            generation_info = _as_mapping(generation.generation_info)
            raw_reason = generation_info.get("finish_reason") or generation_info.get("stop_reason")
            finish_reason = (
                _FINISH_REASON_MAP.get(raw_reason, raw_reason)
                if isinstance(raw_reason, str)
                else None
            )
            if finish_reason and finish_reason not in finish_reasons:
                finish_reasons.append(finish_reason)
            if not send_prompts:
                continue
            message = getattr(generation, "message", None)
            if isinstance(message, BaseMessage):
                output_message = message_to_dict(message)
            else:
                output_message = {
                    "role": "assistant",
                    "parts": content_to_parts(generation.text),
                }
            output_message["finish_reason"] = finish_reason or ""
            output_messages.append(output_message)

    if finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
    if output_messages:
        set_span_attribute(span, "gen_ai.output.messages", json_dumps(output_messages))


_METADATA_PRIMITIVES = (bool, str, int, float)
# Stdlib scalars whose str() is the value itself rather than constructor state
_METADATA_SCALARS = (UUID, datetime.date, datetime.time, Decimal, PurePath, Enum)


def _metadata_value(value: Any) -> Any:
    """Return plain-data metadata as is, or None for anything else.

    Arbitrary objects are dropped rather than stringified: `str()` on a model,
    client or config object renders its constructor state, which routinely
    includes credentials the caller never meant to export.
    """
    if isinstance(value, _METADATA_PRIMITIVES):
        return value
    if isinstance(value, _METADATA_SCALARS):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [
            item
            for item in (_metadata_value(element) for element in cast(Sequence[Any], value))
            if item is not None
        ]
    if isinstance(value, Mapping):
        kept = {
            key: item
            for key, item in (
                (key, _metadata_value(element))
                for key, element in cast(Mapping[Any, Any], value).items()
            )
            if isinstance(key, str) and item is not None
        }
        return kept or None
    return None


def sanitize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        key: sanitized
        for key, value in metadata.items()
        if (sanitized := _metadata_value(value)) is not None
    }


def retriever_output(documents: Sequence[Any]) -> list[Any]:
    output: list[Any] = []
    for document in documents:
        page_content = getattr(document, "page_content", None)
        if page_content is None:
            output.append(document)
            continue
        output.append(
            {
                "page_content": page_content,
                "metadata": getattr(document, "metadata", None) or {},
            }
        )
    return output

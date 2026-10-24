"""Span attributes for the Cohere SDK, as OTel GenAI `{role, parts}` messages."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, cast

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_SEED,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_K,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.trace import Span

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    model_as_dict,
    set_span_attribute,
    should_send_prompts,
)
from lmnr.sdk.utils import json_dumps

RequestType = Literal["chat", "completion", "rerank", "embedding"]

_OPERATION_NAMES: dict[RequestType, str] = {
    "chat": "chat",
    "completion": "text_completion",
    "rerank": "rerank",
    "embedding": "embeddings",
}

# v1 `chat_history` roles
_V1_ROLES = {"USER": "user", "CHATBOT": "assistant", "SYSTEM": "system", "TOOL": "tool"}


def _given(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    # the SDK's `OMIT` sentinel for unset parameters is `...`
    return {key: value for key, value in kwargs.items() if value is not ... and value is not None}


def _get(obj: Any, *path: str) -> Any:
    """Read a nested field from a pydantic model or a dict."""
    for key in path:
        if obj is None:
            return None
        obj = (
            cast(Mapping[str, Any], obj).get(key)
            if isinstance(obj, Mapping)
            else getattr(obj, key, None)
        )
    return obj


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, Any], value))
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return {}
    return model_as_dict(value)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(cast(list[Any], value))
    return []


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _text_part(text: Any) -> dict[str, Any]:
    return {"type": "text", "content": text if isinstance(text, str) else json_dumps(text)}


def _image_part(url: str) -> dict[str, Any]:
    # data URIs are kept inline as a blob, never uploaded anywhere
    if url.startswith("data:") and "," in url:
        header, data = url[len("data:") :].split(",", 1)
        return {
            "type": "blob",
            "modality": "image",
            "mime_type": header.split(";", 1)[0],
            "content": data,
        }
    return {"type": "uri", "modality": "image", "uri": url}


def _content_parts(content: Any) -> list[dict[str, Any]]:
    """Convert v2 message content (a string or a list of content items) into parts."""
    if isinstance(content, str):
        return [_text_part(content)] if content else []
    parts: list[dict[str, Any]] = []
    for item in _as_list(content):
        if isinstance(item, str):
            if item:
                parts.append(_text_part(item))
            continue
        block = _as_dict(item)
        block_type = block.get("type")
        if block_type == "text":
            parts.append(_text_part(block.get("text") or ""))
        elif block_type == "thinking":
            parts.append({"type": "thinking", "content": block.get("thinking") or ""})
        elif block_type == "image_url" and (url := _get(block, "image_url", "url")):
            parts.append(_image_part(url))
        else:
            # documents, citations, ...: keep the block whole rather than guessing
            parts.append(block)
    return parts


def _tool_result(content: Any) -> Any:
    """v2 tool message content: a string, or a list of text / document items."""
    if isinstance(content, str):
        return content
    results: list[Any] = []
    for item in _as_list(content):
        block = _as_dict(item)
        if block.get("type") == "text":
            results.append(block.get("text"))
        elif block.get("type") == "document":
            results.append(_get(block, "document", "data"))
        else:
            results.append(block or item)
    return results[0] if len(results) == 1 else results


def _v2_tool_call_parts(tool_calls: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for tool_call in _as_list(tool_calls):
        part: dict[str, Any] = {
            "type": "tool_call",
            "id": _get(tool_call, "id") or "",
            "name": _get(tool_call, "function", "name") or "",
        }
        arguments = _get(tool_call, "function", "arguments")
        if arguments is not None:
            part["arguments"] = _maybe_json(arguments)
        parts.append(part)
    return parts


def _v1_tool_call_parts(tool_calls: Any) -> list[dict[str, Any]]:
    # v1 tool calls have no id
    return [
        {
            "type": "tool_call",
            "name": _get(tool_call, "name") or "",
            "arguments": _get(tool_call, "parameters"),
        }
        for tool_call in _as_list(tool_calls)
    ]


def _v1_tool_result_parts(tool_results: Any) -> list[dict[str, Any]]:
    return [
        {
            "type": "tool_call_response",
            "name": _get(tool_result, "call", "name") or "",
            "result": _get(tool_result, "outputs"),
        }
        for tool_result in _as_list(tool_results)
    ]


def _v1_message(message: Any) -> dict[str, Any]:
    role = str(_get(message, "role") or "USER")
    parts: list[dict[str, Any]] = []
    if text := _get(message, "message"):
        parts.append(_text_part(text))
    parts.extend(_v1_tool_call_parts(_get(message, "tool_calls")))
    parts.extend(_v1_tool_result_parts(_get(message, "tool_results")))
    return {"role": _V1_ROLES.get(role.upper(), role.lower()), "parts": parts}


def _v2_message(message: Any) -> dict[str, Any]:
    role = _get(message, "role") or "user"
    if role == "tool":
        part: dict[str, Any] = {
            "type": "tool_call_response",
            "id": _get(message, "tool_call_id") or "",
            "result": _tool_result(_get(message, "content")),
        }
        return {"role": "tool", "parts": [part]}
    parts = _content_parts(_get(message, "content"))
    if tool_plan := _get(message, "tool_plan"):
        parts.append(_text_part(tool_plan))
    parts.extend(_v2_tool_call_parts(_get(message, "tool_calls")))
    return {"role": role, "parts": parts}


def _v1_chat_input(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if preamble := kwargs.get("preamble"):
        messages.append({"role": "system", "parts": [_text_part(preamble)]})
    messages.extend(_v1_message(message) for message in _as_list(kwargs.get("chat_history")))
    if tool_results := _v1_tool_result_parts(kwargs.get("tool_results")):
        messages.append({"role": "tool", "parts": tool_results})
    if message := kwargs.get("message"):
        messages.append({"role": "user", "parts": [_text_part(message)]})
    return messages


def _rerank_input(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    # documents first, then the query, as OpenLLMetry did
    messages: list[dict[str, Any]] = []
    for document in _as_list(kwargs.get("documents")):
        text = document if isinstance(document, str) else _get(document, "text")
        messages.append(
            {"role": "system", "parts": [_text_part(text if text is not None else document)]}
        )
    if query := kwargs.get("query"):
        messages.append({"role": "user", "parts": [_text_part(query)]})
    return messages


def _embed_input(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [
        _text_part(text) for text in _as_list(kwargs.get("texts")) if isinstance(text, str)
    ]
    parts.extend(
        _image_part(image) for image in _as_list(kwargs.get("images")) if isinstance(image, str)
    )
    for embed_input in _as_list(kwargs.get("inputs")):
        parts.extend(_content_parts(_get(embed_input, "content")))
    return [{"role": "user", "parts": parts}] if parts else []


def _tool_definitions(tools: Any) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for tool in _as_list(tools):
        # v2: {"type": "function", "function": {...}}; v1: {name, description, parameter_definitions}
        function = _get(tool, "function") or tool
        definition: dict[str, Any] = {"name": _get(function, "name")}
        if description := _get(function, "description"):
            definition["description"] = description
        parameters = _get(function, "parameters") or _get(function, "parameter_definitions")
        if parameters:
            definition["parameters"] = _as_dict(parameters) or parameters
        definitions.append(definition)
    return definitions


@dont_throw
def set_request_attributes(
    span: Span,
    kwargs: Mapping[str, Any],
    request_type: RequestType,
    is_v2: bool,
    is_streaming: bool,
) -> None:
    if not span.is_recording():
        return
    params = _given(kwargs)
    set_span_attribute(span, GEN_AI_OPERATION_NAME, _OPERATION_NAMES[request_type])
    set_span_attribute(span, GEN_AI_REQUEST_MODEL, params.get("model"))
    set_span_attribute(span, GEN_AI_REQUEST_MAX_TOKENS, _int(params.get("max_tokens")))
    set_span_attribute(span, GEN_AI_REQUEST_TEMPERATURE, params.get("temperature"))
    set_span_attribute(span, GEN_AI_REQUEST_TOP_P, params.get("p"))
    set_span_attribute(span, GEN_AI_REQUEST_TOP_K, params.get("k"))
    set_span_attribute(span, GEN_AI_REQUEST_FREQUENCY_PENALTY, params.get("frequency_penalty"))
    set_span_attribute(span, GEN_AI_REQUEST_PRESENCE_PENALTY, params.get("presence_penalty"))
    set_span_attribute(span, GEN_AI_REQUEST_SEED, _int(params.get("seed")))
    stop_sequences = [s for s in _as_list(params.get("stop_sequences")) if isinstance(s, str)]
    if stop_sequences:
        set_span_attribute(span, GEN_AI_REQUEST_STOP_SEQUENCES, stop_sequences)
    set_span_attribute(span, "llm.is_streaming", is_streaming)

    if not should_send_prompts():
        return

    if request_type == "chat":
        messages = (
            [_v2_message(message) for message in _as_list(params.get("messages"))]
            if is_v2
            else _v1_chat_input(params)
        )
        if tools := _tool_definitions(params.get("tools")):
            set_span_attribute(span, "gen_ai.tool.definitions", json_dumps(tools))
    elif request_type == "completion":
        prompt = params.get("prompt")
        messages = [{"role": "user", "parts": [_text_part(prompt)]}] if prompt else []
    elif request_type == "rerank":
        messages = _rerank_input(params)
    else:
        messages = _embed_input(params)

    if messages:
        set_span_attribute(span, "gen_ai.input.messages", json_dumps(messages))


def _set_usage(span: Span, usage: Any) -> None:
    """`usage` (v2) / `meta` (v1): actual `tokens`, falling back to `billed_units`."""
    tokens = _get(usage, "tokens")
    billed_units = _get(usage, "billed_units")
    input_tokens = _int(_get(tokens, "input_tokens"))
    if input_tokens is None:
        input_tokens = _int(_get(billed_units, "input_tokens"))
    output_tokens = _int(_get(tokens, "output_tokens"))
    if output_tokens is None:
        output_tokens = _int(_get(billed_units, "output_tokens"))
    set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
    if input_tokens is not None or output_tokens is not None:
        set_span_attribute(
            span, "llm.usage.total_tokens", (input_tokens or 0) + (output_tokens or 0)
        )
    set_span_attribute(
        span, "gen_ai.usage.cache_read_input_tokens", _int(_get(usage, "cached_tokens"))
    )


def _v1_chat_output(response: Any) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if text := _get(response, "text"):
        parts.append(_text_part(text))
    parts.extend(_v1_tool_call_parts(_get(response, "tool_calls")))
    return {"role": "assistant", "parts": parts}


def _v2_chat_output(response: Any) -> dict[str, Any]:
    output = _v2_message(_get(response, "message") or {})
    output["role"] = "assistant"
    return output


def _completion_output(response: Any) -> list[dict[str, Any]]:
    return [
        {"role": "assistant", "parts": [_text_part(_get(generation, "text") or "")]}
        for generation in _as_list(_get(response, "generations"))
    ]


def _rerank_output(response: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for result in _as_list(_get(response, "results")):
        text = f"Doc {_get(result, 'index')}, Score: {_get(result, 'relevance_score')}"
        if document_text := _get(result, "document", "text"):
            text += f"\n{document_text}"
        parts.append(_text_part(text))
    return [{"role": "assistant", "parts": parts}] if parts else []


@dont_throw
def set_response_attributes(
    span: Span, response: Any, request_type: RequestType, is_v2: bool
) -> None:
    if not span.is_recording():
        return
    # v1 chat returns both `generation_id` and `response_id`; the rest carry `id`
    set_span_attribute(
        span, GEN_AI_RESPONSE_ID, _get(response, "response_id") or _get(response, "id")
    )
    _set_usage(span, _get(response, "usage") or _get(response, "meta"))

    output: list[dict[str, Any]] = []
    if request_type == "chat":
        message = _v2_chat_output(response) if is_v2 else _v1_chat_output(response)
        if finish_reason := _get(response, "finish_reason"):
            finish_reason = str(finish_reason)
            message["finish_reason"] = finish_reason
            set_span_attribute(span, GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])
        output = [message]
    elif request_type == "completion":
        output = _completion_output(response)
    elif request_type == "rerank":
        output = _rerank_output(response)
    # embeddings have no message output: the vectors are not worth the payload

    if output and should_send_prompts():
        set_span_attribute(span, "gen_ai.output.messages", json_dumps(output))


class V1ChatStreamAccumulator:
    """`BaseCohere.chat_stream`: the `stream-end` event carries the full response."""

    def __init__(self) -> None:
        self.response: Any = None
        self.text: str = ""
        self.tool_calls: list[Any] = []

    def add(self, event: Any) -> None:
        event_type = _get(event, "event_type")
        if event_type == "text-generation":
            self.text += _get(event, "text") or ""
        elif event_type == "tool-calls-generation":
            self.tool_calls.extend(_as_list(_get(event, "tool_calls")))
        elif event_type == "stream-end":
            self.response = _get(event, "response")

    def set_attributes(self, span: Span) -> None:
        # a stream abandoned before `stream-end` still records what it received
        response = self.response or {"text": self.text, "tool_calls": self.tool_calls}
        set_response_attributes(span, response, "chat", is_v2=False)


class V2ChatStreamAccumulator:
    """`V2Client.chat_stream`: rebuilds a `V2ChatResponse`-shaped dict from the events."""

    def __init__(self) -> None:
        self.id: str | None = None
        self.content: dict[int, dict[str, Any]] = {}
        self.tool_plan: str = ""
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage: Any = None

    @staticmethod
    def _index(event: Any, items: dict[int, dict[str, Any]], new: bool) -> int:
        index = _get(event, "index")
        if isinstance(index, int):
            return index
        return len(items) if new or not items else max(items)

    def add(self, event: Any) -> None:
        event_type = _get(event, "type")
        delta_message = _get(event, "delta", "message")
        if event_type == "message-start":
            self.id = _get(event, "id") or self.id
        elif event_type == "content-start":
            content = _get(delta_message, "content")
            self.content[self._index(event, self.content, new=True)] = {
                "type": _get(content, "type") or "text",
                "text": _get(content, "text") or "",
                "thinking": _get(content, "thinking") or "",
            }
        elif event_type == "content-delta":
            item = self.content.setdefault(
                self._index(event, self.content, new=False),
                {"type": "text", "text": "", "thinking": ""},
            )
            item["text"] += _get(delta_message, "content", "text") or ""
            item["thinking"] += _get(delta_message, "content", "thinking") or ""
        elif event_type == "tool-plan-delta":
            self.tool_plan += _get(delta_message, "tool_plan") or ""
        elif event_type == "tool-call-start":
            tool_call = _get(delta_message, "tool_calls")
            self.tool_calls[self._index(event, self.tool_calls, new=True)] = {
                "id": _get(tool_call, "id") or "",
                "function": {
                    "name": _get(tool_call, "function", "name") or "",
                    "arguments": _get(tool_call, "function", "arguments") or "",
                },
            }
        elif event_type == "tool-call-delta":
            tool_call = self.tool_calls.setdefault(
                self._index(event, self.tool_calls, new=False),
                {"id": "", "function": {"name": "", "arguments": ""}},
            )
            tool_call["function"]["arguments"] += (
                _get(delta_message, "tool_calls", "function", "arguments") or ""
            )
        elif event_type == "message-end":
            self.id = _get(event, "id") or self.id
            self.finish_reason = _get(event, "delta", "finish_reason")
            self.usage = _get(event, "delta", "usage")

    def set_attributes(self, span: Span) -> None:
        content = [
            {"type": "thinking", "thinking": item["thinking"]}
            if item["type"] == "thinking"
            else {"type": "text", "text": item["text"]}
            for _, item in sorted(self.content.items())
        ]
        response = {
            "id": self.id,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "message": {
                "role": "assistant",
                "content": content,
                "tool_plan": self.tool_plan or None,
                "tool_calls": [tool_call for _, tool_call in sorted(self.tool_calls.items())],
            },
        }
        set_response_attributes(span, response, "chat", is_v2=True)

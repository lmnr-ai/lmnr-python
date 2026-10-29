"""Convert provider-native LLM messages and tools into lmnr-atif shapes.

Instrumentations record `gen_ai.input.messages` / `gen_ai.output.messages` /
`gen_ai.tool.definitions` in each provider's own format. This module turns
them into `LmnrMessage`s (spec §3) and ATIF tool definitions, so the builder
can compare conversations across calls and write ATIF steps.

Shapes understood, detected per message so mixed lists work:

- OpenAI Chat Completions (also LiteLLM, Groq, OpenRouter, LangChain-OpenAI):
  `{role, content, tool_calls: [{id, function: {name, arguments}}]}`, role
  `tool` results; outputs as `choices` (`{message, finish_reason}`) or bare
  messages.
- Anthropic: content blocks (`text`, `image`, `tool_use`, `tool_result`,
  `thinking`); `tool_result` blocks are split out of the user message into
  `tool` messages. Outputs carry `stop_reason`.
- OpenAI Responses: items (`message`, `function_call`, `function_call_output`,
  `reasoning`), with `input_text` / `output_text` parts.
- Gemini (google-genai, snake_case or camelCase): `{role, parts}` with `text`,
  `thought`, `function_call`, `function_response`, `inline_data`,
  `file_data`. Gemini calls often have no id, so the k-th call without one in
  a conversation gets the id `<name>_<k>` (marked `synthetic_id`), and its
  response is matched by name in order.

Anything else degrades to role + text. Nothing here raises on bad input.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

from lmnr.sdk.trajectory.reader import as_dict, dicts

Message = dict[str, Any]

# Stores media and returns the path to reference it by (relative to the
# trajectory file), or None to fall back to a text placeholder.
# Arguments: base64 data, MIME type.
MediaSink = Callable[[str, str], str | None]

ATIF_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Request-shaping keys the model never sees (as in app-server's tool normalizer).
_TRANSPORT_KEYS = {"cache_control", "strict", "providerOptions", "provider_options"}
_SCHEMA_KEYS = (
    "parameters",
    "input_schema",
    "inputSchema",
    "parameters_json_schema",
    "parametersJsonSchema",
)

_ROLE_ALIASES = {"model": "assistant", "human": "user", "ai": "assistant", "function": "tool"}


def _get(obj: dict[str, Any], *keys: str) -> Any:
    """First present key, for snake_case / camelCase variants."""
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def simplify_content(parts: list[dict[str, Any]]) -> Any:
    """A single text part becomes a plain string; no parts become ""."""
    if not parts:
        return ""
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0].get("text") or ""
    return parts


def content_text(content: Any) -> str:
    """Text of a message content, with media as placeholders."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for part in dicts(content):
        if part.get("type") == "text":
            texts.append(part.get("text") or "")
        else:
            texts.append(f"[{part.get('type')}: {as_dict(part.get('source')).get('path')}]")
    return "\n".join(texts)


def _media_part(media_type: Any, data: Any, url: Any, media: MediaSink | None) -> dict[str, Any]:
    media_type = media_type if isinstance(media_type, str) else ""
    if isinstance(url, str) and url.startswith("data:"):
        header, _, payload = url.partition(",")
        media_type = header[5:].split(";", 1)[0] or media_type
        data, url = payload, None
    kind = "image" if media_type.startswith("image/") or not media_type else media_type
    if media_type in ATIF_IMAGE_TYPES:
        path = None
        if isinstance(url, str) and url:
            path = url
        elif isinstance(data, str) and data and media is not None:
            path = media(data, media_type)
        if path:
            return {"type": "image", "source": {"media_type": media_type, "path": path}}
    if isinstance(url, str) and url and not media_type:
        return text_part(f"[image: {url}]")
    return text_part(f"[{kind}]" if kind != "image" else f"[image: {media_type or 'unknown'}]")


def parse_arguments(value: Any) -> tuple[dict[str, Any], str | None]:
    """Tool call arguments as ATIF wants them (an object), plus the raw string
    when they weren't a JSON object."""
    if isinstance(value, dict):
        return as_dict(value), None
    if value is None or value == "":
        return {}, None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}, value
        if isinstance(parsed, dict):
            return as_dict(parsed), None
        return {}, value
    return {}, json.dumps(value, ensure_ascii=False, default=str)


def tool_call(
    call_id: Any, name: Any, arguments: Any, synthetic_id: bool = False
) -> dict[str, Any]:
    parsed, raw = parse_arguments(arguments)
    result: dict[str, Any] = {
        "tool_call_id": str(call_id) if call_id else "",
        "function_name": str(name) if name else "",
        "arguments": parsed,
    }
    lmnr: dict[str, Any] = {}
    if raw is not None:
        lmnr["raw_arguments"] = raw
    if synthetic_id:
        lmnr["synthetic_id"] = True
    if lmnr:
        result["extra"] = {"lmnr": lmnr}
    return result


def _result_content(value: Any, media: MediaSink | None) -> Any:
    """Content of a tool result: a string, or parts if it has media."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return simplify_content(_parts(cast(list[Any], value), media, {}))
    return json.dumps(value, ensure_ascii=False, default=str)


def _parts(
    blocks: list[Any], media: MediaSink | None, sink: dict[str, list[Any]]
) -> list[dict[str, Any]]:
    """Content parts of a block list; tool calls, results and reasoning go to `sink`."""
    parts: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(text_part(block))
            continue
        if not isinstance(block, dict):
            continue
        block = as_dict(block)
        kind = block.get("type")
        if kind in ("text", "input_text", "output_text") or (kind is None and "text" in block):
            if block.get("thought") is True:
                sink.setdefault("reasoning", []).append(str(block.get("text") or ""))
            else:
                parts.append(text_part(str(block.get("text") or "")))
        elif kind in ("thinking", "reasoning"):
            text = block.get("thinking") or block.get("text") or ""
            if not text and isinstance(block.get("summary"), list):
                text = "\n".join(str(s.get("text") or "") for s in dicts(block["summary"]))
            if text:
                sink.setdefault("reasoning", []).append(str(text))
        elif kind == "redacted_thinking":
            continue
        elif kind == "tool_use":
            sink.setdefault("tool_calls", []).append(
                tool_call(block.get("id"), block.get("name"), block.get("input"))
            )
        elif kind == "tool_result":
            sink.setdefault("tool_results", []).append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id") or ""),
                    "content": _result_content(block.get("content"), media),
                    **({"_is_error": True} if block.get("is_error") is True else {}),
                }
            )
        elif kind in ("image", "input_image", "image_url"):
            source = as_dict(block.get("source"))
            image_url = block.get("image_url")
            url = as_dict(image_url).get("url") if isinstance(image_url, dict) else image_url
            parts.append(
                _media_part(
                    source.get("media_type"),
                    source.get("data"),
                    url or source.get("url"),
                    media,
                )
            )
        elif kind in ("document", "file", "input_file"):
            parts.append(
                text_part(f"[file: {block.get('filename') or block.get('file_id') or ''}]")
            )
        elif kind in ("input_audio", "audio"):
            parts.append(text_part("[audio]"))
        else:
            parts.append(text_part(json.dumps(block, ensure_ascii=False, default=str)))
    return parts


def _gemini_parts(
    parts_raw: list[Any], media: MediaSink | None, sink: dict[str, list[Any]]
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for part in dicts(parts_raw):
        call = as_dict(_get(part, "function_call", "functionCall"))
        response = as_dict(_get(part, "function_response", "functionResponse"))
        inline = as_dict(_get(part, "inline_data", "inlineData"))
        file_data = as_dict(_get(part, "file_data", "fileData"))
        if call:
            sink.setdefault("gemini_calls", []).append(call)
        elif response:
            sink.setdefault("gemini_responses", []).append(response)
        elif inline:
            parts.append(
                _media_part(_get(inline, "mime_type", "mimeType"), inline.get("data"), None, media)
            )
        elif file_data:
            parts.append(
                _media_part(
                    _get(file_data, "mime_type", "mimeType"),
                    None,
                    _get(file_data, "file_uri", "fileUri"),
                    media,
                )
            )
        elif "text" in part:
            if part.get("thought") is True:
                sink.setdefault("reasoning", []).append(str(part.get("text") or ""))
            else:
                parts.append(text_part(str(part.get("text") or "")))
    return parts


class _Conversation:
    """Normalizes the messages of one LLM call, input then output.

    Keeps the per-conversation state Gemini needs: generated ids for calls
    without one, and the queue matching responses to them by name.
    """

    def __init__(self, media: MediaSink | None):
        self.media: MediaSink | None = media
        self.messages: list[Message] = []
        self._gemini_count: int = 0
        self._gemini_pending: dict[str, list[str]] = {}

    def _gemini_call(self, call: dict[str, Any]) -> dict[str, Any]:
        name = str(call.get("name") or "")
        call_id = call.get("id")
        synthetic = not call_id
        if synthetic:
            self._gemini_count += 1
            call_id = f"{name}_{self._gemini_count}"
        self._gemini_pending.setdefault(name, []).append(str(call_id))
        return tool_call(call_id, name, call.get("args"), synthetic_id=synthetic)

    def _gemini_response(self, response: dict[str, Any]) -> Message:
        name = str(response.get("name") or "")
        call_id = response.get("id")
        pending = self._gemini_pending.get(name) or []
        if call_id and str(call_id) in pending:
            pending.remove(str(call_id))
        elif not call_id and pending:
            call_id = pending.pop(0)
        value: Any = response.get("response")
        wrapper = as_dict(value)
        if wrapper and set(wrapper) <= {"output", "result"}:
            inner = _get(wrapper, "output", "result")
            value = inner if inner is not None else value
        return {
            "role": "tool",
            "tool_call_id": str(call_id or ""),
            "content": _result_content(value, self.media),
        }

    def _append_assistant(self, message: Message) -> None:
        """Merge into the trailing assistant message (Responses items split one
        turn into several), or start a new one."""
        last = self.messages[-1] if self.messages else None
        if last is not None and last.get("role") == "assistant" and not last.get("_closed"):
            if message.get("reasoning_content"):
                joined = "\n".join(
                    t for t in (last.get("reasoning_content"), message["reasoning_content"]) if t
                )
                last["reasoning_content"] = joined
            if message.get("content") not in (None, ""):
                if last.get("content") in (None, ""):
                    last["content"] = message["content"]
                else:
                    last["content"] = simplify_content(
                        _as_parts(last["content"]) + _as_parts(message["content"])
                    )
            if message.get("tool_calls"):
                last.setdefault("tool_calls", []).extend(message["tool_calls"])
            return
        self.messages.append(message)

    def close_assistant(self) -> None:
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages[-1]["_closed"] = True

    def add(self, raw: Any, merge_assistant: bool = False) -> None:
        """Normalize one raw message / item and append the result."""
        if isinstance(raw, str):
            self.close_assistant()
            self.messages.append({"role": "user", "content": raw})
            return
        item = as_dict(raw)
        if not item:
            return
        kind = item.get("type")

        # OpenAI Responses items.
        if kind == "function_call":
            self._append_assistant(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        tool_call(
                            item.get("call_id") or item.get("id"),
                            item.get("name"),
                            item.get("arguments"),
                        )
                    ],
                }
            )
            return
        if kind == "function_call_output":
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or ""),
                    "content": _result_content(item.get("output"), self.media),
                }
            )
            return
        if kind == "reasoning":
            sink: dict[str, list[Any]] = {}
            _ = _parts([item], self.media, sink)
            reasoning = "\n".join(str(r) for r in sink.get("reasoning", []))
            self._append_assistant(
                {"role": "assistant", "content": "", "reasoning_content": reasoning}
            )
            return

        role = item.get("role")
        role = _ROLE_ALIASES.get(str(role), role) if role is not None else None
        if role is None and kind not in (None, "message"):
            # Built-in tool calls (web_search_call, ...) and other items.
            return
        role = role or "user"

        sink = {}
        if "parts" in item:
            parts = _gemini_parts(item.get("parts") or [], self.media, sink)
        else:
            content = item.get("content")
            if isinstance(content, list):
                parts = _parts(cast(list[Any], content), self.media, sink)
            elif content is None:
                parts = []
            else:
                parts = [text_part(content if isinstance(content, str) else json.dumps(content))]

        tool_calls: list[dict[str, Any]] = list(sink.get("tool_calls", []))
        for call in dicts(item.get("tool_calls")):
            function = as_dict(call.get("function"))
            tool_calls.append(
                tool_call(
                    call.get("id"),
                    function.get("name") or call.get("name"),
                    function.get("arguments"),
                )
            )
        for call in sink.get("gemini_calls", []):
            tool_calls.append(self._gemini_call(call))
        reasoning = [*sink.get("reasoning", [])]
        if isinstance(item.get("reasoning_content"), str) and item["reasoning_content"]:
            reasoning.append(item["reasoning_content"])

        if role == "tool":
            self.close_assistant()
            message: Message = {
                "role": "tool",
                "tool_call_id": str(item.get("tool_call_id") or item.get("name") or ""),
                "content": simplify_content(parts),
            }
            self.messages.append(message)
            return

        results = [self._gemini_response(r) for r in sink.get("gemini_responses", [])]
        results += sink.get("tool_results", [])
        if role != "assistant":
            self.close_assistant()
            # Tool results come first: they answer the previous assistant turn.
            self.messages.extend(results)
            if parts or not results:
                self.messages.append({"role": role, "content": simplify_content(parts)})
            return

        message = {"role": "assistant", "content": simplify_content(parts)}
        if reasoning:
            message["reasoning_content"] = "\n".join(reasoning)
        if tool_calls:
            message["tool_calls"] = tool_calls
        if merge_assistant:
            self._append_assistant(message)
        else:
            self.close_assistant()
            self.messages.append(message)
        self.messages.extend(results)

    def finish(self) -> list[Message]:
        for message in self.messages:
            message.pop("_closed", None)
        return self.messages


def _as_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [text_part(content)] if content else []
    return dicts(content)


def normalize_call(
    input_raw: Any, output_raw: Any, media: MediaSink | None = None
) -> tuple[list[Message], Message | None, str | None]:
    """Normalize one LLM call: `(input messages, output message, finish reason)`.

    The output is one assistant message (None when the call produced no
    output, e.g. it failed). Input and output are normalized together because
    Gemini's generated call ids continue from the input into the output.
    """
    conversation = _Conversation(media)
    if isinstance(input_raw, str):
        input_raw = [{"role": "user", "content": input_raw}]
    for raw in cast(list[Any], input_raw) if isinstance(input_raw, list) else []:
        conversation.add(raw, merge_assistant=as_dict(raw).get("type") is not None)
    # Copies: the output is appended to the same conversation below. `finish`
    # drops the closed marks, so close again: an input that ends on an
    # assistant turn (an Anthropic prefill) must not absorb the output.
    input_messages = [dict(m) for m in conversation.finish()]
    conversation.close_assistant()
    count = len(input_messages)

    finish_reason: str | None = None
    if isinstance(output_raw, str):
        output_raw = [{"role": "assistant", "content": output_raw}]
    elif isinstance(output_raw, dict):
        output_raw = [output_raw]
    output_items: list[Any] = []
    for raw in cast(list[Any], output_raw) if isinstance(output_raw, list) else []:
        item = as_dict(raw)
        if not item:
            continue
        # OpenAI choices and Gemini candidates wrap the message.
        if "message" in item or ("content" in item and isinstance(item.get("content"), dict)):
            reason = _get(item, "finish_reason", "finishReason")
            finish_reason = finish_reason or (str(reason) if reason else None)
            wrapped = as_dict(item.get("message") or item.get("content"))
            output_items.append({**wrapped, "role": wrapped.get("role") or "assistant"})
            if "message" in item:
                break  # first choice only
            continue
        reason = _get(item, "stop_reason", "finish_reason")
        finish_reason = finish_reason or (str(reason) if reason else None)
        output_items.append(
            item if item.get("type") or item.get("role") else {**item, "role": "assistant"}
        )
    for item in output_items:
        if item.get("type") is None and item.get("role") not in (None, "assistant", "model"):
            continue
        conversation.add(
            {**item, "role": "assistant"} if item.get("type") is None else item,
            merge_assistant=True,
        )
    messages = conversation.finish()
    produced = [m for m in messages[count:] if m.get("role") == "assistant"]
    if not produced:
        return input_messages, None, finish_reason
    output: Message = {"role": "assistant", "content": ""}
    for message in produced:
        if message.get("content") not in (None, ""):
            output["content"] = (
                message["content"]
                if output["content"] == ""
                else simplify_content(_as_parts(output["content"]) + _as_parts(message["content"]))
            )
        if message.get("reasoning_content"):
            output["reasoning_content"] = "\n".join(
                t for t in (output.get("reasoning_content"), message["reasoning_content"]) if t
            )
        if message.get("tool_calls"):
            output.setdefault("tool_calls", []).extend(message["tool_calls"])
    return input_messages, output, finish_reason


def normalize_tools(raw: Any) -> list[dict[str, Any]]:
    """Tool definitions in ATIF's OpenAI function shape, minus transport keys."""
    tools: list[dict[str, Any]] = []
    for tool in dicts(raw):
        function = as_dict(tool.get("function"))
        if function:
            fn = {k: v for k, v in function.items() if k not in _TRANSPORT_KEYS}
            tools.append({"type": "function", "function": fn})
            continue
        declarations = _get(tool, "function_declarations", "functionDeclarations")
        if isinstance(declarations, list):
            tools.extend(normalize_tools(declarations))
            continue
        kind = tool.get("type")
        schema = next((tool[k] for k in _SCHEMA_KEYS if k in tool), None)
        if tool.get("name") and (kind in (None, "function", "custom") or schema is not None):
            fn: dict[str, Any] = {"name": tool["name"]}
            if tool.get("description"):
                fn["description"] = tool["description"]
            if schema is not None:
                fn["parameters"] = schema
            tools.append({"type": "function", "function": fn})
            continue
        if kind:
            tools.append({k: v for k, v in tool.items() if k not in _TRANSPORT_KEYS})
    return tools

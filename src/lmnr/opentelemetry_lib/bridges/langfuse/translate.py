"""Pure functions that convert Langfuse / OpenInference span shapes into
Laminar / OTel GenAI shapes.

No `SpanProcessor` state lives here; `processor.py` calls into these.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypeAlias, cast

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_REQUEST_CHOICE_COUNT,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_SEED,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_K,
    GEN_AI_REQUEST_TOP_P,
)
from opentelemetry.util.types import AttributeValue

from lmnr.opentelemetry_lib.bridges.langfuse.attributes import (
    GEN_AI_REQUEST_REASONING_LEVEL,
    GEN_AI_REQUEST_REASONING_SUMMARY,
    GEN_AI_REQUEST_SERVICE_TIER,
    LANGFUSE_TRACER_NAME,
    OI_LLM_INPUT_MESSAGES,
    OI_LLM_OUTPUT_MESSAGES,
    OI_SPAN_KIND,
)

# A JSON-like value reassembled from openinference's flat, dotted/indexed span
# attributes (see `oi_collect_indexed` / `_oi_assign_path`), or produced by
# reshaping one of those into a GenAI message/tool dict (`oi_message_to_genai`
# / `oi_tool_to_genai`). Deliberately narrower than `AttributeValue`: OTel
# attributes CAN be homogeneous sequences (`Sequence[str]`, ...), but
# openinference never emits one as a leaf here — a real list is always
# expressed as its own family of indexed dotted keys, reassembled into
# `list["OIValue"]` by `_oi_assign_path`. Keeping those sequence variants out
# of the union is what lets `isinstance(cur, list)` narrow to a single
# concrete list type below instead of a union of `list[str] | list[bool] |
# ...`. Containers are plain `dict`/`list` nested to arbitrary depth; `None`
# covers fields looked up with `dict.get` on a reshaped dict.
OIValue: TypeAlias = str | bool | int | float | None | dict[str, "OIValue"] | list["OIValue"]


def parse_json(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


# Responses API request keys `langfuse.openai` records next to `input`.
_RESPONSES_PROMPT_KEYS = {"input", "instructions", "tools", "tool_choice", "parallel_tool_calls"}
_TEXT_PART_TYPES = {"text", "input_text", "output_text"}
# Responses API items that hold an assistant tool call, and their results.
_RESPONSES_CALL_TYPES = {"function_call", "custom_tool_call"}
_RESPONSES_CALL_OUTPUT_TYPES = {"function_call_output", "custom_tool_call_output"}


def genai_input_from_langfuse(input_raw: Any) -> tuple[list[Any], Any] | None:
    """Split a Langfuse LLM input into (messages, tool_definitions).

    Shapes `langfuse.openai` records (`_extract_chat_prompt`,
    `_extract_responses_prompt`):

    - chat: a bare message list, or `{messages, tools|functions, ...}` when the
      call passed tools;
    - Responses API: the `input` (string or item list, with `instructions`
      already prepended as a system message), or `{input, tools, tool_choice,
      ...}` when the call passed tools.

    A bare string (legacy completions, manual generations) is a user message.
    Returns None for any other shape; the caller keeps the raw input.
    """
    parsed = parse_json(input_raw)
    if isinstance(parsed, str):
        return ([{"role": "user", "content": parsed}], None) if parsed else None
    if isinstance(parsed, list):
        return cast(list[Any], parsed), None
    if not isinstance(parsed, dict):
        return None
    prompt = cast(dict[str, Any], parsed)
    tools = prompt.get("tools")
    if tools is None:
        tools = prompt.get("functions")
    if isinstance(prompt.get("messages"), list):
        return prompt["messages"], tools
    if "input" in prompt and set(prompt) <= _RESPONSES_PROMPT_KEYS:
        inp = prompt["input"]
        if isinstance(inp, str):
            messages: list[Any] = [{"role": "user", "content": inp}]
        elif isinstance(inp, list):
            messages = list(cast(list[Any], inp))
        else:
            return None
        instructions = prompt.get("instructions")
        if isinstance(instructions, str) and instructions:
            messages.insert(0, {"role": "system", "content": instructions})
        return messages, tools
    return None


def split_messages_and_tool_defs_langchain(
    messages: Any,
) -> tuple[Any, list[Any] | None]:
    """Splits langfuse.langchain input into the actual messages and tool definitions

    The callback inlines tool definitions into the message array as messages with
    role "tool" and a "content" dict. Two provider-specific shapes occur:

    * OpenAI: ``{"type": "function", "function": {...}}`` — the actual definition
      lives under the ``"function"`` key.
    * Anthropic: ``{"name", "input_schema", "description"}`` — the content dict IS
      the definition (already the Anthropic-native shape Laminar's own Anthropic
      instrumentor emits into ``gen_ai.tool.definitions``).

    Both are pulled out into the tool-definitions list; everything else stays a
    message.
    """
    if not isinstance(messages, list):
        return (messages, None)
    new_msgs = []
    tool_defs = []
    for msg in messages:  # pyright: ignore[reportUnknownVariableType]
        if isinstance(msg, dict) and msg.get("role") == "tool":  # pyright: ignore[reportUnknownMemberType]
            content = msg.get("content")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            if isinstance(content, dict):
                if content.get("type") == "function" and isinstance(  # pyright: ignore[reportUnknownMemberType]
                    content.get("function"),
                    dict,  # pyright: ignore[reportUnknownMemberType]
                ):
                    tool_defs.append(content["function"])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    continue
                if "input_schema" in content and "name" in content:
                    tool_defs.append(content)  # pyright: ignore[reportUnknownMemberType]
                    continue
        new_msgs.append(msg)  # pyright: ignore[reportUnknownMemberType]
    return (new_msgs, tool_defs)  # pyright: ignore[reportUnknownVariableType]


def genai_output_from_langfuse(output_raw: Any) -> list[Any] | None:
    """A Langfuse LLM output as a list, before `normalize_messages`.

    `langfuse.openai` records one message dict (chat), a bare string (streamed
    chat text), or the Responses API output items (a list, or the only item
    as a dict). Returns None for any other shape; the caller keeps the raw
    output.
    """
    parsed = parse_json(output_raw)
    if isinstance(parsed, list):
        return cast(list[Any], parsed)
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, str) and parsed:
        return [{"role": "assistant", "content": parsed}]
    return None


def _tool_call_part(call_id: Any, name: Any, arguments: Any) -> dict[str, Any]:
    return {"type": "tool_call", "id": call_id, "name": name, "arguments": arguments}


def _reasoning_text(item: dict[str, Any]) -> str:
    """Summary (else raw reasoning) text of a Responses API reasoning item."""
    for key in ("summary", "content"):
        blocks = item.get(key)
        if isinstance(blocks, list):
            texts = [
                cast(dict[str, Any], b)["text"]
                for b in cast(list[Any], blocks)
                if isinstance(b, dict) and isinstance(cast(dict[str, Any], b).get("text"), str)
            ]
            if texts:
                return "\n\n".join(texts)
    return ""


def _normalize_part(part: Any) -> list[Any]:
    """One content part in Laminar's shape (`text`, `reasoning`, `tool_call`);
    an empty list drops it. Unknown parts are kept as they are."""
    if isinstance(part, str):
        return [{"type": "text", "text": part}]
    if not isinstance(part, dict):
        return [part]
    p = cast(dict[str, Any], part)
    kind = p.get("type")
    if kind in _TEXT_PART_TYPES and isinstance(p.get("text"), str):
        return [{"type": "text", "text": p["text"]}]
    if kind == "tool_call":
        # LangChain names the arguments `args`.
        args = p["arguments"] if "arguments" in p else p.get("args")
        return [_tool_call_part(p.get("id"), p.get("name"), args)]
    if kind == "tool_use":
        return [_tool_call_part(p.get("id"), p.get("name"), p.get("input"))]
    if kind in _RESPONSES_CALL_TYPES:
        args = p["arguments"] if "arguments" in p else p.get("input")
        return [_tool_call_part(p.get("call_id") or p.get("id"), p.get("name"), args)]
    if kind in ("reasoning", "thinking"):
        text = p.get("text") if kind == "reasoning" else p.get("thinking")
        if not isinstance(text, str):
            text = _reasoning_text(p)
        return [{"type": "reasoning", "text": text}] if text else []
    if kind == "redacted_thinking":
        return []
    # openinference message contents: `{message_content: {type, text}}`, and
    # `{message_content: {type: "tool_use"}, tool_call: {...}}` for Anthropic.
    content = p.get("message_content")
    if kind is None and isinstance(content, dict):
        tc = p.get("tool_call")
        if isinstance(tc, dict):
            tool_call = cast(dict[str, Any], tc)
            fn = tool_call.get("function")
            fn = cast(dict[str, Any], fn) if isinstance(fn, dict) else {}
            return [_tool_call_part(tool_call.get("id"), fn.get("name"), fn.get("arguments"))]
        if cast(dict[str, Any], content).get("type") == "tool_use":
            return []
        return _normalize_part(content)
    return [p]


def _dedupe_tool_calls(parts: list[Any]) -> list[Any]:
    """Drop repeated tool calls (same id) within one message: LangChain keeps a
    call both in the content blocks (Anthropic `tool_use`, Responses
    `function_call`) and in `tool_calls`."""
    seen: set[str] = set()
    out: list[Any] = []
    for part in parts:
        if isinstance(part, dict) and cast(dict[str, Any], part).get("type") == "tool_call":
            call_id = cast(dict[str, Any], part).get("id")
            if isinstance(call_id, str) and call_id:
                if call_id in seen:
                    continue
                seen.add(call_id)
        out.append(part)
    return out


def _collapse(parts: list[Any]) -> Any:
    """Content parts as message content. A single text part becomes a string,
    so `"hi"` and `[{"type": "text", "text": "hi"}]` hash the same."""
    if len(parts) == 1 and isinstance(parts[0], dict):
        only = cast(dict[str, Any], parts[0])
        if only.get("type") == "text" and set(only) == {"type", "text"}:
            return only["text"]
    return parts


def _normalize_content(content: Any) -> Any:
    if isinstance(content, list):
        return _collapse(
            _dedupe_tool_calls([p for c in cast(list[Any], content) for p in _normalize_part(c)])
        )
    return content


def _normalize_message(msg: dict[str, Any], is_output: bool) -> list[dict[str, Any]]:
    """A chat message as `{role, content, tool_call_id?, name?}`; provider
    extras (`refusal`, `annotations`, LangChain's `additional_kwargs`, ...)
    are dropped so a turn's output equals its copy in the next call's history.
    Anthropic `tool_result` parts become `tool` messages."""
    role = msg.get("role")
    if role == "model" or (role is None and is_output):
        role = "assistant"
    content = msg.get("content")
    parts: list[Any] | None = None
    if isinstance(content, list):
        parts = [p for c in cast(list[Any], content) for p in _normalize_part(c)]

    calls: list[Any] = []
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in cast(list[Any], tool_calls):
            fn = cast(dict[str, Any], tc).get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict):
                fn = cast(dict[str, Any], fn)
                calls.append(
                    _tool_call_part(
                        cast(dict[str, Any], tc).get("id"), fn.get("name"), fn.get("arguments")
                    )
                )
            elif isinstance(tc, dict) and "type" not in tc and "name" in tc:
                # LangChain `{id, name, args}`.
                calls.extend(_normalize_part({**cast(dict[str, Any], tc), "type": "tool_call"}))
            else:
                calls.extend(_normalize_part(tc))
    function_call = msg.get("function_call")
    if isinstance(function_call, dict):
        fc = cast(dict[str, Any], function_call)
        calls.append(_tool_call_part(None, fc.get("name"), fc.get("arguments")))
    if calls:
        if parts is None:
            parts = (
                [{"type": "text", "text": content}] if isinstance(content, str) and content else []
            )
        parts.extend(calls)

    results: list[dict[str, Any]] = []
    if parts is not None and role == "user":
        rest: list[Any] = []
        for part in parts:
            if isinstance(part, dict) and cast(dict[str, Any], part).get("type") == "tool_result":
                p = cast(dict[str, Any], part)
                results.append(
                    {
                        "role": "tool",
                        "content": _normalize_content(p.get("content")),
                        "tool_call_id": p.get("tool_use_id"),
                    }
                )
            else:
                rest.append(part)
        parts = rest
        if results and not parts:
            return results

    out: dict[str, Any] = {
        "role": role,
        "content": _collapse(_dedupe_tool_calls(parts)) if parts is not None else content,
    }
    for key in ("tool_call_id", "name"):
        if msg.get(key) is not None:
            out[key] = msg[key]
    return [*results, out]


def normalize_messages(messages: list[Any], is_output: bool) -> list[Any]:
    """Bring LLM input / output messages to Laminar's shape:
    `{role, content}`, where content is a string or a list of `text`,
    `reasoning` and `tool_call` (`{id, name, arguments}`) parts, and a tool
    result is `{role: "tool", content, tool_call_id}`.

    Covers OpenAI chat (`tool_calls`), Responses API items (`function_call`,
    `function_call_output`, `reasoning`, `message`), LangChain (`args`),
    Anthropic (`tool_use`, `tool_result`, `thinking`), Gemini's `model` role
    and openinference's `message_content` wrappers. Inputs and outputs go
    through the same rules, so an assistant turn is identical in the call that
    produced it and in the next call's history (app-server dedups messages by
    hash).

    Consecutive Responses API items of one assistant turn (reasoning, calls,
    the assistant message) are merged into one assistant message.
    """
    out: list[Any] = []
    turn: list[Any] | None = None  # parts of the open Responses assistant turn
    turns: list[dict[str, Any]] = []

    def open_turn() -> list[Any]:
        nonlocal turn
        if turn is None:
            turn = []
            message = {"role": "assistant", "content": turn}
            out.append(message)
            turns.append(message)
        return turn

    for item in messages:
        if not isinstance(item, dict):
            out.append(item)
            turn = None
            continue
        it = cast(dict[str, Any], item)
        kind = it.get("type")
        if "role" not in it and (
            kind in _RESPONSES_CALL_TYPES
            or kind == "reasoning"
            or (isinstance(kind, str) and kind.endswith("_call"))
        ):
            open_turn().extend(_normalize_part(it))
            continue
        if kind in _RESPONSES_CALL_OUTPUT_TYPES:
            out.append(
                {
                    "role": "tool",
                    "content": _normalize_content(it.get("output")),
                    "tool_call_id": it.get("call_id"),
                }
            )
            turn = None
            continue
        if "role" in it or (is_output and ("content" in it or "tool_calls" in it)):
            normalized = _normalize_message(it, is_output)
            last = normalized[-1]
            if (
                len(normalized) == 1
                and last["role"] == "assistant"
                and (turn is not None or kind == "message")
            ):
                content = last["content"]
                if isinstance(content, list):
                    open_turn().extend(cast(list[Any], content))
                elif content is not None:
                    open_turn().extend(_normalize_part(content))
                continue
            out.extend(normalized)
            turn = None
            continue
        out.append(it)
        turn = None

    for message in turns:
        message["content"] = _collapse(_dedupe_tool_calls(message["content"]))
    return out


def _sum_fields(details: dict[str, Any], keep: Callable[[str], bool]) -> float | None:
    values = [v for k, v in details.items() if keep(k) and isinstance(v, (int, float))]
    return sum(values) if values else None


def _is_input_key(k: str) -> bool:
    return k in ("input", "prompt_tokens", "input_tokens") or (
        k.startswith("input_") and k != "input_tokens"
    )


def _is_output_key(k: str) -> bool:
    return k in ("output", "completion_tokens", "output_tokens") or (
        k.startswith("output_") and k != "output_tokens"
    )


def usage_attributes(
    usage: Any,
    cost: Any,
    tokens: tuple[str, str, str, str, str, str],
    costs: tuple[str, str, str],
) -> dict[str, AttributeValue]:
    """Map Langfuse `usage_details` / `cost_details` to GenAI usage attributes.

    Langfuse's details are additive: `input` excludes cached tokens, which come
    as separate `input_*` keys (`input_cached_tokens`, `input_cache_read`,
    `input_cache_creation`, ...), and the same for `output_*` (reasoning).
    Laminar's `gen_ai.usage.input_tokens` includes cached tokens, so the totals
    are sums. `tokens` is (input, output, total, cache read, cache write,
    reasoning) attribute names; `costs` is (input, output, total).
    """
    out: dict[str, AttributeValue] = {}
    if isinstance(usage, dict):
        details = cast(dict[str, Any], usage)
        input_tokens = _sum_fields(details, _is_input_key)
        output_tokens = _sum_fields(details, _is_output_key)
        total_tokens = _sum_fields(details, lambda k: k in ("total", "total_tokens"))
        cache_read = _sum_fields(
            details, lambda k: k.startswith("input_") and ("cache_read" in k or "cached" in k)
        )
        cache_write = _sum_fields(
            details,
            lambda k: k.startswith("input_") and ("cache_creation" in k or "cache_write" in k),
        )
        reasoning = _sum_fields(details, lambda k: k.startswith("output_") and "reasoning" in k)
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        for name, value in zip(
            tokens,
            (input_tokens, output_tokens, total_tokens, cache_read, cache_write, reasoning),
            strict=True,
        ):
            if value is not None:
                out[name] = int(value)
    if isinstance(cost, dict):
        details = cast(dict[str, Any], cost)
        input_cost = _sum_fields(details, lambda k: k == "input" or k.startswith("input_"))
        output_cost = _sum_fields(details, lambda k: k == "output" or k.startswith("output_"))
        total_cost = _sum_fields(details, lambda k: k == "total")
        if total_cost is None and (input_cost is not None or output_cost is not None):
            total_cost = (input_cost or 0.0) + (output_cost or 0.0)
        for name, value in zip(costs, (input_cost, output_cost, total_cost), strict=True):
            if value is not None:
                out[name] = float(value)
    return out


# Model parameter keys (Langfuse `model_parameters`, openinference
# `llm.invocation_parameters`) -> the request attributes Laminar's own
# instrumentations write. The first key present wins (the max-token aliases);
# the rest stay in the raw attribute.
_NUMBER_MODEL_PARAMETERS = (
    ("temperature", GEN_AI_REQUEST_TEMPERATURE),
    ("top_p", GEN_AI_REQUEST_TOP_P),
    ("top_k", GEN_AI_REQUEST_TOP_K),
    ("max_tokens", GEN_AI_REQUEST_MAX_TOKENS),
    ("max_completion_tokens", GEN_AI_REQUEST_MAX_TOKENS),
    ("max_output_tokens", GEN_AI_REQUEST_MAX_TOKENS),
    ("frequency_penalty", GEN_AI_REQUEST_FREQUENCY_PENALTY),
    ("presence_penalty", GEN_AI_REQUEST_PRESENCE_PENALTY),
    ("seed", GEN_AI_REQUEST_SEED),
    ("n", GEN_AI_REQUEST_CHOICE_COUNT),
)
_STRING_MODEL_PARAMETERS = (
    ("reasoning_effort", GEN_AI_REQUEST_REASONING_LEVEL),
    ("reasoning_summary", GEN_AI_REQUEST_REASONING_SUMMARY),
    ("service_tier", GEN_AI_REQUEST_SERVICE_TIER),
)


def model_parameter_attributes(
    params: Any,
) -> tuple[dict[str, AttributeValue], dict[str, Any] | None]:
    """Request attributes from a parsed model parameters object, and the
    parameters left unmapped (None when `params` isn't an object)."""
    out: dict[str, AttributeValue] = {}
    if not isinstance(params, dict):
        return out, None
    rest = dict(cast(dict[str, Any], params))
    for key, attr in _NUMBER_MODEL_PARAMETERS:
        value = rest.get(key)
        # Python LangChain records an unset `max_tokens` as "Infinity".
        if isinstance(value, (int, float)) and not isinstance(value, bool) and attr not in out:
            out[attr] = value
            del rest[key]
    for key, attr in _STRING_MODEL_PARAMETERS:
        value = rest.get(key)
        if isinstance(value, str) and value:
            out[attr] = value
            del rest[key]
    stop = rest.get("stop")
    stops = [stop] if isinstance(stop, str) else stop
    if (
        isinstance(stops, list)
        and stops
        and all(isinstance(s, str) for s in cast(list[Any], stops))
    ):
        out[GEN_AI_REQUEST_STOP_SEQUENCES] = cast(list[str], stops)
        del rest["stop"]
    return out, rest


def oi_collect_indexed(attrs: dict[str, AttributeValue], prefix: str) -> list[dict[str, OIValue]]:
    """Reassemble openinference's flat indexed attributes into a list of dicts.

    OpenInference flattens nested structures into dotted keys with numeric
    indices, e.g. for `llm.input_messages`:

        llm.input_messages.0.message.role     = "user"
        llm.input_messages.0.message.content  = "hi"
        llm.input_messages.1.message.role     = "assistant"
        llm.input_messages.1.message.tool_calls.0.tool_call.function.name = ...

    Given the list prefix (`llm.input_messages`) this walks every matching key,
    parses the remaining dotted path (treating all-digit segments as list
    indices and everything else as dict keys), and rebuilds the nested
    structure. Returns the list ordered by leading index. Leaf JSON-string
    values (openinference stamps e.g. `*.arguments` as a JSON string) are left
    as-is; the caller decides whether to re-parse.
    """
    by_index: dict[int, dict[str, OIValue]] = {}
    plen = len(prefix) + 1
    for key, value in attrs.items():
        if not isinstance(key, str) or not key.startswith(prefix + "."):  # pyright: ignore[reportUnnecessaryIsInstance]
            continue
        rest = key[plen:]
        segments = rest.split(".")
        if not segments or not segments[0].isdigit():
            continue
        idx = int(segments[0])
        container = by_index.setdefault(idx, {})
        # openinference never emits a real sequence-valued attribute as one of
        # these leaves (see `OIValue`); a genuine list is its own family of
        # indexed dotted keys, reassembled above instead.
        _oi_assign_path(container, segments[1:], cast(OIValue, value))
    return [by_index[i] for i in sorted(by_index)]


def _oi_assign_path(container: dict[str, OIValue], segments: list[str], value: OIValue) -> None:
    """Assign `value` into `container` following a dotted openinference path.

    Numeric segments index into lists, named segments index into dicts. Lists
    are grown with placeholder dicts as needed. The first segment is always a
    dict key (openinference never starts a sub-path with an index once the
    leading list index has been stripped).
    """
    # `cast`, not a plain annotated assignment: pyright flow-narrows a
    # `cur: OIValue = container` assignment to `container`'s own concrete type
    # (`dict[str, OIValue]`) rather than the full `OIValue` union, since that's
    # the most specific type consistent with the declared annotation. The
    # `isinstance(cur, list)` checks below would then be narrowing a `dict`
    # against an unrelated `list`, which pyright resolves by synthesizing a
    # bogus `<subclass of dict and list>` instead of `list[OIValue]`. The cast
    # forces the flow type to the full union so isinstance narrows normally.
    cur: OIValue = cast(OIValue, container)
    for i, seg in enumerate(segments):
        last = i == len(segments) - 1
        nxt = segments[i + 1] if not last else None
        if seg.isdigit():
            seg_idx = int(seg)
            if not isinstance(cur, list):
                return
            while len(cur) <= seg_idx:
                cur.append({})
            if last:
                cur[seg_idx] = value
            else:
                if not isinstance(cur[seg_idx], (dict, list)):
                    cur[seg_idx] = [] if (nxt and nxt.isdigit()) else {}
                cur = cur[seg_idx]
        else:
            if not isinstance(cur, dict):
                return
            if last:
                cur[seg] = value
            else:
                child = cur.get(seg)
                if not isinstance(child, (dict, list)):
                    child = cast(OIValue, [] if (nxt and nxt.isdigit()) else {})
                    cur[seg] = child
                cur = child


def oi_message_to_genai(raw: dict[str, OIValue]) -> dict[str, OIValue]:
    """Convert one reassembled openinference message dict into an OpenAI-style
    GenAI message dict.

    OpenInference nests the message under a `message` key with fields like
    `role`, `content`, `contents` (multi-part), `tool_calls` (each
    `{tool_call: {id, function: {name, arguments}}}`), `tool_call_id`,
    `function_call_name` / `function_call_arguments_json`. We flatten that into
    the `{role, content, tool_calls: [{id, type, function: {...}}]}` shape
    Laminar's other instrumentors emit.
    """
    msg = raw.get("message", raw) if isinstance(raw, dict) else {}  # pyright: ignore[reportUnnecessaryIsInstance]
    if not isinstance(msg, dict):
        return {"role": "assistant", "content": str(msg)}
    out: dict[str, OIValue] = {"role": msg.get("role") or "assistant"}

    content = msg.get("content")
    contents = msg.get("contents")
    if content is not None:
        out["content"] = content
    elif isinstance(contents, list):
        out["content"] = contents

    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        converted: list[OIValue] = []
        for tc in tool_calls:
            inner = tc.get("tool_call", tc) if isinstance(tc, dict) else {}  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(inner, dict):
                continue
            fn = inner.get("function", {})
            fn = fn if isinstance(fn, dict) else {}
            converted.append(
                {
                    "id": inner.get("id"),
                    "type": "tool_call",
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments"),
                }
            )
        if converted:
            existing = out.get("content")
            if isinstance(existing, list):
                out["content"] = [*existing, *converted]
            elif isinstance(existing, str):
                out["content"] = [{"type": "text", "text": existing}, *converted]
            else:
                out["content"] = converted

    if msg.get("function_call_name") is not None:
        out["function_call"] = {
            "name": msg.get("function_call_name"),
            "arguments": msg.get("function_call_arguments_json"),
        }
    if msg.get("tool_call_id") is not None:
        out["tool_call_id"] = msg.get("tool_call_id")
    if msg.get("name") is not None:
        out["name"] = msg.get("name")
    return out


def oi_tool_to_genai(raw: dict[str, OIValue]) -> OIValue:
    """Convert one reassembled openinference tool dict into a GenAI tool
    definition.

    OpenInference stamps each tool as `llm.tools.K.tool.json_schema`, where the
    value is the full JSON-schema tool definition (usually a JSON string). We
    return the parsed schema so it lands in `gen_ai.tool.definitions` in the
    same shape OpenAI/litellm tools use.
    """
    tool = raw.get("tool", raw) if isinstance(raw, dict) else raw  # pyright: ignore[reportUnnecessaryIsInstance]
    if isinstance(tool, dict) and "json_schema" in tool:
        return cast(OIValue, parse_json(tool["json_schema"]))
    return tool


def is_openinference_span(span: ReadableSpan) -> bool:
    attrs = span.attributes or {}
    if OI_SPAN_KIND in attrs:
        return True
    return any(
        isinstance(k, str)  # pyright: ignore[reportUnnecessaryIsInstance]
        and (
            k.startswith(
                (OI_LLM_INPUT_MESSAGES + ".", OI_LLM_OUTPUT_MESSAGES + ".", "llm.token_count.")
            )
        )
        for k in attrs
    )


def is_langfuse_span(span: ReadableSpan) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    if scope is not None and scope.name == LANGFUSE_TRACER_NAME:
        return True
    attrs = span.attributes or {}
    return any(isinstance(k, str) and k.startswith("langfuse.") for k in attrs)  # pyright: ignore[reportUnnecessaryIsInstance]

"""Pure functions that convert Langfuse / OpenInference span shapes into
Laminar / OTel GenAI shapes.

No `SpanProcessor` state lives here — see `processor.py` for the
`SpanProcessor` that calls into these.
"""

from __future__ import annotations

import json
from typing import Any, TypeAlias, cast

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse.attributes import (
    _OI_LLM_INPUT_MESSAGES,  # pyright: ignore[reportPrivateUsage],
    _OI_LLM_OUTPUT_MESSAGES,  # pyright: ignore[reportPrivateUsage],
    _OI_SPAN_KIND,  # pyright: ignore[reportPrivateUsage],
    LANGFUSE_TRACER_NAME,
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



def parse_json(raw: Any) -> Any:  # pyright: ignore[reportAny, reportExplicitAny]
    if not isinstance(raw, str):
        return raw  # pyright: ignore[reportAny]
    try:
        return json.loads(raw)  # pyright: ignore[reportAny]
    except (ValueError, TypeError):
        return raw


def genai_input_from_langfuse(input_raw: Any) -> tuple[list[Any], Any] | None:  # pyright: ignore[reportAny, reportExplicitAny]
    """Split a Langfuse LLM input into (messages, tool_definitions).

    Langfuse's OpenAI integration ships the call input in one of two shapes
    (see `langfuse.openai._extract_chat_prompt`):

    - A bare list of OpenAI-style message dicts (`[{role, content}, ...]`) when
      the caller passed no `tools`/`functions`.
    - A dict `{"messages": [...], "tools": [...], "functions": [...],
      "function_call": ...}` when tools/functions were supplied.

    Laminar (and the app-server's GenAI parser) want these split into
    `gen_ai.input.messages` (the message array) and `gen_ai.tool.definitions`
    (the tool/function array) — dumping the whole dict into `lmnr.span.input`
    leaves them unparsed in the UI. Returns `(messages, tools_or_none)` when
    the shape is recognized, or `None` when it isn't (caller falls back to the
    raw input). Langchain-style inputs share the OpenAI message shape, so the
    same split applies.
    """
    parsed = parse_json(input_raw)  # pyright: ignore[reportAny]
    if isinstance(parsed, list):
        return parsed, None  # pyright: ignore[reportUnknownVariableType]
    if isinstance(parsed, dict) and isinstance(cast(dict[str, Any], parsed).get("messages"), list):  # pyright: ignore[reportExplicitAny]
        tools = cast(dict[str, Any], parsed).get("tools")  # pyright: ignore[reportExplicitAny]
        if tools is None:
            tools = parsed.get("functions")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        return parsed["messages"], tools  # pyright: ignore[reportUnknownVariableType]
    return None


def split_messages_and_tool_defs_langchain(
    messages: Any,  # pyright: ignore[reportAny, reportExplicitAny]
) -> tuple[Any, list[Any] | None]:  # pyright: ignore[reportExplicitAny]
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
                    content.get("function"), dict  # pyright: ignore[reportUnknownMemberType]
                ):
                    tool_defs.append(content["function"])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    continue
                if "input_schema" in content and "name" in content:
                    tool_defs.append(content)  # pyright: ignore[reportUnknownMemberType]
                    continue
        new_msgs.append(msg)  # pyright: ignore[reportUnknownMemberType]
    return (new_msgs, tool_defs)  # pyright: ignore[reportUnknownVariableType]


def genai_output_from_langfuse(output_raw: Any) -> Any | None:  # pyright: ignore[reportAny, reportExplicitAny]
    """Normalize a Langfuse LLM output into a `gen_ai.output.messages` array.

    Langfuse's OpenAI integration emits the response as a single message dict
    (`{role, content, tool_calls, function_call, audio}` — see
    `_extract_chat_response`). The GenAI convention is an array of such message
    dicts (one per choice), matching what Laminar's own litellm wrapper stamps.
    A single dict is wrapped into a one-element list; an already-list value is
    passed through. Returns `None` for any other shape so the caller falls back
    to `lmnr.span.output`.
    """
    parsed = parse_json(output_raw)  # pyright: ignore[reportAny]
    if isinstance(parsed, list):
        return parsed  # pyright: ignore[reportUnknownVariableType]
    if isinstance(parsed, dict):
        return [parsed]
    return None


def convert_openai_tool_calls_to_content_parts(
    messages: list[Any], is_output: bool  # pyrgiht: ignore[reportExplicitAny]
) -> list[Any]:  # pyright: ignore[reportExplicitAny]
    """The output of LangChain integration looks a lot like OpenAI's output,
    i.e. separate tool_calls and content keys. This function extracts the tool
    calls and converts them to a more generic content-part style.

    Three assistant-message shapes occur:

    * OpenAI/langchain: string ``content`` + a separate ``tool_calls`` list. The
      string is wrapped into a text part and the tool calls are appended as
      content parts.
    * Anthropic/langchain: list ``content`` that ALREADY embeds the calls as
      ``{"type": "tool_use", ...}`` blocks, PLUS a redundant top-level
      ``tool_calls`` list mirroring the same calls. Keep the content blocks and
      drop the duplicate top-level ``tool_calls`` so the call isn't rendered
      twice.
    * OpenAI (from langfuse.openai). Function calls are content blocks of
      ``{"type": "function", "function": {...}}``. If such are detected,
      the entire message is best-effort wrapped into an OpenAI choices schema.
      This is only relevant to the output messages.
    """

    def is_assistant(msg: Any) -> bool:  # pyright: ignore[reportAny, reportExplicitAny]
        return isinstance(msg, dict) and msg.get("role") == "assistant"  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    def has_inlined_tool_calls(content: Any) -> bool:  # pyright: ignore[reportAny, reportExplicitAny]
        return isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in ("tool_use", "tool_call")  # pyright: ignore[reportUnknownMemberType]
            for part in content  # pyright: ignore[reportUnknownVariableType]
        )

    def is_raw_openai_tool_call_format(tc: Any) -> bool:  # pyright: ignore[reportExplicitAny, reportAny]
        return (  # pyright: ignore[reportUnknownVariableType]
            isinstance(tc, dict)
            and tc.get("type") == "function"  # pyright: ignore[reportUnknownMemberType]
            and isinstance(tc.get("function"), dict)  # pyright: ignore[reportUnknownMemberType]
        )

    def normalize(msg: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return msg
        # OpenAI output choice. Guard on a NON-EMPTY tool_calls list:
        # `all([])` is True, so a plain text completion that carries
        # `tool_calls: []` would otherwise be wrapped into the choices shape
        # and break transcript rendering.
        if (
            is_output
            and tool_calls
            and all(is_raw_openai_tool_call_format(tc) for tc in tool_calls)  # pyright:ignore[reportUnknownVariableType]
        ):
            return {"message": msg}
        # Anthropic: calls already embedded in the content blocks — drop the
        # redundant mirror to avoid double rendering.
        if has_inlined_tool_calls(content):
            return {k: v for k, v in msg.items() if k != "tool_calls"}  # pyright: ignore[reportAny]
        # OpenAI tool calls as "function" in input convert to tool_call block
        if not is_output:
            new_tool_calls = []
            for tc in tool_calls:  # pyright: ignore[reportUnknownVariableType]
                if is_raw_openai_tool_call_format(tc):
                    fn = tc["function"]  # pyright: ignore[reportUnknownVariableType]
                    new_tool_calls.append(  # pyright: ignore[reportUnknownMemberType]
                        {
                            "id": tc.get("id"),  # pyright: ignore[reportUnknownMemberType]
                            "type": "tool_call",
                            "name": fn.get("name"),  # pyright: ignore[reportUnknownMemberType]
                            "arguments": fn.get("arguments"),  # pyright: ignore[reportUnknownMemberType]
                        }
                    )
                else:
                    new_tool_calls.append(tc)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            tool_calls = new_tool_calls

        # OpenAI: fold the separate tool_calls into the content parts.
        new_cnt = (  # pyright: ignore [reportUnknownVariableType]
            content
            if isinstance(content, list)
            else [{"type": "text", "text": content}]
            if isinstance(content, str) and content != ""
            else []
        )
        return {
            "role": msg.get("role"),
            "content": [*new_cnt, *tool_calls],
        }

    return [normalize(msg) if is_assistant(msg) else msg for msg in messages]  # pyright: ignore[reportAny]


def usage_field(usage: Any, *keys: str) -> int | None:  # pyright: ignore[reportExplicitAny, reportAny]
    if not isinstance(usage, dict):
        return None
    for k in keys:
        v = usage.get(k)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(v, (int, float)):
            return int(v)
    return None


def cost_field(cost: Any, *keys: str) -> float | None:  # pyright: ignore[reportExplicitAny, reportAny]
    if not isinstance(cost, dict):
        return None
    for k in keys:
        v = cost.get(k)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if isinstance(v, (int, float)):
            return float(v)
    return None


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


def _oi_assign_path(
    container: dict[str, OIValue], segments: list[str], value: OIValue
) -> None:
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
    if _OI_SPAN_KIND in attrs:
        return True
    return any(
        isinstance(k, str)  # pyright: ignore[reportUnnecessaryIsInstance]
        and (
            k.startswith((_OI_LLM_INPUT_MESSAGES + ".", _OI_LLM_OUTPUT_MESSAGES + ".", "llm.token_count."))
        )
        for k in attrs
    )


def is_llm_span(span: Span) -> bool:
    """True if `span` is already typed as an LLM span.

    Used to gate LiteLLM's primary-span forcing: when the active parent is
    Laminar's own `litellm.completion` LLM span (present when
    `Instruments.LITELLM` runs alongside the bridge), letting LiteLLM fold its
    `gen_ai.*` attrs onto that parent is the correct deduplicated shape, so we
    must NOT redirect it into a second nested `litellm_request` LLM span.
    Reads the live (still-recording) span's attributes defensively — any
    unexpected span shape falls back to False (fold not LLM → safe to force).
    """
    try:
        attrs = getattr(span, "attributes", None) or {}  # pyright: ignore[reportUnknownVariableType]
        if attrs.get("lmnr.span.type") == "LLM":  # pyright: ignore[reportUnknownMemberType]
            return True
        if attrs.get(_OI_SPAN_KIND) == "LLM":  # pyright: ignore[reportUnknownMemberType]
            return True
        return any(
            isinstance(k, str)
            and (
                k == "gen_ai.request.model"
                or k == "gen_ai.response.model"
                or k == "gen_ai.system"
            )
            for k in attrs  # pyright: ignore[reportUnknownVariableType]
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def is_langfuse_span(span: ReadableSpan) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    if scope is not None and scope.name == LANGFUSE_TRACER_NAME:  # pyright: ignore[reportAny]
        return True
    attrs = span.attributes or {}
    return any(isinstance(k, str) and k.startswith("langfuse.") for k in attrs)  # pyright: ignore[reportUnnecessaryIsInstance]

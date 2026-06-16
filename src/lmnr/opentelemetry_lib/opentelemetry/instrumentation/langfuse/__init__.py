"""OpenTelemetry Langfuse instrumentation.

Langfuse v3 is a thin wrapper over OpenTelemetry: its `Langfuse(...)` constructor
registers a `LangfuseSpanProcessor` (subclass of `BatchSpanProcessor`) on a real
`TracerProvider`, creating one globally if none exists, and shipping spans via
OTLP/HTTP to `/api/public/otel/v1/traces`. Every `@observe`, `langfuse.openai`,
and `langfuse.langchain` path flows through that one provider.

This instrumentor makes Laminar and Langfuse coexist so a user can run both
SDKs side-by-side with zero extra config — spans emitted by Langfuse show up in
Laminar with correct LLM / tool / trace attribution.

Two operations are performed:

1. **Attach Laminar's `SpanProcessor` to Langfuse's `TracerProvider`.**
   - Covers the case where Langfuse initialized first and owns the global
     `TracerProvider` — Laminar's provider would otherwise never see those
     spans.
   - Iterates `LangfuseResourceManager._instances` (module-level singleton
     dict keyed by public_key) and attaches to every Langfuse-owned
     `TracerProvider`.
   - Monkey-patches `LangfuseResourceManager._initialize_instance` so any
     Langfuse client constructed AFTER this instrumentor runs also gets its
     processor dual-attached. Idempotent — we track handled providers by id().

2. **Translate `langfuse.*` attributes → Laminar / OTel GenAI.**
   - A lightweight `SpanProcessor` sits in front of Laminar's exporter and
     rewrites attributes on `on_end` so the span renders with correct model,
     tokens, cost, input/output in the Laminar UI.
   - Langfuse-scoped spans (detected by instrumentation_scope.name ==
     "langfuse-sdk" or any `langfuse.*` attribute) have LLM input/output split
     into the GenAI message conventions (`gen_ai.input.messages` /
     `gen_ai.output.messages` / `gen_ai.tool.definitions`).
   - openinference-instrumented spans (groq / google_genai, which Langfuse's
     docs recommend) carry a different flat/indexed attribute layout and no
     `langfuse.*` keys; they're detected and translated separately.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Collection

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider

from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SESSION_ID,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_TYPE,
    USER_ID,
)
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import json_dumps

logger = get_default_logger(__name__)

LANGFUSE_TRACER_NAME = "langfuse-sdk"

# Langfuse attribute names (mirrors langfuse._client.attributes.LangfuseOtelSpanAttributes)
# Duplicated here so this module has no hard import dependency on langfuse.
_TRACE_INPUT = "langfuse.trace.input"
_TRACE_OUTPUT = "langfuse.trace.output"
_TRACE_TAGS = "langfuse.trace.tags"
_TRACE_METADATA_PREFIX = "langfuse.trace.metadata"
_TRACE_USER_ID = "user.id"
_TRACE_SESSION_ID = "session.id"

_OBSERVATION_TYPE = "langfuse.observation.type"
_OBSERVATION_INPUT = "langfuse.observation.input"
_OBSERVATION_OUTPUT = "langfuse.observation.output"
_OBSERVATION_MODEL = "langfuse.observation.model.name"
_OBSERVATION_USAGE_DETAILS = "langfuse.observation.usage_details"
_OBSERVATION_COST_DETAILS = "langfuse.observation.cost_details"
_OBSERVATION_METADATA_PREFIX = "langfuse.observation.metadata"

_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
_GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
_GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
_GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
_GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"
_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_GEN_AI_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens"
_GEN_AI_USAGE_INPUT_COST = "gen_ai.usage.input_cost"
_GEN_AI_USAGE_OUTPUT_COST = "gen_ai.usage.output_cost"
_GEN_AI_USAGE_TOTAL_COST = "gen_ai.usage.cost"

# Langfuse "observation types" that represent LLM calls. See
# langfuse._client.constants.ObservationTypeGenerationLike. Any of these maps
# to Laminar's LLM span type.
_LLM_OBSERVATION_TYPES = {"generation", "completion", "embedding"}
_TOOL_OBSERVATION_TYPES = {"tool"}

# --- OpenInference semantic conventions -------------------------------------
# Langfuse's docs recommend openinference instrumentations for groq and
# google_genai (e.g. `openinference-instrumentation-google-genai`). Those emit
# a flat, indexed attribute layout — `llm.input_messages.0.message.role`,
# `llm.input_messages.0.message.content`, `llm.output_messages.0...`,
# `llm.token_count.prompt`, `llm.tools.0.tool.json_schema`, etc. — completely
# different from both Langfuse's `langfuse.*` blobs and Laminar's GenAI shape.
# See the openinference-semantic-conventions package in the Arize-ai/
# openinference repo. We translate the flat layout into Laminar / OTel GenAI
# conventions.
_OI_SPAN_KIND = "openinference.span.kind"
_OI_LLM_MODEL_NAME = "llm.model_name"
_OI_LLM_INPUT_MESSAGES = "llm.input_messages"
_OI_LLM_OUTPUT_MESSAGES = "llm.output_messages"
_OI_LLM_TOOLS = "llm.tools"
_OI_TOKEN_PROMPT = "llm.token_count.prompt"
_OI_TOKEN_COMPLETION = "llm.token_count.completion"
_OI_TOKEN_TOTAL = "llm.token_count.total"
_OI_INPUT_VALUE = "input.value"
_OI_OUTPUT_VALUE = "output.value"
# openinference.span.kind values that mean "LLM call".
_OI_LLM_SPAN_KINDS = {"LLM"}


def _parse_json(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _genai_input_from_langfuse(input_raw: Any) -> tuple[Any, Any] | None:
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
    parsed = _parse_json(input_raw)
    if isinstance(parsed, list):
        return parsed, None
    if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
        tools = parsed.get("tools")
        if tools is None:
            tools = parsed.get("functions")
        return parsed["messages"], tools
    return None


def _split_messages_and_tool_defs_langchain(
    messages: Any,
) -> tuple[Any, list[Any] | None]:
    """Splits langfuse.langchain input into the actual messages and tool definitions

    The callback inlines everything into one array, where tool definitions are messages
    with role "tool" and "content" dict with type "function" and "function" key, as per
    gen_ai.tool.definitions
    """
    if not isinstance(messages, list):
        return (messages, None)
    new_msgs = []
    tool_defs = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            if content := msg.get("content"):
                if (
                    isinstance(content, dict)
                    and content["type"] == "function"
                    and isinstance(content["function"], dict)
                ):
                    tool_defs.append(content["function"])
                    continue
        new_msgs.append(msg)
    return (new_msgs, tool_defs)


def _genai_output_from_langfuse(output_raw: Any) -> Any | None:
    """Normalize a Langfuse LLM output into a `gen_ai.output.messages` array.

    Langfuse's OpenAI integration emits the response as a single message dict
    (`{role, content, tool_calls, function_call, audio}` — see
    `_extract_chat_response`). The GenAI convention is an array of such message
    dicts (one per choice), matching what Laminar's own litellm wrapper stamps.
    A single dict is wrapped into a one-element list; an already-list value is
    passed through. Returns `None` for any other shape so the caller falls back
    to `lmnr.span.output`.
    """
    parsed = _parse_json(output_raw)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return None


def _convert_openai_tool_calls_to_content_parts(messages: list[Any]) -> list[Any]:
    """The output of LangChain integration looks a lot like OpenAI's output,
    i.e. separate tool_calls and content keys. This function wraps extracts
    tool calls and converts them to a more generic content part style tool
    calls
    """

    def looks_like_openai_assistant(msg: Any) -> bool:
        return (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and isinstance(msg.get("content"), str)
            and isinstance(msg.get("tool_calls"), list)
        )

    def wrap_in_choice_shape(msg: dict) -> dict:
        content = msg.get("content")
        new_cnt = (
            content
            if isinstance(content, list)
            else [{"type": "text", "text": content}]
            if isinstance(content, str)
            else []
        )
        return {
            "role": msg.get("role"),
            "content": [*new_cnt, *msg.get("tool_calls", [])],
        }

    return [
        wrap_in_choice_shape(msg) if looks_like_openai_assistant(msg) else msg
        for msg in messages
    ]


def _usage_field(usage: Any, *keys: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    for k in keys:
        v = usage.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return None


def _cost_field(cost: Any, *keys: str) -> float | None:
    if not isinstance(cost, dict):
        return None
    for k in keys:
        v = cost.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _oi_collect_indexed(attrs: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
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
    by_index: dict[int, Any] = {}
    plen = len(prefix) + 1
    for key, value in attrs.items():
        if not isinstance(key, str) or not key.startswith(prefix + "."):
            continue
        rest = key[plen:]
        segments = rest.split(".")
        if not segments or not segments[0].isdigit():
            continue
        idx = int(segments[0])
        container = by_index.setdefault(idx, {})
        _oi_assign_path(container, segments[1:], value)
    return [by_index[i] for i in sorted(by_index)]


def _oi_assign_path(container: dict[str, Any], segments: list[str], value: Any) -> None:
    """Assign `value` into `container` following a dotted openinference path.

    Numeric segments index into lists, named segments index into dicts. Lists
    are grown with placeholder dicts as needed. The first segment is always a
    dict key (openinference never starts a sub-path with an index once the
    leading list index has been stripped).
    """
    cur: Any = container
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
                    child = [] if (nxt and nxt.isdigit()) else {}
                    cur[seg] = child
                cur = child


def _oi_message_to_genai(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert one reassembled openinference message dict into an OpenAI-style
    GenAI message dict.

    OpenInference nests the message under a `message` key with fields like
    `role`, `content`, `contents` (multi-part), `tool_calls` (each
    `{tool_call: {id, function: {name, arguments}}}`), `tool_call_id`,
    `function_call_name` / `function_call_arguments_json`. We flatten that into
    the `{role, content, tool_calls: [{id, type, function: {...}}]}` shape
    Laminar's other instrumentors emit.
    """
    msg = raw.get("message", raw) if isinstance(raw, dict) else {}
    if not isinstance(msg, dict):
        return {"role": "assistant", "content": str(msg)}
    out: dict[str, Any] = {"role": msg.get("role") or "assistant"}

    content = msg.get("content")
    contents = msg.get("contents")
    if content is not None:
        out["content"] = content
    elif isinstance(contents, list):
        out["content"] = contents

    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        converted = []
        for tc in tool_calls:
            inner = tc.get("tool_call", tc) if isinstance(tc, dict) else {}
            if not isinstance(inner, dict):
                continue
            fn = inner.get("function", {})
            fn = fn if isinstance(fn, dict) else {}
            converted.append(
                {
                    "id": inner.get("id"),
                    "type": "function",
                    "function": {
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments"),
                    },
                }
            )
        if converted:
            out["tool_calls"] = converted

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


def _oi_tool_to_genai(raw: dict[str, Any]) -> Any:
    """Convert one reassembled openinference tool dict into a GenAI tool
    definition.

    OpenInference stamps each tool as `llm.tools.K.tool.json_schema`, where the
    value is the full JSON-schema tool definition (usually a JSON string). We
    return the parsed schema so it lands in `gen_ai.tool.definitions` in the
    same shape OpenAI/litellm tools use.
    """
    tool = raw.get("tool", raw) if isinstance(raw, dict) else raw
    if isinstance(tool, dict) and "json_schema" in tool:
        return _parse_json(tool["json_schema"])
    return tool


def _is_openinference_span(span: ReadableSpan) -> bool:
    attrs = span.attributes or {}
    if _OI_SPAN_KIND in attrs:
        return True
    return any(
        isinstance(k, str)
        and (
            k.startswith(_OI_LLM_INPUT_MESSAGES + ".")
            or k.startswith(_OI_LLM_OUTPUT_MESSAGES + ".")
            or k.startswith("llm.token_count.")
        )
        for k in attrs.keys()
    )


def _prepend_span_processor(provider: Any, processor: SpanProcessor) -> bool:
    """Add `processor` to `provider` so that it runs BEFORE any processors
    already registered on the provider.

    Ordering matters: the translator must mutate `langfuse.*` attributes
    before the exporter (`LaminarSpanProcessor` wrapping `SimpleSpanProcessor`
    when `disable_batch=True`) consumes the span in `on_end` — otherwise the
    exporter ships the pre-translation shape. `TracerProvider.add_span_processor`
    appends to the end, so we call it, then reorder the underlying tuple.

    The reorder touches `SynchronousMultiSpanProcessor._span_processors` /
    `ConcurrentMultiSpanProcessor._span_processors` — both expose the same
    private layout and lock. If the attribute shape changes upstream we fall
    back to a plain `add_span_processor` and log.
    """
    add = getattr(provider, "add_span_processor", None)
    if not callable(add):
        return False
    add(processor)
    active = getattr(provider, "_active_span_processor", None)
    if active is None:
        return True
    lock = getattr(active, "_lock", None)
    current = getattr(active, "_span_processors", None)
    if current is None:
        return True
    try:
        new_order = (processor,) + tuple(p for p in current if p is not processor)
        if lock is not None:
            with lock:
                active._span_processors = new_order
        else:
            active._span_processors = new_order
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug(
            "Could not reorder span processors on %r; translator will run "
            "after the exporter (%s)",
            provider,
            exc,
        )
    return True


def _remove_span_processor(provider: Any, processor: SpanProcessor) -> bool:
    """Remove `processor` from `provider`'s active span processor, if present.

    Mirror of `_prepend_span_processor` for the uninstall path: touches the
    private `_span_processors` tuple on `SynchronousMultiSpanProcessor` /
    `ConcurrentMultiSpanProcessor` under its lock. Returns True if the
    processor was removed, False otherwise.
    """
    active = getattr(provider, "_active_span_processor", None)
    if active is None:
        return False
    lock = getattr(active, "_lock", None)
    current = getattr(active, "_span_processors", None)
    if current is None:
        return False
    try:
        filtered = tuple(p for p in current if p is not processor)
        if len(filtered) == len(current):
            return False
        if lock is not None:
            with lock:
                active._span_processors = filtered
        else:
            active._span_processors = filtered
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.debug(
            "Could not remove span processor from %r (%s)",
            provider,
            exc,
        )
        return False
    return True


def _is_langfuse_span(span: ReadableSpan) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    if scope is not None and scope.name == LANGFUSE_TRACER_NAME:
        return True
    attrs = span.attributes or {}
    return any(isinstance(k, str) and k.startswith("langfuse.") for k in attrs.keys())


class LangfuseAttributeTranslator(SpanProcessor):
    """Rewrites `langfuse.*` attributes to Laminar / OTel GenAI attributes.

    Runs as its own `SpanProcessor` on Laminar's `TracerProvider`. On `on_end`,
    if the span looks like a Langfuse span, we mutate its attributes in place —
    Laminar's `LaminarSpanProcessor` (also attached to the same provider) then
    sees the translated shape when it exports to Laminar's OTLP endpoint.

    Mutation-in-place is safe because the `ReadableSpan` handed to `on_end`
    shares its `_attributes` dict with the underlying recording `Span`, which
    is the same object the exporter eventually serializes. We cannot call
    `span.set_attribute(...)` here: `on_end` receives a `ReadableSpan` (no
    `set_attribute` method), and even if it were the recording `Span`, the
    span is already ended at this point and `set_attribute` would be a no-op.
    We write to `span._attributes` directly instead.
    """

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        # Routing precedence — openinference FIRST:
        #   * groq / google_genai (the openinference instrumentations Langfuse
        #     recommends) emit purely openinference `llm.*` attrs, no
        #     `langfuse.*` keys.
        #   * LiteLLM's `langfuse_otel` callback emits a HYBRID: openinference
        #     `llm.*` attrs (model, tokens, indexed messages, tools, span kind)
        #     AND `langfuse.*` attrs — but it never sets
        #     `langfuse.observation.type`, so the langfuse path can't tell the
        #     span is an LLM call and would miss the model/tokens/messages.
        #     The openinference path carries all of that, so it wins; it also
        #     promotes the `langfuse.*` trace-level session/user/metadata.
        #   * Real Langfuse-SDK spans never carry `openinference.span.kind` /
        #     `llm.token_count.*` / `llm.input_messages.*`, so they fall
        #     through to the langfuse path.
        try:
            if _is_openinference_span(span):
                self._translate_openinference(span)
            elif _is_langfuse_span(span):
                self._translate(span)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Langfuse attribute translation failed: %s", exc)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    @staticmethod
    def _translate(span: ReadableSpan) -> None:
        attrs = dict(span.attributes or {})
        if not attrs:
            return
        new_attrs: dict[str, Any] = {}

        is_llm = False
        obs_type = attrs.get(_OBSERVATION_TYPE)
        if isinstance(obs_type, str):
            if obs_type.lower() in _LLM_OBSERVATION_TYPES:
                new_attrs[SPAN_TYPE] = "LLM"
                is_llm = True
            elif obs_type.lower() in _TOOL_OBSERVATION_TYPES:
                new_attrs[SPAN_TYPE] = "TOOL"

        # Model
        model = attrs.get(_OBSERVATION_MODEL)
        if isinstance(model, str) and model:
            new_attrs.setdefault(_GEN_AI_REQUEST_MODEL, model)
            new_attrs.setdefault(_GEN_AI_RESPONSE_MODEL, model)

        # Usage details (tokens)
        usage_raw = _parse_json(attrs.get(_OBSERVATION_USAGE_DETAILS))
        input_tokens = _usage_field(usage_raw, "input", "prompt_tokens", "input_tokens")
        output_tokens = _usage_field(
            usage_raw, "output", "completion_tokens", "output_tokens"
        )
        total_tokens = _usage_field(usage_raw, "total", "total_tokens")
        if input_tokens is not None:
            new_attrs[_GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
        if output_tokens is not None:
            new_attrs[_GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
        if total_tokens is None and (
            input_tokens is not None or output_tokens is not None
        ):
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        if total_tokens is not None:
            new_attrs[_GEN_AI_USAGE_TOTAL_TOKENS] = total_tokens

        # Cost details
        cost_raw = _parse_json(attrs.get(_OBSERVATION_COST_DETAILS))
        input_cost = _cost_field(cost_raw, "input")
        output_cost = _cost_field(cost_raw, "output")
        total_cost = _cost_field(cost_raw, "total")
        if input_cost is not None:
            new_attrs[_GEN_AI_USAGE_INPUT_COST] = input_cost
        if output_cost is not None:
            new_attrs[_GEN_AI_USAGE_OUTPUT_COST] = output_cost
        if total_cost is None and (input_cost is not None or output_cost is not None):
            total_cost = (input_cost or 0.0) + (output_cost or 0.0)
        if total_cost is not None:
            new_attrs[_GEN_AI_USAGE_TOTAL_COST] = total_cost

        # Input / output — prefer observation-level, fall back to trace-level.
        # For LLM observations we translate into the GenAI message conventions
        # (`gen_ai.input.messages` / `gen_ai.output.messages` /
        # `gen_ai.tool.definitions`) so the Laminar UI renders them as a chat
        # transcript and the app-server parses tokens/tools correctly. For
        # non-LLM observations (or when the LLM shape isn't recognized) we fall
        # back to the raw `lmnr.span.input/output` blob.
        input_raw = attrs.get(_OBSERVATION_INPUT)
        if input_raw is None:
            input_raw = attrs.get(_TRACE_INPUT)
        input_handled = False
        if is_llm and input_raw is not None:
            split = _genai_input_from_langfuse(input_raw)
            if split is not None:
                messages, tools = split
                messages, lc_tools = _split_messages_and_tool_defs_langchain(messages)
                messages = _convert_openai_tool_calls_to_content_parts(messages)
                new_attrs[_GEN_AI_INPUT_MESSAGES] = json_dumps(messages)
                if tools:
                    new_attrs[_GEN_AI_TOOL_DEFINITIONS] = json_dumps(tools)
                elif lc_tools:
                    new_attrs[_GEN_AI_TOOL_DEFINITIONS] = json_dumps(lc_tools)
                input_handled = True
        if (
            not input_handled
            and isinstance(input_raw, str)
            and input_raw
            and SPAN_INPUT not in attrs
        ):
            new_attrs[SPAN_INPUT] = input_raw

        output_raw = attrs.get(_OBSERVATION_OUTPUT)
        if output_raw is None:
            output_raw = attrs.get(_TRACE_OUTPUT)
        output_handled = False
        if is_llm and output_raw is not None:
            messages = _genai_output_from_langfuse(output_raw)
            if messages is not None:
                if isinstance(messages, list):
                    messages = _convert_openai_tool_calls_to_content_parts(messages)
                new_attrs[_GEN_AI_OUTPUT_MESSAGES] = json_dumps(messages)
                output_handled = True
        if (
            not output_handled
            and isinstance(output_raw, str)
            and output_raw
            and SPAN_OUTPUT not in attrs
        ):
            new_attrs[SPAN_OUTPUT] = output_raw

        LangfuseAttributeTranslator._promote_trace_attributes(attrs, new_attrs)
        LangfuseAttributeTranslator._write_attrs(span, new_attrs)

    @staticmethod
    def _promote_trace_attributes(
        attrs: dict[str, Any], new_attrs: dict[str, Any]
    ) -> None:
        """Promote Langfuse trace-level `langfuse.*` attrs to Laminar
        association properties.

        Shared by both the langfuse and openinference translators: LiteLLM's
        `langfuse_otel` callback stamps `session.id` / `user.id` /
        `langfuse.trace.*` alongside the openinference `llm.*` attrs, so the
        openinference path needs this promotion too (the openinference
        instrumentations for groq / google_genai simply won't have these keys,
        making it a harmless no-op there).
        """
        # Session / user id — promote so the trace groups by session in the UI.
        session_id = attrs.get(_TRACE_SESSION_ID)
        if isinstance(session_id, str) and session_id:
            new_attrs[f"{ASSOCIATION_PROPERTIES}.{SESSION_ID}"] = session_id
        user_id = attrs.get(_TRACE_USER_ID)
        if isinstance(user_id, str) and user_id:
            new_attrs[f"{ASSOCIATION_PROPERTIES}.{USER_ID}"] = user_id

        # Tags
        tags = attrs.get(_TRACE_TAGS)
        if isinstance(tags, (list, tuple)) and tags:
            new_attrs[f"{ASSOCIATION_PROPERTIES}.tags"] = list(tags)

        # Trace / observation metadata — flat form is `langfuse.trace.metadata.<k>`,
        # unflattened form is `langfuse.trace.metadata`. Route both into Laminar's
        # `lmnr.association.properties.metadata.<k>` namespace.
        for k, v in attrs.items():
            if not isinstance(k, str):
                continue
            for prefix in (_TRACE_METADATA_PREFIX, _OBSERVATION_METADATA_PREFIX):
                if k == prefix:
                    parsed = _parse_json(v)
                    if isinstance(parsed, dict):
                        for mk, mv in parsed.items():
                            new_attrs[f"{ASSOCIATION_PROPERTIES}.metadata.{mk}"] = (
                                mv
                                if isinstance(mv, (str, int, float, bool))
                                else json.dumps(mv)
                            )
                    break
                if k.startswith(prefix + "."):
                    sub = k[len(prefix) + 1 :]
                    new_attrs[f"{ASSOCIATION_PROPERTIES}.metadata.{sub}"] = v
                    break

    @staticmethod
    def _write_attrs(span: ReadableSpan, new_attrs: dict[str, Any]) -> None:
        if not new_attrs:
            return
        # `on_end` receives a `ReadableSpan` (no `set_attribute` method), and
        # the underlying span is already ended so `set_attribute` would be a
        # silent no-op anyway. Write to the shared `_attributes` dict directly.
        target = getattr(span, "_attributes", None)
        if target is None:
            return
        for k, v in new_attrs.items():
            try:
                target[k] = v
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    @staticmethod
    def _translate_openinference(span: ReadableSpan) -> None:
        """Translate openinference-flattened LLM attributes into Laminar / OTel
        GenAI conventions.

        Handles the groq / google_genai path: Langfuse's docs route those
        through openinference instrumentations, whose attribute layout is a
        flat set of indexed keys rather than Langfuse's `langfuse.*` JSON
        blobs. We reassemble the indexed messages/tools, convert tokens, model
        name, and mark the span as LLM so it renders correctly.
        """
        attrs = dict(span.attributes or {})
        if not attrs:
            return
        new_attrs: dict[str, Any] = {}

        kind = attrs.get(_OI_SPAN_KIND)
        is_llm = (isinstance(kind, str) and kind.upper() in _OI_LLM_SPAN_KINDS) or any(
            isinstance(k, str)
            and (
                k.startswith(_OI_LLM_INPUT_MESSAGES + ".")
                or k.startswith(_OI_LLM_OUTPUT_MESSAGES + ".")
            )
            for k in attrs.keys()
        )
        if is_llm:
            new_attrs[SPAN_TYPE] = "LLM"
        elif isinstance(kind, str) and kind.upper() == "TOOL":
            new_attrs[SPAN_TYPE] = "TOOL"

        # Model
        model = attrs.get(_OI_LLM_MODEL_NAME)
        if isinstance(model, str) and model:
            new_attrs.setdefault(_GEN_AI_REQUEST_MODEL, model)
            new_attrs.setdefault(_GEN_AI_RESPONSE_MODEL, model)

        # Tokens
        prompt_tokens = attrs.get(_OI_TOKEN_PROMPT)
        completion_tokens = attrs.get(_OI_TOKEN_COMPLETION)
        total_tokens = attrs.get(_OI_TOKEN_TOTAL)
        if isinstance(prompt_tokens, (int, float)):
            new_attrs[_GEN_AI_USAGE_INPUT_TOKENS] = int(prompt_tokens)
        if isinstance(completion_tokens, (int, float)):
            new_attrs[_GEN_AI_USAGE_OUTPUT_TOKENS] = int(completion_tokens)
        if isinstance(total_tokens, (int, float)):
            new_attrs[_GEN_AI_USAGE_TOTAL_TOKENS] = int(total_tokens)
        elif isinstance(prompt_tokens, (int, float)) or isinstance(
            completion_tokens, (int, float)
        ):
            new_attrs[_GEN_AI_USAGE_TOTAL_TOKENS] = int(
                (prompt_tokens or 0) + (completion_tokens or 0)
            )

        # Messages
        input_messages = [
            _oi_message_to_genai(m)
            for m in _oi_collect_indexed(attrs, _OI_LLM_INPUT_MESSAGES)
        ]
        if input_messages:
            new_attrs[_GEN_AI_INPUT_MESSAGES] = json_dumps(input_messages)
        output_messages = [
            _oi_message_to_genai(m)
            for m in _oi_collect_indexed(attrs, _OI_LLM_OUTPUT_MESSAGES)
        ]
        if output_messages:
            new_attrs[_GEN_AI_OUTPUT_MESSAGES] = json_dumps(output_messages)

        # Tool definitions
        tools = [
            _oi_tool_to_genai(t) for t in _oi_collect_indexed(attrs, _OI_LLM_TOOLS)
        ]
        tools = [t for t in tools if t]
        if tools:
            new_attrs[_GEN_AI_TOOL_DEFINITIONS] = json_dumps(tools)

        # Fall back to opaque input/output blobs only when the structured
        # messages weren't available. openinference stamps `input.value` /
        # `output.value`; LiteLLM's hybrid spans additionally carry the
        # `langfuse.observation.input/output` blobs, so try those too.
        if not input_messages:
            in_val = attrs.get(_OI_INPUT_VALUE)
            if not (isinstance(in_val, str) and in_val):
                in_val = attrs.get(_OBSERVATION_INPUT)
            if isinstance(in_val, str) and in_val and SPAN_INPUT not in attrs:
                new_attrs[SPAN_INPUT] = in_val
        if not output_messages:
            out_val = attrs.get(_OI_OUTPUT_VALUE)
            if not (isinstance(out_val, str) and out_val):
                out_val = attrs.get(_OBSERVATION_OUTPUT)
            if isinstance(out_val, str) and out_val and SPAN_OUTPUT not in attrs:
                new_attrs[SPAN_OUTPUT] = out_val

        # LiteLLM's `langfuse_otel` hybrid spans also carry trace-level
        # `langfuse.*` session/user/metadata; promote those too. For the pure
        # openinference (groq / google_genai) case these keys are absent, so
        # this is a no-op.
        LangfuseAttributeTranslator._promote_trace_attributes(attrs, new_attrs)

        LangfuseAttributeTranslator._write_attrs(span, new_attrs)


class LangfuseInstrumentor:
    """Attaches Laminar's span processor to every Langfuse `TracerProvider`.

    Unlike the other Laminar instrumentors, this one does NOT extend
    `BaseInstrumentor` — `BaseInstrumentor.instrument()` ignores extra keyword
    arguments that aren't `tracer_provider`/`logger_provider`, and we need the
    caller-supplied Laminar `SpanProcessor` to attach. `init_instrumentations`
    has a special-case branch for this class.
    """

    # State is class-level so repeated calls to `Laminar.connect_to_langfuse()`
    # (or repeated auto-install via `init_instrumentations`) don't re-attach
    # processors or re-wrap `LangfuseResourceManager._initialize_instance`.
    _installed: bool = False
    _handled_providers: set[int] = set()
    #: Providers we attached the translator / span processor to, keyed by id().
    #: `uninstrument` walks this map to detach what we added. The reference is
    #: only held for the duration of `instrument(...)`; `uninstrument` clears
    #: it immediately so the instrumentor never pins a provider long-term.
    _attached_providers: "dict[int, Any]" = {}
    _lmnr_tracer_provider: Any = None
    _original_initialize_instance: Callable[..., Any] | None = None
    #: Saved reference to LiteLLM's
    #: `litellm_logging._init_custom_logger_compatible_class` factory so the
    #: monkey-patch installed for late-constructed `langfuse_otel` loggers can
    #: be reverted on `uninstrument`. See `_patch_litellm_logger_factory`.
    _original_litellm_init_logger: Callable[..., Any] | None = None
    _translator: LangfuseAttributeTranslator | None = None
    _lmnr_span_processor: SpanProcessor | None = None

    def __init__(self) -> None:
        pass

    def instrumentation_dependencies(self) -> Collection[str]:
        return ("langfuse >= 3.0.0",)

    def instrument(
        self,
        lmnr_tracer_provider: SdkTracerProvider,
        lmnr_span_processor: SpanProcessor,
    ) -> None:
        cls = type(self)
        if cls._installed:
            return

        # 1. Translator lives on Laminar's own provider so it sees every
        #    Langfuse span that reaches the Laminar exporter (including spans
        #    that arrived via Laminar's own provider being shared with
        #    Langfuse). Prepend it so it mutates `langfuse.*` attrs before
        #    `LaminarSpanProcessor` exports them — critical for the
        #    `disable_batch=True` / SimpleSpanProcessor path, where export
        #    happens synchronously inside `on_end`.
        translator = LangfuseAttributeTranslator()
        try:
            _prepend_span_processor(lmnr_tracer_provider, translator)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to install Langfuse attribute translator: %s", exc)
            return
        cls._translator = translator
        cls._lmnr_span_processor = lmnr_span_processor
        cls._lmnr_tracer_provider = lmnr_tracer_provider

        # Pre-register Laminar's provider id so `_attach_to_provider` short-
        # circuits if Langfuse happens to share it. The `TracerWrapper.instance`
        # fallback check is racy during auto-install — `init_instrumentations`
        # runs BEFORE `TracerWrapper.instance` is assigned, so
        # `verify_initialized()` returns False, and a Langfuse client that
        # had already been constructed against a pre-existing global provider
        # identical to Laminar's would otherwise get the translator +
        # laminar span processor attached a second time. id()-based
        # short-circuit is independent of TracerWrapper lifecycle.
        cls._handled_providers.add(id(lmnr_tracer_provider))

        # 2 & 3. Attach to already-initialized Langfuse clients and patch
        # future-client construction. If either raises an uncaught exception
        # before we flip `_installed=True`, the translator we just prepended
        # to Laminar's provider would be left orphaned — a subsequent
        # `instrument()` (e.g. via `Laminar.connect_to_langfuse()`) would pass
        # the `_installed` guard and prepend a SECOND translator, causing
        # every Langfuse span to be translated twice. Roll back by walking
        # the half-applied state through `uninstrument()`.
        cls._installed = True
        try:
            # For every already-initialized Langfuse client, attach our span
            # processor and our translator to its `TracerProvider`. If
            # Langfuse reused Laminar's provider, both are already attached —
            # the `_handled_providers` guard makes this a no-op.
            self._attach_to_existing_langfuse_providers()
            # Patch future Langfuse-client construction.
            self._patch_resource_manager()
            # LiteLLM's `langfuse_otel` success callback never registers with
            # `LangfuseResourceManager`, so the two steps above can't reach it.
            # Attach to its private TracerProvider separately (existing loggers
            # + a factory patch for ones constructed later).
            self._attach_to_existing_litellm_loggers()
            self._patch_litellm_logger_factory()
        except Exception:  # pylint: disable=broad-exception-caught
            # Best-effort cleanup: detach whatever we've attached so far and
            # clear class-level state. `uninstrument()` is idempotent and
            # tolerant of partial state (it only touches providers we
            # recorded in `_attached_providers`).
            self.uninstrument()
            raise

    def uninstrument(self) -> None:
        """Reverse `instrument`: detach the translator and Laminar span
        processor from every provider we attached them to, restore the
        resource-manager patch, and clear class-level state so a subsequent
        `instrument()` starts from a clean slate.

        Without the full reset, a re-install would prepend a second
        translator onto Laminar's provider (the first was never removed) and
        `_handled_providers` would still contain stale ids from the previous
        session so `_attach_to_existing_langfuse_providers` would skip
        already-seen Langfuse providers instead of re-attaching.
        """
        cls = type(self)
        if not cls._installed:
            return
        self._unpatch_resource_manager()
        self._unpatch_litellm_logger_factory()

        translator = cls._translator
        lmnr_processor = cls._lmnr_span_processor
        lmnr_provider = cls._lmnr_tracer_provider

        # Detach translator from Laminar's provider.
        if lmnr_provider is not None and translator is not None:
            _remove_span_processor(lmnr_provider, translator)

        # Detach translator + laminar span processor from every Langfuse
        # provider we attached them to.
        for provider in list(cls._attached_providers.values()):
            if translator is not None:
                _remove_span_processor(provider, translator)
            if lmnr_processor is not None:
                _remove_span_processor(provider, lmnr_processor)

        cls._attached_providers = {}
        cls._handled_providers = set()
        cls._translator = None
        cls._lmnr_span_processor = None
        cls._lmnr_tracer_provider = None
        cls._installed = False

    # --- internals ---

    def _attach_to_existing_langfuse_providers(self) -> None:
        try:
            from langfuse._client.resource_manager import (  # type: ignore[import-not-found]
                LangfuseResourceManager,
            )
        except Exception:
            # ImportError if not installed. Other exceptions (e.g. pydantic
            # v1 ConfigError on Python 3.14 due to langfuse's own pydantic
            # compat bug) mean the SDK is unusable in this interpreter —
            # treat the same as absent and leave the bridge installed but
            # inert for the resource-manager path.
            return

        instances = getattr(LangfuseResourceManager, "_instances", {}) or {}
        for rm in instances.values():
            provider = getattr(rm, "tracer_provider", None)
            self._attach_to_provider(provider)

    def _attach_to_provider(self, provider: Any) -> None:
        if provider is None:
            return
        pid = id(provider)
        if pid in self._handled_providers:
            return
        self._handled_providers.add(pid)

        # Skip the Laminar provider itself — our processor and translator are
        # already attached there.
        from lmnr.opentelemetry_lib.tracing import TracerWrapper

        if (
            TracerWrapper.verify_initialized()
            and TracerWrapper.instance._tracer_provider is provider
        ):
            return

        try:
            # Prepend the translator so it runs before any existing exporter
            # attached by Langfuse (its own OTLP BatchSpanProcessor). For the
            # Laminar processor, plain append is fine — it's the exporter; as
            # long as the translator runs first, export order among exporters
            # doesn't matter.
            #
            # Record the provider in `_attached_providers` as soon as the
            # FIRST processor lands — not after both. If attaching the
            # Laminar span processor raises after the translator was already
            # prepended, an end-of-block record would be skipped and
            # `uninstrument` could never detach the orphaned translator,
            # letting a reinstall stack a second one. `_remove_span_processor`
            # tolerates a processor that was never attached, so recording
            # eagerly is safe.
            if self._translator is not None:
                _prepend_span_processor(provider, self._translator)
                type(self)._attached_providers[pid] = provider
            if self._lmnr_span_processor is not None:
                add = getattr(provider, "add_span_processor", None)
                if callable(add):
                    add(self._lmnr_span_processor)
                    type(self)._attached_providers[pid] = provider
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Failed to attach Laminar processor to Langfuse TracerProvider: %s",
                exc,
            )
            # Roll back the partial attach and un-mark the provider so a later
            # attempt (e.g. the resource-manager re-init hook) can retry. Left
            # as-is, `pid` would stay in `_handled_providers` and every future
            # `_attach_to_provider` call would short-circuit, so dual-export to
            # Laminar would never run for this provider.
            if self._translator is not None:
                _remove_span_processor(provider, self._translator)
            if self._lmnr_span_processor is not None:
                _remove_span_processor(provider, self._lmnr_span_processor)
            self._handled_providers.discard(pid)
            type(self)._attached_providers.pop(pid, None)

    def _patch_resource_manager(self) -> None:
        try:
            from langfuse._client.resource_manager import (  # type: ignore[import-not-found]
                LangfuseResourceManager,
            )
        except Exception:
            # See `_attach_to_existing_langfuse_providers` — any import-time
            # failure means Langfuse isn't usable in this interpreter.
            return

        cls = type(self)
        if cls._original_initialize_instance is not None:
            return

        original = LangfuseResourceManager._initialize_instance
        cls._original_initialize_instance = original
        instrumentor = self

        def patched(self_rm, *args, **kwargs):  # type: ignore[no-untyped-def]
            result = original(self_rm, *args, **kwargs)
            try:
                instrumentor._attach_to_provider(
                    getattr(self_rm, "tracer_provider", None)
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Langfuse post-init attach failed: %s", exc)
            return result

        LangfuseResourceManager._initialize_instance = patched

    def _unpatch_resource_manager(self) -> None:
        cls = type(self)
        if cls._original_initialize_instance is None:
            return
        try:
            from langfuse._client.resource_manager import (  # type: ignore[import-not-found]
                LangfuseResourceManager,
            )

            LangfuseResourceManager._initialize_instance = (
                cls._original_initialize_instance
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # If langfuse can't be imported here it isn't usable in this
            # interpreter, so the patched hook can never be invoked anyway.
            # Clear the bookkeeping regardless so `uninstrument` leaves no
            # half-reset state (a retained `_original_initialize_instance`
            # would make a later `_patch_resource_manager` short-circuit).
            logger.debug(
                "Could not restore Langfuse _initialize_instance patch: %s", exc
            )
        finally:
            cls._original_initialize_instance = None

    # --- LiteLLM `langfuse_otel` bridge ---
    #
    # LiteLLM ships a `langfuse_otel` success callback
    # (`litellm.integrations.langfuse.langfuse_otel.LangfuseOtelLogger`) that
    # subclasses LiteLLM's base `OpenTelemetry` integration with
    # `skip_set_global=True`. As a result it builds its OWN private
    # `TracerProvider` (a stock `opentelemetry.sdk.trace.TracerProvider`,
    # stored on `logger._tracer_provider`) and exports OTLP straight to
    # Langfuse Cloud — it never registers with `LangfuseResourceManager`, so
    # the resource-manager attach/patch path above can't see it.
    #
    # The spans it emits carry a hybrid of `langfuse.*` attrs (via
    # `_set_langfuse_specific_attributes`) AND openinference `llm.*` indexed
    # attrs (via `litellm.integrations.arize._utils.set_attributes`), both of
    # which `LangfuseAttributeTranslator` already knows how to translate. So
    # the only missing piece is provider attachment: prepend our translator and
    # append Laminar's span processor onto that private provider, exactly like
    # we do for a Langfuse-owned provider. The translator only ADDS Laminar /
    # GenAI attributes, so LiteLLM's own export to Langfuse Cloud is unchanged.

    def _attach_to_existing_litellm_loggers(self) -> None:
        for logger_obj in self._iter_litellm_langfuse_otel_loggers():
            provider = getattr(logger_obj, "_tracer_provider", None)
            self._attach_to_provider(provider)

    @staticmethod
    def _iter_litellm_langfuse_otel_loggers() -> list[Any]:
        """Return every constructed LiteLLM `langfuse_otel` logger instance.

        LiteLLM keeps callback singletons in
        `litellm.litellm_core_utils.litellm_logging._in_memory_loggers`. We
        filter that list for `LangfuseOtelLogger` instances. Any import/attr
        failure (LiteLLM absent, internal layout changed) yields an empty
        list so the bridge stays inert rather than raising.
        """
        try:
            from litellm.integrations.langfuse.langfuse_otel import (  # type: ignore[import-not-found]
                LangfuseOtelLogger,
            )
            from litellm.litellm_core_utils import (  # type: ignore[import-not-found]
                litellm_logging,
            )
        except Exception:
            return []
        loggers = getattr(litellm_logging, "_in_memory_loggers", None) or []
        return [lg for lg in loggers if isinstance(lg, LangfuseOtelLogger)]

    def _patch_litellm_logger_factory(self) -> None:
        """Wrap LiteLLM's `_init_custom_logger_compatible_class` so any
        `langfuse_otel` logger constructed AFTER the bridge installs also gets
        its private provider dual-attached.

        LiteLLM constructs the callback lazily — the first LLM call (or a
        `litellm.success_callback = ["langfuse_otel"]` assignment that triggers
        `litellm.utils._init_custom_callbacks`) is what builds the logger. All
        call sites import the factory freshly from the module each time (see
        `litellm/utils.py`, `litellm/proxy/...`), so a module-level patch is
        observed by every caller. We attach on the way out, reusing the
        idempotent `_attach_to_provider` (its `_handled_providers` guard makes
        repeat calls for the same provider a no-op).
        """
        try:
            from litellm.integrations.langfuse.langfuse_otel import (  # type: ignore[import-not-found]
                LangfuseOtelLogger,
            )
            from litellm.litellm_core_utils import (  # type: ignore[import-not-found]
                litellm_logging,
            )
        except Exception:
            return

        cls = type(self)
        if cls._original_litellm_init_logger is not None:
            return

        original = getattr(
            litellm_logging, "_init_custom_logger_compatible_class", None
        )
        if not callable(original):
            return
        cls._original_litellm_init_logger = original
        instrumentor = self

        def patched(*args, **kwargs):  # type: ignore[no-untyped-def]
            result = original(*args, **kwargs)
            try:
                # Only the `langfuse_otel` callback should be bridged. The
                # factory builds many OTel-based callbacks (arize, otel, …),
                # all of which carry a private `_tracer_provider`; attaching
                # Laminar's translator + exporter to those would ship
                # unrelated spans into Laminar.
                if isinstance(result, LangfuseOtelLogger):
                    provider = getattr(result, "_tracer_provider", None)
                    if provider is not None:
                        instrumentor._attach_to_provider(provider)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("LiteLLM post-init attach failed: %s", exc)
            return result

        litellm_logging._init_custom_logger_compatible_class = patched

    def _unpatch_litellm_logger_factory(self) -> None:
        cls = type(self)
        if cls._original_litellm_init_logger is None:
            return
        try:
            from litellm.litellm_core_utils import (  # type: ignore[import-not-found]
                litellm_logging,
            )

            litellm_logging._init_custom_logger_compatible_class = (
                cls._original_litellm_init_logger
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Could not restore LiteLLM logger factory patch: %s", exc)
        finally:
            cls._original_litellm_init_logger = None

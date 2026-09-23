"""The `SpanProcessor` that mutates `langfuse.*` / openinference attributes
into Laminar / OTel GenAI shape, plus the span-processor-ordering helpers it
(and `provider_attachment.py`) need.
"""

from __future__ import annotations

import json
from threading import Lock
from typing import Any, cast

from opentelemetry.attributes import BoundedAttributes
from opentelemetry.context import Context
from opentelemetry.sdk.trace import (
    ReadableSpan,
    Span,
    SpanProcessor,
    SynchronousMultiSpanProcessor,
)
from opentelemetry.trace import TracerProvider
from opentelemetry.util.types import AttributeValue
from typing_extensions import override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse.attributes import (
    _GEN_AI_INPUT_MESSAGES,  # pyright: ignore[reportPrivateUsage]
    _GEN_AI_OUTPUT_MESSAGES,  # pyright: ignore[reportPrivateUsage]
    _GEN_AI_REQUEST_MODEL,  # pyright: ignore[reportPrivateUsage]
    _GEN_AI_RESPONSE_MODEL,  # pyright: ignore[reportPrivateUsage]
    _GEN_AI_TOOL_DEFINITIONS,  # pyright: ignore[reportPrivateUsage]
    _GEN_AI_USAGE_INPUT_COST,  # pyright: ignore[reportPrivateUsage]
    _GEN_AI_USAGE_INPUT_TOKENS,  # pyright: ignore[reportPrivateUsage]
    _GEN_AI_USAGE_OUTPUT_COST,  # pyright: ignore[reportPrivateUsage]
    _GEN_AI_USAGE_OUTPUT_TOKENS,  # pyright: ignore[reportPrivateUsage]
    _GEN_AI_USAGE_TOTAL_COST,  # pyright: ignore[reportPrivateUsage]
    _GEN_AI_USAGE_TOTAL_TOKENS,  # pyright: ignore[reportPrivateUsage]
    _LLM_OBSERVATION_TYPES,  # pyright: ignore[reportPrivateUsage]
    _OBSERVATION_COST_DETAILS,  # pyright: ignore[reportPrivateUsage]
    _OBSERVATION_INPUT,  # pyright: ignore[reportPrivateUsage]
    _OBSERVATION_METADATA_PREFIX,  # pyright: ignore[reportPrivateUsage]
    _OBSERVATION_MODEL,  # pyright: ignore[reportPrivateUsage]
    _OBSERVATION_OUTPUT,  # pyright: ignore[reportPrivateUsage]
    _OBSERVATION_TYPE,  # pyright: ignore[reportPrivateUsage]
    _OBSERVATION_USAGE_DETAILS,  # pyright: ignore[reportPrivateUsage]
    _OI_INPUT_VALUE,  # pyright: ignore[reportPrivateUsage]
    _OI_LLM_INPUT_MESSAGES,  # pyright: ignore[reportPrivateUsage]
    _OI_LLM_MODEL_NAME,  # pyright: ignore[reportPrivateUsage]
    _OI_LLM_OUTPUT_MESSAGES,  # pyright: ignore[reportPrivateUsage]
    _OI_LLM_SPAN_KINDS,  # pyright: ignore[reportPrivateUsage]
    _OI_LLM_TOOLS,  # pyright: ignore[reportPrivateUsage]
    _OI_OUTPUT_VALUE,  # pyright: ignore[reportPrivateUsage]
    _OI_SPAN_KIND,  # pyright: ignore[reportPrivateUsage]
    _OI_TOKEN_COMPLETION,  # pyright: ignore[reportPrivateUsage]
    _OI_TOKEN_PROMPT,  # pyright: ignore[reportPrivateUsage]
    _OI_TOKEN_TOTAL,  # pyright: ignore[reportPrivateUsage]
    _TOOL_OBSERVATION_TYPES,  # pyright: ignore[reportPrivateUsage]
    _TRACE_INPUT,  # pyright: ignore[reportPrivateUsage]
    _TRACE_METADATA_PREFIX,  # pyright: ignore[reportPrivateUsage]
    _TRACE_OUTPUT,  # pyright: ignore[reportPrivateUsage]
    _TRACE_SESSION_ID,  # pyright: ignore[reportPrivateUsage]
    _TRACE_TAGS,  # pyright: ignore[reportPrivateUsage]
    _TRACE_USER_ID,  # pyright: ignore[reportPrivateUsage]
)
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

from .translate import (
    convert_openai_tool_calls_to_content_parts,
    cost_field,
    genai_input_from_langfuse,
    genai_output_from_langfuse,
    is_langfuse_span,
    is_openinference_span,
    oi_collect_indexed,
    oi_message_to_genai,
    oi_tool_to_genai,
    parse_json,
    split_messages_and_tool_defs_langchain,
    usage_field,
)

logger = get_default_logger(__name__)


def prepend_span_processor(provider: TracerProvider, processor: SpanProcessor) -> bool:
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
    _processor = add(processor)
    active: SynchronousMultiSpanProcessor | None = getattr(provider, "_active_span_processor", None)
    if active is None:
        return True
    lock: Lock | None = getattr(active, "_lock", None)
    current: tuple[SpanProcessor, ...] | None = getattr(active, "_span_processors", None)
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
            "Could not reorder span processors on %r; translator will run " +
            "after the exporter (%s)",
            provider,
            exc,
        )
    return True


def remove_span_processor(provider: TracerProvider, processor: SpanProcessor | None) -> bool:
    """Remove `processor` from `provider`'s active span processor, if present.

    Mirror of `_prepend_span_processor` for the uninstall path: touches the
    private `_span_processors` tuple on `SynchronousMultiSpanProcessor` /
    `ConcurrentMultiSpanProcessor` under its lock. Returns True if the
    processor was removed, False otherwise.
    """
    active: SynchronousMultiSpanProcessor | None = getattr(provider, "_active_span_processor", None)
    if active is None:
        return False
    lock: Lock | None = getattr(active, "_lock", None)
    current: tuple[SpanProcessor, ...] | None = getattr(active, "_span_processors", None)
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


def _write_attrs(span: ReadableSpan, new_attrs: dict[str, AttributeValue]) -> None:
    if not new_attrs:
        return
    # `on_end` receives a `ReadableSpan` (no `set_attribute` method), and
    # the underlying span is already ended so `set_attribute` would be a
    # silent no-op anyway. Write to the shared `_attributes` dict directly.
    target: BoundedAttributes | None = getattr(span, "_attributes", None)
    if target is None:
        return
    # `Span.end()` marks `BoundedAttributes._immutable = True` before
    # `on_end` runs (opentelemetry-sdk >= 1.4x), so `target[k] = v` now
    # raises `TypeError` here. Flip the flag off for the duration of the
    # write — this is the same private flag `end()` sets, so toggling it
    # is safe and is a no-op on older SDKs where the attribute is absent.
    #
    # The flag toggle (not the writes) is guarded by `target._lock` — the
    # same lock `BoundedAttributes.__setitem__` takes internally to
    # serialize `_dict` mutations. We must NOT hold it across the
    # `target[k] = v` calls below: that lock is a plain (non-reentrant)
    # `threading.Lock`, and `__setitem__` acquires it again itself, so
    # holding it here too would deadlock. This only protects the flag
    # flip itself from a concurrent `_write_attrs` call on the same span
    # (possible if a `TracerProvider` were ever built with
    # `ConcurrentMultiSpanProcessor`, which dispatches every processor's
    # `on_end` for a span to a thread pool concurrently — not what this
    # codebase constructs today). It does not make the whole multi-key
    # write atomic with a concurrent reader; that would require locking
    # inside `BoundedAttributes` itself, upstream of this code.
    lock: Lock | None = getattr(target, "_lock", None)

    def _set_immutable(value: bool) -> None:
        if lock is not None:
            with lock:
                target._immutable = value
        else:
            target._immutable = value

    was_immutable: bool = getattr(target, "_immutable", False)
    if was_immutable:
        _set_immutable(False)
    try:
        for k, v in new_attrs.items():
            try:
                target[k] = v
            except Exception as e:
               logger.debug(f"Failed to set attribute to span: {e}")
    finally:
        if was_immutable:
            _set_immutable(True)


def _promote_trace_attributes(
    attrs: dict[str, AttributeValue], new_attrs: dict[str, AttributeValue]
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
        if all(isinstance(tag, float) for tag in tags):
            new_attrs[f"{ASSOCIATION_PROPERTIES}.tags"] = list(cast(list[float] | tuple[float], tags))
        else:
            new_attrs[f"{ASSOCIATION_PROPERTIES}.tags"] = [str(tag) for tag in tags]

    # Trace / observation metadata — flat form is `langfuse.trace.metadata.<k>`,
    # unflattened form is `langfuse.trace.metadata`. Route both into Laminar's
    # `lmnr.association.properties.metadata.<k>` namespace.
    for k, v in attrs.items():
        if not isinstance(k, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            continue
        for prefix in (_TRACE_METADATA_PREFIX, _OBSERVATION_METADATA_PREFIX):
            if k == prefix:
                parsed = parse_json(v)  # pyright: ignore[reportAny]
                if isinstance(parsed, dict):
                    for mk, mv in (cast(dict[str, Any], parsed)).items():  # pyright: ignore[reportAny, reportExplicitAny]
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


def _translate(span: ReadableSpan) -> None:
    attrs = dict(span.attributes or {})
    if not attrs:
        return
    new_attrs: dict[str, AttributeValue] = {}

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
        _existing = new_attrs.setdefault(_GEN_AI_REQUEST_MODEL, model)
        _existing = new_attrs.setdefault(_GEN_AI_RESPONSE_MODEL, model)

    # Usage details (tokens)
    usage_raw = cast(dict[str, float | dict[str, float]], parse_json(attrs.get(_OBSERVATION_USAGE_DETAILS)))
    input_tokens = usage_field(usage_raw, "input", "prompt_tokens", "input_tokens")
    output_tokens = usage_field(
        usage_raw, "output", "completion_tokens", "output_tokens"
    )
    total_tokens = usage_field(usage_raw, "total", "total_tokens")
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
    cost_raw = cast(dict[str, float | dict[str, float]], parse_json(attrs.get(_OBSERVATION_COST_DETAILS)))
    input_cost = cost_field(cost_raw, "input")
    output_cost = cost_field(cost_raw, "output")
    total_cost = cost_field(cost_raw, "total")
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
        split = genai_input_from_langfuse(input_raw)
        if split is not None:
            messages, tools = split  # pyright: ignore[reportAny]
            messages, lc_tools = split_messages_and_tool_defs_langchain(messages)  # pyright: ignore[reportAny]
            messages = convert_openai_tool_calls_to_content_parts(messages, False)  # pyright: ignore[reportAny]
            new_attrs[_GEN_AI_INPUT_MESSAGES] = json_dumps(messages)
            if tools:
                new_attrs[_GEN_AI_TOOL_DEFINITIONS] = json_dumps(tools)  # pyright: ignore[reportAny]
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
        messages = genai_output_from_langfuse(output_raw)
        if messages is not None:
            if isinstance(messages, list):
                messages = convert_openai_tool_calls_to_content_parts(
                    messages, True,  # pyright: ignore[reportUnknownArgumentType]
                )
            new_attrs[_GEN_AI_OUTPUT_MESSAGES] = json_dumps(messages)
            output_handled = True
    if (
        not output_handled
        and isinstance(output_raw, str)
        and output_raw
        and SPAN_OUTPUT not in attrs
    ):
        new_attrs[SPAN_OUTPUT] = output_raw

    _promote_trace_attributes(attrs, new_attrs)
    _write_attrs(span, new_attrs)


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
    new_attrs: dict[str, AttributeValue] = {}

    # litellm's `langfuse_otel` callback runs through arize's attribute
    # setter, which forces `openinference.span.kind=TOOL` on ANY completion
    # that merely passes `tools=[...]` (see litellm
    # `integrations/arize/_utils.py`: `if optional_tools ... span_kind =
    # TOOL`). That's a genuine LLM call, not a tool execution, so we must
    # not trust a `TOOL` kind when the span also carries LLM signals — a
    # model name, token counts, or indexed input/output messages. Otherwise
    # `litellm_request` spans get mis-typed `TOOL` whenever the caller used
    # tool-calling.
    kind = attrs.get(_OI_SPAN_KIND)
    has_llm_signals = (
        _OI_LLM_MODEL_NAME in attrs
        or _OI_TOKEN_PROMPT in attrs
        or _OI_TOKEN_COMPLETION in attrs
        or _OI_TOKEN_TOTAL in attrs
        or any(
            isinstance(k, str)  # pyright: ignore[reportUnnecessaryIsInstance]
            and (
                k.startswith((_OI_LLM_INPUT_MESSAGES + ".", _OI_LLM_OUTPUT_MESSAGES + "."))
            )
            for k in attrs
        )
    )
    is_llm = (
        isinstance(kind, str) and kind.upper() in _OI_LLM_SPAN_KINDS
    ) or has_llm_signals
    if is_llm:
        new_attrs[SPAN_TYPE] = "LLM"
    elif isinstance(kind, str) and kind.upper() == "TOOL":
        new_attrs[SPAN_TYPE] = "TOOL"

    # Model
    model = attrs.get(_OI_LLM_MODEL_NAME)
    if isinstance(model, str) and model:
        _existing = new_attrs.setdefault(_GEN_AI_REQUEST_MODEL, model)
        _existing = new_attrs.setdefault(_GEN_AI_RESPONSE_MODEL, model)

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
            (cast(int | float, prompt_tokens) or 0) + (cast(int | float, completion_tokens) or 0)
        )

    # Messages
    input_messages = [
        oi_message_to_genai(m)
        for m in oi_collect_indexed(attrs, _OI_LLM_INPUT_MESSAGES)
    ]
    if input_messages:
        new_attrs[_GEN_AI_INPUT_MESSAGES] = json_dumps(input_messages)
    output_messages = [
        oi_message_to_genai(m)
        for m in oi_collect_indexed(attrs, _OI_LLM_OUTPUT_MESSAGES)
    ]
    if output_messages:
        new_attrs[_GEN_AI_OUTPUT_MESSAGES] = json_dumps(output_messages)

    # Tool definitions
    tools = [
        oi_tool_to_genai(t) for t in oi_collect_indexed(attrs, _OI_LLM_TOOLS)
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
    _promote_trace_attributes(attrs, new_attrs)

    _write_attrs(span, new_attrs)


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

    @override
    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        return None

    @override
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
            if is_openinference_span(span):
                _translate_openinference(span)
            elif is_langfuse_span(span):
                _translate(span)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Langfuse attribute translation failed: %s", exc)

    @override
    def shutdown(self) -> None:
        return None

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

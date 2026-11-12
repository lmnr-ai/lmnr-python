"""Forwards spans from a Langfuse-owned `TracerProvider` to Laminar without
touching the spans Langfuse exports.

`LangfuseBridgeSpanProcessor` sits on Langfuse's provider next to
`LangfuseSpanProcessor`. Laminar's own processor never sees the live span:
its `on_start` stamps go into a side table, and `on_end` hands it a copy with
those stamps plus the `langfuse.*` -> Laminar translation.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any, TypeAlias, cast

from opentelemetry.context import Context
from opentelemetry.sdk.trace import Event, ReadableSpan, Span, SpanProcessor
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_DEFINITIONS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.types import AttributeValue
from typing_extensions import override

from lmnr.opentelemetry_lib.bridges.base import Bridge
from lmnr.opentelemetry_lib.bridges.langfuse.attributes import (
    GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
    GEN_AI_USAGE_COST,
    GEN_AI_USAGE_INPUT_COST,
    GEN_AI_USAGE_OUTPUT_COST,
    LF_AS_ROOT,
    LF_LLM_OBSERVATION_TYPES,
    LF_METADATA_SESSION_ID,
    LF_METADATA_TAGS,
    LF_METADATA_TRACE_KEYS,
    LF_METADATA_USER_ID,
    LF_OBSERVATION_COST_DETAILS,
    LF_OBSERVATION_INPUT,
    LF_OBSERVATION_LEVEL,
    LF_OBSERVATION_METADATA_PREFIX,
    LF_OBSERVATION_MODEL,
    LF_OBSERVATION_MODEL_PARAMETERS,
    LF_OBSERVATION_OUTPUT,
    LF_OBSERVATION_STATUS_MESSAGE,
    LF_OBSERVATION_TYPE,
    LF_OBSERVATION_USAGE_DETAILS,
    LF_TOOL_OBSERVATION_TYPES,
    LF_TRACE_COMPAT_SESSION_ID,
    LF_TRACE_COMPAT_USER_ID,
    LF_TRACE_INPUT,
    LF_TRACE_METADATA_PREFIX,
    LF_TRACE_OUTPUT,
    LF_TRACE_SESSION_ID,
    LF_TRACE_TAGS,
    LF_TRACE_USER_ID,
    LLM_USAGE_TOTAL_TOKENS,
    LS_PROVIDER_ALIASES,
    OI_INPUT_VALUE,
    OI_LLM_INPUT_MESSAGES,
    OI_LLM_INVOCATION_PARAMETERS,
    OI_LLM_MODEL_NAME,
    OI_LLM_OUTPUT_MESSAGES,
    OI_LLM_PROVIDER,
    OI_LLM_SPAN_KINDS,
    OI_LLM_TOOLS,
    OI_OUTPUT_VALUE,
    OI_SPAN_KIND,
    OI_TOKEN_COMPLETION,
    OI_TOKEN_PROMPT,
    OI_TOKEN_TOTAL,
)
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SESSION_ID,
    SPAN_BRIDGE_NAME,
    SPAN_BRIDGE_VERSION,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_TYPE,
    USER_ID,
)
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import json_dumps

from .translate import (
    genai_input_from_langfuse,
    genai_output_from_langfuse,
    is_langfuse_span,
    is_openinference_span,
    model_parameter_attributes,
    normalize_messages,
    oi_collect_indexed,
    oi_message_to_genai,
    oi_tool_to_genai,
    parse_json,
    split_messages_and_tool_defs_langchain,
    usage_attributes,
)

logger = get_default_logger(__name__)

_USAGE_TOKEN_KEYS = (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    LLM_USAGE_TOTAL_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
)
_USAGE_COST_KEYS = (GEN_AI_USAGE_INPUT_COST, GEN_AI_USAGE_OUTPUT_COST, GEN_AI_USAGE_COST)

_DISABLED = "lmnr.internal.disabled"

# A translation: attributes to set on Laminar's copy of the span. None removes
# the attribute from the copy: a `langfuse.*` / openinference value already
# carried by the Laminar attributes it was converted to. Laminar would
# otherwise store (and bill) every payload twice.
Translated: TypeAlias = dict[str, AttributeValue | None]


def _metadata_value(value: Any, json_encoded: bool) -> AttributeValue:
    if json_encoded and isinstance(value, str):
        value = parse_json(value)
    if isinstance(value, (str, bool, int, float)):
        return value
    return json_dumps(value)


def _first_string(*values: Any) -> str | None:
    """The first non-empty id, like Langfuse's server. A number counts, as
    its text: langfuse 5 JSON-encodes a metadata `12345` as `"12345"`, which
    decodes to an int, and Langfuse's server keeps such a value as text."""
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if isinstance(value, str) and value:
            return value
    return None


def _tags(value: Any) -> list[str] | None:
    """Tags the way Langfuse's server reads them: a list, a JSON list (LiteLLM's
    `langfuse_otel`, the `langfuse_tags` metadata key), a comma-separated
    string, or one tag."""
    parsed = parse_json(value)
    # A JSON list in a JSON-encoded (langfuse 5) metadata value.
    if isinstance(parsed, str) and parsed.strip().startswith("["):
        parsed = parse_json(parsed)
    if isinstance(parsed, (list, tuple)):
        tags = [str(tag) for tag in cast(list[Any], parsed) if tag is not None]
    elif isinstance(value, str):
        text = parsed if isinstance(parsed, str) else value
        tags = [tag.strip() for tag in text.split(",") if tag.strip()]
    else:
        return None
    return list(dict.fromkeys(tags)) or None


def _metadata(
    attrs: dict[str, AttributeValue], prefix: str, json_encoded_metadata: bool
) -> dict[str, AttributeValue]:
    """`prefix` as one JSON object and `prefix.*` keys, the latter winning."""
    metadata: dict[str, AttributeValue] = {}
    parsed = parse_json(attrs.get(prefix))
    if isinstance(parsed, dict):
        for k, v in cast(dict[str, Any], parsed).items():
            metadata[k] = _metadata_value(v, False)
    for k, v in attrs.items():
        if k.startswith(prefix + "."):
            metadata[k[len(prefix) + 1 :]] = _metadata_value(v, json_encoded_metadata)
    return metadata


def _promote_trace_attributes(
    attrs: dict[str, AttributeValue],
    new_attrs: Translated,
    json_encoded_metadata: bool,
    is_root: bool,
) -> None:
    """Map Langfuse's trace session / user / tags / metadata to Laminar
    association properties, from the same attributes Langfuse's server reads
    them from: the `propagate_attributes` / `update_current_trace` keys, the
    older `langfuse.session.id` / `langfuse.user.id`, and the
    `langfuse_session_id` / `langfuse_user_id` / `langfuse_tags` metadata keys
    of the `langfuse.openai` integration.

    Observation metadata becomes trace metadata on the root span only, as in
    Langfuse; elsewhere it stays raw (mostly LangChain's `ls_*` keys).
    """
    # Both metadata forms (one JSON object and `.*` keys), decoded. The
    # `langfuse_*` keys are read from them, so a key left out of Laminar's
    # metadata below always became the session, user or tags.
    trace_metadata = _metadata(attrs, LF_TRACE_METADATA_PREFIX, json_encoded_metadata)
    observation_metadata = _metadata(attrs, LF_OBSERVATION_METADATA_PREFIX, json_encoded_metadata)

    # In Langfuse's server's order of precedence.
    session_id = _first_string(
        attrs.get(LF_TRACE_COMPAT_SESSION_ID),
        attrs.get(LF_TRACE_SESSION_ID),
        observation_metadata.get(LF_METADATA_SESSION_ID),
        trace_metadata.get(LF_METADATA_SESSION_ID),
    )
    if session_id:
        new_attrs[f"{ASSOCIATION_PROPERTIES}.{SESSION_ID}"] = session_id
    user_id = _first_string(
        attrs.get(LF_TRACE_COMPAT_USER_ID),
        attrs.get(LF_TRACE_USER_ID),
        observation_metadata.get(LF_METADATA_USER_ID),
        trace_metadata.get(LF_METADATA_USER_ID),
    )
    if user_id:
        new_attrs[f"{ASSOCIATION_PROPERTIES}.{USER_ID}"] = user_id
    for value in (
        attrs.get(LF_TRACE_TAGS),
        observation_metadata.get(LF_METADATA_TAGS),
        trace_metadata.get(LF_METADATA_TAGS),
    ):
        if tags := _tags(value):
            new_attrs[f"{ASSOCIATION_PROPERTIES}.tags"] = tags
            break

    metadata = {**trace_metadata, **(observation_metadata if is_root else {})}
    for k, v in metadata.items():
        if k not in LF_METADATA_TRACE_KEYS:
            new_attrs[f"{ASSOCIATION_PROPERTIES}.metadata.{k}"] = v


def _convert_model_parameters(
    attrs: dict[str, AttributeValue], key: str, new_attrs: Translated
) -> None:
    mapped, rest = model_parameter_attributes(parse_json(attrs.get(key)))
    new_attrs.update(mapped)
    if rest is not None and mapped:
        # Only the parameters that have no request attribute stay.
        new_attrs[key] = json_dumps(rest) if rest else None


def _translate_langfuse(
    attrs: dict[str, AttributeValue], json_encoded_metadata: bool, is_root: bool
) -> Translated:
    new_attrs: Translated = {}

    is_llm = False
    obs_type = attrs.get(LF_OBSERVATION_TYPE)
    if isinstance(obs_type, str):
        if obs_type.lower() in LF_LLM_OBSERVATION_TYPES:
            new_attrs[SPAN_TYPE] = "LLM"
            is_llm = True
        elif obs_type.lower() in LF_TOOL_OBSERVATION_TYPES:
            new_attrs[SPAN_TYPE] = "TOOL"

    model = attrs.get(LF_OBSERVATION_MODEL)
    if isinstance(model, str) and model:
        new_attrs[GEN_AI_REQUEST_MODEL] = model
        new_attrs[GEN_AI_RESPONSE_MODEL] = model
        new_attrs[LF_OBSERVATION_MODEL] = None

    # Langfuse records a provider only through LangChain's `ls_provider`
    # metadata. `langfuse.openai` records none: the client's base URL may point
    # at any OpenAI-compatible API.
    provider = attrs.get(f"{LF_OBSERVATION_METADATA_PREFIX}.ls_provider")
    if isinstance(provider, str) and json_encoded_metadata:
        provider = parse_json(provider)
    if isinstance(provider, str) and provider:
        new_attrs[GEN_AI_SYSTEM] = LS_PROVIDER_ALIASES.get(provider, provider)
    _convert_model_parameters(attrs, LF_OBSERVATION_MODEL_PARAMETERS, new_attrs)

    usage = usage_attributes(
        parse_json(attrs.get(LF_OBSERVATION_USAGE_DETAILS)),
        parse_json(attrs.get(LF_OBSERVATION_COST_DETAILS)),
        tokens=_USAGE_TOKEN_KEYS,
        costs=_USAGE_COST_KEYS,
    )
    new_attrs.update(usage)
    if any(k in usage for k in _USAGE_TOKEN_KEYS):
        new_attrs[LF_OBSERVATION_USAGE_DETAILS] = None
    if any(k in usage for k in _USAGE_COST_KEYS):
        new_attrs[LF_OBSERVATION_COST_DETAILS] = None

    # LLM input/output becomes GenAI messages so the transcript view renders;
    # anything else (or an unrecognized LLM shape) stays a raw blob. Either way
    # the `langfuse.*` key it came from is dropped.
    input_key = LF_OBSERVATION_INPUT if LF_OBSERVATION_INPUT in attrs else LF_TRACE_INPUT
    input_raw = attrs.get(input_key)
    input_handled = False
    if is_llm and input_raw is not None:
        split = genai_input_from_langfuse(input_raw)
        if split is not None:
            messages, tools = split
            messages, lc_tools = split_messages_and_tool_defs_langchain(messages)
            new_attrs[GEN_AI_INPUT_MESSAGES] = json_dumps(normalize_messages(messages, False))
            if tools:
                new_attrs[GEN_AI_TOOL_DEFINITIONS] = json_dumps(tools)
            elif lc_tools:
                new_attrs[GEN_AI_TOOL_DEFINITIONS] = json_dumps(lc_tools)
            input_handled = True
    if not input_handled and isinstance(input_raw, str) and input_raw and SPAN_INPUT not in attrs:
        new_attrs[SPAN_INPUT] = input_raw
        input_handled = True
    if input_handled:
        new_attrs[input_key] = None

    output_key = LF_OBSERVATION_OUTPUT if LF_OBSERVATION_OUTPUT in attrs else LF_TRACE_OUTPUT
    output_raw = attrs.get(output_key)
    output_handled = False
    if is_llm and output_raw is not None:
        messages = genai_output_from_langfuse(output_raw)
        if messages is not None:
            new_attrs[GEN_AI_OUTPUT_MESSAGES] = json_dumps(normalize_messages(messages, True))
            output_handled = True
    if (
        not output_handled
        and isinstance(output_raw, str)
        and output_raw
        and SPAN_OUTPUT not in attrs
    ):
        new_attrs[SPAN_OUTPUT] = output_raw
        output_handled = True
    if output_handled:
        new_attrs[output_key] = None

    _promote_trace_attributes(attrs, new_attrs, json_encoded_metadata, is_root)
    return new_attrs


def _translate_openinference(
    attrs: dict[str, AttributeValue], json_encoded_metadata: bool, is_root: bool
) -> Translated:
    """Translate openinference's flat indexed `llm.*` attributes (groq /
    google_genai instrumentations, LiteLLM's `langfuse_otel` callback)."""
    new_attrs: Translated = {}

    # LiteLLM's `langfuse_otel` callback forces `openinference.span.kind=TOOL`
    # on any completion that passes `tools=[...]`, so a TOOL kind is only
    # trusted when the span carries no LLM signal.
    kind = attrs.get(OI_SPAN_KIND)
    has_llm_signals = (
        OI_LLM_MODEL_NAME in attrs
        or OI_TOKEN_PROMPT in attrs
        or OI_TOKEN_COMPLETION in attrs
        or OI_TOKEN_TOTAL in attrs
        or any(
            k.startswith((OI_LLM_INPUT_MESSAGES + ".", OI_LLM_OUTPUT_MESSAGES + ".")) for k in attrs
        )
    )
    is_llm = (isinstance(kind, str) and kind.upper() in OI_LLM_SPAN_KINDS) or has_llm_signals
    if is_llm:
        new_attrs[SPAN_TYPE] = "LLM"
    elif isinstance(kind, str) and kind.upper() == "TOOL":
        new_attrs[SPAN_TYPE] = "TOOL"

    model = attrs.get(OI_LLM_MODEL_NAME)
    if isinstance(model, str) and model:
        new_attrs[GEN_AI_REQUEST_MODEL] = model
        new_attrs[GEN_AI_RESPONSE_MODEL] = model
        new_attrs[OI_LLM_MODEL_NAME] = None

    # LiteLLM writes its `custom_llm_provider`, "Unknown" when it has none.
    provider = attrs.get(OI_LLM_PROVIDER)
    if isinstance(provider, str) and provider and provider != "Unknown":
        new_attrs[GEN_AI_SYSTEM] = provider
    _convert_model_parameters(attrs, OI_LLM_INVOCATION_PARAMETERS, new_attrs)

    prompt_tokens = attrs.get(OI_TOKEN_PROMPT)
    completion_tokens = attrs.get(OI_TOKEN_COMPLETION)
    total_tokens = attrs.get(OI_TOKEN_TOTAL)
    if isinstance(prompt_tokens, (int, float)):
        new_attrs[GEN_AI_USAGE_INPUT_TOKENS] = int(prompt_tokens)
        new_attrs[OI_TOKEN_PROMPT] = None
    if isinstance(completion_tokens, (int, float)):
        new_attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = int(completion_tokens)
        new_attrs[OI_TOKEN_COMPLETION] = None
    if isinstance(total_tokens, (int, float)):
        new_attrs[LLM_USAGE_TOTAL_TOKENS] = int(total_tokens)
        new_attrs[OI_TOKEN_TOTAL] = None
    elif isinstance(prompt_tokens, (int, float)) or isinstance(completion_tokens, (int, float)):
        new_attrs[LLM_USAGE_TOTAL_TOKENS] = int(
            (cast(int | float, prompt_tokens) or 0) + (cast(int | float, completion_tokens) or 0)
        )

    def drop_family(prefix: str) -> None:
        for k in attrs:
            if k.startswith(prefix + "."):
                new_attrs[k] = None

    input_messages = [
        oi_message_to_genai(m) for m in oi_collect_indexed(attrs, OI_LLM_INPUT_MESSAGES)
    ]
    if input_messages:
        new_attrs[GEN_AI_INPUT_MESSAGES] = json_dumps(normalize_messages(input_messages, False))
        drop_family(OI_LLM_INPUT_MESSAGES)
    output_messages = [
        oi_message_to_genai(m) for m in oi_collect_indexed(attrs, OI_LLM_OUTPUT_MESSAGES)
    ]
    if output_messages:
        new_attrs[GEN_AI_OUTPUT_MESSAGES] = json_dumps(normalize_messages(output_messages, True))
        drop_family(OI_LLM_OUTPUT_MESSAGES)

    tools = [oi_tool_to_genai(t) for t in oi_collect_indexed(attrs, OI_LLM_TOOLS)]
    tools = [t for t in tools if t]
    if tools:
        new_attrs[GEN_AI_TOOL_DEFINITIONS] = json_dumps(tools)
        drop_family(OI_LLM_TOOLS)

    # LiteLLM's hybrid spans also carry `input.value` and
    # `langfuse.observation.input/output`: the same payload again. Once one of
    # them is converted (to messages, or copied to `lmnr.span.input/output`),
    # both are dropped.
    if not input_messages:
        in_val = attrs.get(OI_INPUT_VALUE)
        if not (isinstance(in_val, str) and in_val):
            in_val = attrs.get(LF_OBSERVATION_INPUT)
        if isinstance(in_val, str) and in_val and SPAN_INPUT not in attrs:
            new_attrs[SPAN_INPUT] = in_val
    if input_messages or SPAN_INPUT in new_attrs:
        new_attrs[OI_INPUT_VALUE] = None
        new_attrs[LF_OBSERVATION_INPUT] = None
    if not output_messages:
        out_val = attrs.get(OI_OUTPUT_VALUE)
        if not (isinstance(out_val, str) and out_val):
            out_val = attrs.get(LF_OBSERVATION_OUTPUT)
        if isinstance(out_val, str) and out_val and SPAN_OUTPUT not in attrs:
            new_attrs[SPAN_OUTPUT] = out_val
    if output_messages or SPAN_OUTPUT in new_attrs:
        new_attrs[OI_OUTPUT_VALUE] = None
        new_attrs[LF_OBSERVATION_OUTPUT] = None

    # Keys the span never had need no removal.
    for k in [k for k, v in new_attrs.items() if v is None and k not in attrs]:
        del new_attrs[k]
    _promote_trace_attributes(attrs, new_attrs, json_encoded_metadata, is_root)
    return new_attrs


def translate_span(span: ReadableSpan, json_encoded_metadata: bool = False) -> Translated:
    """Laminar attributes to add on top of `span`'s own, and (as None) the
    ones to remove from Laminar's copy. Pure: never touches `span`.

    openinference is checked first: LiteLLM's hybrid spans carry both layouts
    but no `langfuse.observation.type`, and their model / tokens / messages
    live in the `llm.*` keys. Real Langfuse SDK spans never carry those.
    """
    attrs = dict(span.attributes or {})
    if not attrs:
        return {}
    # Langfuse's server treats a span as the trace root (whose observation
    # metadata is also trace metadata) when it has no parent or is forced to.
    is_root = span.parent is None or str(attrs.get(LF_AS_ROOT)).lower() == "true"
    try:
        if is_openinference_span(span):
            return _translate_openinference(attrs, json_encoded_metadata, is_root)
        if is_langfuse_span(span):
            return _translate_langfuse(attrs, json_encoded_metadata, is_root)
    except Exception:
        logger.debug("Langfuse attribute translation failed", exc_info=True)
    return {}


def error_event(span: ReadableSpan) -> Event | None:
    """An `exception` event for a failed span that has none.

    Laminar marks a span failed only through an `exception` event, and
    Langfuse never records one: a failure is an OTel ERROR status or
    `langfuse.observation.level=ERROR` with a status message.
    """
    if any(event.name == "exception" for event in span.events):
        return None
    attrs = span.attributes or {}
    status = span.status
    failed = status.status_code == StatusCode.ERROR or attrs.get(LF_OBSERVATION_LEVEL) == "ERROR"
    if not failed:
        return None
    message = attrs.get(LF_OBSERVATION_STATUS_MESSAGE) or status.description or ""
    return Event(
        name="exception",
        attributes={"exception.type": "Error", "exception.message": str(message)},
        timestamp=span.end_time,
    )


class _ShadowSpan:
    """Stands in for a live span in Laminar's `on_start`.

    Reads go to the real span with the shadow's own writes on top; writes stay
    in `added`, so the real span (which Langfuse exports) is never touched.
    """

    def __init__(self, span: Span) -> None:
        self._span: Span = span
        self.added: dict[str, AttributeValue] = {}

    @property
    def attributes(self) -> dict[str, AttributeValue]:
        return {**(self._span.attributes or {}), **self.added}

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self.added[key] = value

    def set_attributes(self, attributes: dict[str, AttributeValue]) -> None:
        self.added.update(attributes)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._span, name)


class LangfuseBridgeSpanProcessor(SpanProcessor):
    """Feeds Laminar's span processor a translated copy of every span that
    ends on a Langfuse-owned provider and that Langfuse itself exports.

    One instance per provider, all wrapping the same Laminar processor, so
    Laminar's span-path bookkeeping sees every bridged span.
    """

    def __init__(
        self,
        lmnr_span_processor: SpanProcessor,
        bridge_version: str | None,
        should_forward: Callable[[ReadableSpan], bool] | None = None,
        json_encoded_metadata: bool = False,
    ) -> None:
        self.lmnr_span_processor: SpanProcessor = lmnr_span_processor
        self._bridge_version: str | None = bridge_version
        self._should_forward: Callable[[ReadableSpan], bool] | None = should_forward
        self._json_encoded_metadata: bool = json_encoded_metadata
        self._lock: Lock = Lock()
        self._added: dict[int, dict[str, AttributeValue]] = {}

    @override
    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        try:
            shadow = _ShadowSpan(span)
            self.lmnr_span_processor.on_start(cast(Span, cast(object, shadow)), parent_context)
            span_context = span.get_span_context()
            with self._lock:
                self._added[span_context.span_id] = shadow.added
        except Exception:
            logger.debug("Langfuse bridge failed to start a span", exc_info=True)

    @override
    def on_end(self, span: ReadableSpan) -> None:
        try:
            with self._lock:
                added = self._added.pop(span.context.span_id, None) if span.context else None
            if added is None:
                return
            forward = True
            if self._should_forward is not None:
                try:
                    forward = self._should_forward(span)
                except Exception:
                    forward = False
            attributes: dict[str, AttributeValue] = {**(span.attributes or {}), **added}
            if forward:
                for key, value in translate_span(span, self._json_encoded_metadata).items():
                    if value is None:
                        _ = attributes.pop(key, None)
                    else:
                        attributes[key] = value
                attributes[SPAN_BRIDGE_NAME] = Bridge.LANGFUSE.value
                if self._bridge_version:
                    attributes[SPAN_BRIDGE_VERSION] = self._bridge_version
            else:
                # Still end it in Laminar's processor so its per-span
                # bookkeeping is released; disabled spans aren't exported.
                attributes[_DISABLED] = True
            events = list(span.events)
            if forward and (event := error_event(span)) is not None:
                events.append(event)
            self.lmnr_span_processor.on_end(
                ReadableSpan(
                    name=span.name,
                    context=span.context,
                    parent=span.parent,
                    resource=span.resource,
                    attributes=attributes,
                    events=events,
                    links=span.links,
                    kind=span.kind,
                    status=span.status,
                    start_time=span.start_time,
                    end_time=span.end_time,
                    instrumentation_scope=span.instrumentation_scope,
                )
            )
        except Exception:
            logger.debug("Langfuse bridge failed to forward a span", exc_info=True)

    @override
    def shutdown(self) -> None:
        return None

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        # Copies are handed to Laminar's processor in `on_end`, where they wait
        # in its batch. Without this, `langfuse.flush()` would leave them there.
        try:
            return self.lmnr_span_processor.force_flush(timeout_millis)
        except Exception:
            logger.debug("Langfuse bridge failed to flush", exc_info=True)
            return False


def langfuse_would_export(provider: object, span: ReadableSpan) -> bool:
    """Whether any `LangfuseSpanProcessor` on `provider` exports `span`.

    Mirrors `LangfuseSpanProcessor.on_end` in langfuse 3.x-5.x: the project
    check, `blocked_instrumentation_scopes` (3.x, 4.x) and
    `should_export_span` (4.x, 5.x). Those are private, so each is probed and
    skipped when absent.
    """
    try:
        from langfuse._client.span_processor import LangfuseSpanProcessor
    except Exception:
        return False
    active = getattr(provider, "_active_span_processor", None)
    processors = cast(tuple[SpanProcessor, ...], getattr(active, "_span_processors", ()))
    scope = span.instrumentation_scope
    is_langfuse_scope = scope is not None and scope.name == "langfuse-sdk"
    for processor in processors:
        if not isinstance(processor, LangfuseSpanProcessor):
            continue
        p = cast(Any, processor)
        project_check = getattr(p, "_is_langfuse_project_span", None)
        if is_langfuse_scope and callable(project_check) and not project_check(span):
            continue
        blocked = getattr(p, "_is_blocked_instrumentation_scope", None)
        if callable(blocked) and blocked(span):
            continue
        should_export = getattr(p, "_should_export_span", None)
        if callable(should_export) and not should_export(span):
            continue
        return True
    return False


__all__ = [
    "LangfuseBridgeSpanProcessor",
    "error_event",
    "langfuse_would_export",
    "translate_span",
]

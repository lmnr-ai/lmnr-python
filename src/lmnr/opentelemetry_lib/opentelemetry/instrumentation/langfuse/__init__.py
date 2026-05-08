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
   - Only attributes from Langfuse-scoped spans are mutated (detected by
     instrumentation_scope.name == "langfuse-sdk").
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


def _parse_json(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


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
        new_order = (processor,) + tuple(
            p for p in current if p is not processor
        )
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
            "Could not remove span processor from %r (%s)", provider, exc,
        )
        return False
    return True


def _is_langfuse_span(span: ReadableSpan) -> bool:
    scope = getattr(span, "instrumentation_scope", None)
    if scope is None:
        return False
    if scope.name == LANGFUSE_TRACER_NAME:
        return True
    attrs = span.attributes or {}
    return any(
        isinstance(k, str) and k.startswith("langfuse.") for k in attrs.keys()
    )


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

    def on_start(self, span: Span, parent_context: Context |
                 None = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        if not _is_langfuse_span(span):
            return
        try:
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

        obs_type = attrs.get(_OBSERVATION_TYPE)
        if isinstance(obs_type, str):
            if obs_type.lower() in _LLM_OBSERVATION_TYPES:
                new_attrs[SPAN_TYPE] = "LLM"
            elif obs_type.lower() in _TOOL_OBSERVATION_TYPES:
                new_attrs[SPAN_TYPE] = "TOOL"

        # Model
        model = attrs.get(_OBSERVATION_MODEL)
        if isinstance(model, str) and model:
            new_attrs.setdefault(_GEN_AI_REQUEST_MODEL, model)
            new_attrs.setdefault(_GEN_AI_RESPONSE_MODEL, model)

        # Usage details (tokens)
        usage_raw = _parse_json(attrs.get(_OBSERVATION_USAGE_DETAILS))
        input_tokens = _usage_field(
            usage_raw,
            "input",
            "prompt_tokens",
            "input_tokens")
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
        if total_cost is None and (
            input_cost is not None or output_cost is not None
        ):
            total_cost = (input_cost or 0.0) + (output_cost or 0.0)
        if total_cost is not None:
            new_attrs[_GEN_AI_USAGE_TOTAL_COST] = total_cost

        # Input / output — prefer observation-level, fall back to trace-level.
        input_raw = attrs.get(_OBSERVATION_INPUT)
        if input_raw is None:
            input_raw = attrs.get(_TRACE_INPUT)
        if isinstance(input_raw,
                      str) and input_raw and SPAN_INPUT not in attrs:
            new_attrs[SPAN_INPUT] = input_raw

        output_raw = attrs.get(_OBSERVATION_OUTPUT)
        if output_raw is None:
            output_raw = attrs.get(_TRACE_OUTPUT)
        if isinstance(output_raw,
                      str) and output_raw and SPAN_OUTPUT not in attrs:
            new_attrs[SPAN_OUTPUT] = output_raw

        # Session / user id — promote to Laminar association properties so the
        # trace groups by session in the UI.
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
            for prefix in (_TRACE_METADATA_PREFIX,
                           _OBSERVATION_METADATA_PREFIX):
                if k == prefix:
                    parsed = _parse_json(v)
                    if isinstance(parsed, dict):
                        for mk, mv in parsed.items():
                            new_attrs[
                                f"{ASSOCIATION_PROPERTIES}.metadata.{mk}"
                            ] = (
                                mv
                                if isinstance(mv, (str, int, float, bool))
                                else json.dumps(mv)
                            )
                    break
                if k.startswith(prefix + "."):
                    sub = k[len(prefix) + 1:]
                    new_attrs[f"{ASSOCIATION_PROPERTIES}.metadata.{sub}"] = v
                    break

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
            logger.warning(
                "Failed to install Langfuse attribute translator: %s", exc)
            return
        cls._translator = translator
        cls._lmnr_span_processor = lmnr_span_processor
        cls._lmnr_tracer_provider = lmnr_tracer_provider

        # 2. For every already-initialized Langfuse client, attach our span
        #    processor and our translator to its `TracerProvider`. If Langfuse
        #    reused Laminar's provider, both are already attached — the
        #    `_handled_providers` guard makes this a no-op.
        self._attach_to_existing_langfuse_providers()

        # 3. Patch future Langfuse-client construction.
        self._patch_resource_manager()

        cls._installed = True

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
        except ImportError:
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
            attached = False
            if self._translator is not None:
                _prepend_span_processor(provider, self._translator)
                attached = True
            if self._lmnr_span_processor is not None:
                add = getattr(provider, "add_span_processor", None)
                if callable(add):
                    add(self._lmnr_span_processor)
                    attached = True
            if attached:
                # Track so `uninstrument` can walk back and detach.
                type(self)._attached_providers[pid] = provider
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Failed to attach Laminar processor to Langfuse TracerProvider: %s",
                exc,
            )

    def _patch_resource_manager(self) -> None:
        try:
            from langfuse._client.resource_manager import (  # type: ignore[import-not-found]
                LangfuseResourceManager,
            )
        except ImportError:
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
        except ImportError:
            return
        LangfuseResourceManager._initialize_instance = cls._original_initialize_instance
        cls._original_initialize_instance = None

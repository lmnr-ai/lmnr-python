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

This package is organized as:
- `attributes.py` — attribute-name constants (pure data).
- `translate.py` — pure Langfuse/openinference → Laminar/GenAI shape functions.
- `processor.py` — the `LangfuseAttributeTranslator` `SpanProcessor` and the
  span-processor-ordering helpers.
- `provider_attachment.py` — `ProviderAttachment`, which owns attaching /
  detaching the translator + Laminar span processor to Langfuse-owned
  `TracerProvider`s (including future ones, via the resource-manager patch).
- `litellm_bridge.py` — `LiteLLMLangfuseBridge`, which bridges LiteLLM's
  `langfuse_otel` success callback (a separate code path Langfuse's own
  resource manager never sees).
- This module — the `LangfuseInstrumentor` orchestrator (a real
  `BaseInstrumentor`) and the module-level singleton accessor.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from typing_extensions import override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse.litellm_bridge import (
    LiteLLMLangfuseBridge,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse.processor import (
    LangfuseAttributeTranslator,
    prepend_span_processor,
    remove_span_processor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse.provider_attachment import (
    ProviderAttachment,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse.translate import (
    is_langfuse_span,
    is_llm_span,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

__all__ = [
    "LangfuseAttributeTranslator",
    "LangfuseInstrumentor",
    "get_langfuse_instrumentor",
    # Re-exported for tests / callers that reached into the old monolithic
    # module directly; not part of the documented public API.
    "is_langfuse_span",
    "is_llm_span",
    "langfuse_sdk_importable",
    "prepend_span_processor",
    "remove_span_processor",
]


def langfuse_sdk_importable() -> bool:
    """Probe whether the real `langfuse` SDK can actually be imported.

    `_langfuse_installed()` (in `tracing.instruments`) only reads install
    *metadata* — it deliberately never imports the SDK. But langfuse pins
    pydantic v1, whose generated API models fail to build on Python 3.14
    (`pydantic.v1.errors.ConfigError: unable to infer type`). On such an
    interpreter the package is present (metadata says >= 3.0) yet
    `import langfuse._client.resource_manager` raises, so the bridge's
    resource-manager attach/patch path silently no-ops and SDK spans
    (`@observe`, `langfuse.openai`, `langfuse.langchain`) never reach Laminar.

    Callers use this to refuse to report success for an install that would be
    inert. We import `resource_manager` specifically because that's the exact
    module the attach/patch path needs.
    """
    try:
        import langfuse._client.resource_manager  # noqa: F401, # pyright: ignore[reportMissingTypeStubs, reportUnusedImport]
    except Exception:
        return False
    return True


class LangfuseInstrumentor(BaseInstrumentor):
    """Attaches Laminar's span processor to every Langfuse `TracerProvider`.

    A real `BaseInstrumentor`. `init_instrumentations` still has a
    special-case branch for it (see `tracing/instruments.py`): unlike every
    other instrumentor, this one needs the caller-supplied Laminar
    `SpanProcessor` (not just a `tracer_provider`/`logger_provider`) so it can
    dual-attach that same processor onto Langfuse-owned `TracerProvider`s —
    `_instrument` reads `lmnr_tracer_provider` / `lmnr_span_processor` from
    kwargs.
    """

    def __init__(self) -> None:
        super().__init__()
        self._translator: LangfuseAttributeTranslator | None = None
        self._lmnr_span_processor: SpanProcessor | None = None
        self._lmnr_tracer_provider: SdkTracerProvider | None = None
        self._provider_attachment: ProviderAttachment | None = None
        self._litellm_bridge: LiteLLMLangfuseBridge | None = None

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return ("langfuse >= 3.0.0",)

    @override
    def _instrument(self, **kwargs: Any) -> None:  # pyright: ignore[reportAny, reportExplicitAny]
        lmnr_tracer_provider: SdkTracerProvider = kwargs["lmnr_tracer_provider"]  # pyright: ignore[reportAny]
        lmnr_span_processor: SpanProcessor = kwargs["lmnr_span_processor"]  # pyright: ignore[reportAny]

        # 1. Translator lives on Laminar's own provider so it sees every
        #    Langfuse span that reaches the Laminar exporter (including spans
        #    that arrived via Laminar's own provider being shared with
        #    Langfuse). Prepend it so it mutates `langfuse.*` attrs before
        #    `LaminarSpanProcessor` exports them — critical for the
        #    `disable_batch=True` / SimpleSpanProcessor path, where export
        #    happens synchronously inside `on_end`.
        translator = LangfuseAttributeTranslator()
        try:
            _success = prepend_span_processor(lmnr_tracer_provider, translator)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to install Langfuse attribute translator: %s", exc)
            # Re-raise (rather than silently returning) so
            # `BaseInstrumentor.instrument()` — which only flips
            # `_is_instrumented_by_opentelemetry = True` after `_instrument`
            # returns WITHOUT raising — does not report success. Callers
            # (`connect_to_langfuse()`, `init_instrumentations`) already wrap
            # `instrument()` in their own try/except and degrade to a
            # warning + `False`/`continue`.
            raise
        self._translator = translator
        self._lmnr_span_processor = lmnr_span_processor
        self._lmnr_tracer_provider = lmnr_tracer_provider

        provider_attachment = ProviderAttachment(translator, lmnr_span_processor)
        # Pre-register Laminar's provider id so `attach()` short-circuits if
        # Langfuse happens to share it. The `get_tracer_wrapper()` fallback
        # check is racy during auto-install — `init_instrumentations` runs
        # BEFORE `init_tracing` publishes the wrapper, so `get_tracer_wrapper()`
        # returns None, and a Langfuse client that had already been
        # constructed against a pre-existing global provider identical to
        # Laminar's would otherwise get the translator + laminar span
        # processor attached a second time. id()-based short-circuit is
        # independent of the tracing lifecycle.
        provider_attachment.mark_handled(lmnr_tracer_provider)
        self._provider_attachment = provider_attachment
        self._litellm_bridge = LiteLLMLangfuseBridge(provider_attachment, is_llm_span)

        # 2 & 3. Attach to already-initialized Langfuse clients and patch
        # future-client construction. If either raises an uncaught exception,
        # the translator we just prepended to Laminar's provider would be
        # left orphaned — a subsequent `instrument()` (e.g. via
        # `Laminar.connect_to_langfuse()`) would prepend a SECOND translator,
        # causing every Langfuse span to be translated twice. Roll back by
        # walking the half-applied state through `_teardown()`.
        try:
            # For every already-initialized Langfuse client, attach our span
            # processor and our translator to its `TracerProvider`. If
            # Langfuse reused Laminar's provider, both are already attached —
            # the `_handled_providers` guard makes this a no-op.
            provider_attachment.attach_to_existing_langfuse_providers()
            # Patch future Langfuse-client construction.
            provider_attachment.patch_resource_manager()
            # LiteLLM's `langfuse_otel` success callback never registers with
            # `LangfuseResourceManager`, so the two steps above can't reach it.
            # Attach to its private TracerProvider separately (existing loggers
            # + a factory patch for ones constructed later).
            self._litellm_bridge.attach_to_existing_loggers()
            self._litellm_bridge.patch_logger_factory()
        except Exception:  # pylint: disable=broad-exception-caught
            # Best-effort cleanup: detach whatever we've attached so far and
            # clear instance state. `_teardown()` is idempotent and tolerant
            # of partial state (it only touches providers we recorded in
            # `_attached_providers`).
            self._teardown()
            raise

    def rebind(
        self,
        lmnr_tracer_provider: SdkTracerProvider,
        lmnr_span_processor: SpanProcessor,
    ) -> bool:
        """Point an already-installed bridge at a new Laminar span processor.

        `Laminar.shutdown()` retires the run's `LaminarSpanProcessor` and a
        later `initialize()` builds a fresh one, but `instrument()` (via
        `BaseInstrumentor`) returns early once already instrumented — so
        without this every Langfuse-owned provider (and every LiteLLM
        `langfuse_otel` logger provider) would keep calling `on_end` on the
        shut-down processor and its spans would never reach the new exporter.
        Driven from `Laminar.initialize()`, because nothing else runs on a
        re-init: `LANGFUSE` is never in the default instrument set and
        `connect_to_langfuse()` is a one-shot the user calls.

        Returns True if a swap happened.
        """
        if not self.is_instrumented_by_opentelemetry:
            return False

        old_processor = self._lmnr_span_processor
        provider_moved = self._lmnr_tracer_provider is not lmnr_tracer_provider
        if old_processor is lmnr_span_processor and not provider_moved:
            return False

        # The translator lives on Laminar's OWN provider, which is reused
        # across initialize()/shutdown() cycles — so this branch is normally
        # dead. It only fires if that provider is ever swapped.
        if provider_moved and self._translator is not None:
            if self._lmnr_tracer_provider is not None:
                _success = remove_span_processor(self._lmnr_tracer_provider, self._translator)
            try:
                _success = prepend_span_processor(lmnr_tracer_provider, self._translator)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to move Langfuse translator: %s", exc)
                return False
            if self._provider_attachment is not None:
                self._provider_attachment.mark_handled(lmnr_tracer_provider)

        self._lmnr_tracer_provider = lmnr_tracer_provider
        self._lmnr_span_processor = lmnr_span_processor

        # Every attach path (Langfuse clients, the resource-manager patch, and
        # the LiteLLM logger providers) records into `_attached_providers`, so
        # this covers all of them.
        if self._provider_attachment is not None:
            self._provider_attachment.rebind(old_processor, lmnr_span_processor)
        return True

    def _teardown(self) -> None:
        """Reverse `_instrument`: detach the translator and Laminar span
        processor from every provider we attached them to, restore the
        resource-manager / LiteLLM-factory patches, and clear instance state
        so a subsequent `_instrument()` starts from a clean slate.

        Called from BOTH `_uninstrument()` and the exception-rollback path in
        `_instrument()` — `BaseInstrumentor.instrument()` only flips
        `_is_instrumented_by_opentelemetry = True` AFTER `_instrument()`
        returns successfully, so calling the public `uninstrument()` from a
        failing `_instrument()` would no-op (flag still False) and leak the
        translator. This private method runs unconditionally instead.

        Without the full reset, a re-install would prepend a second
        translator onto Laminar's provider (the first was never removed) and
        `_handled_providers` would still contain stale ids from the previous
        session so `attach_to_existing_langfuse_providers` would skip
        already-seen Langfuse providers instead of re-attaching.
        """
        if self._provider_attachment is not None:
            self._provider_attachment.unpatch_resource_manager()
        if self._litellm_bridge is not None:
            self._litellm_bridge.unpatch_logger_factory()
            self._litellm_bridge.unwrap_loggers()

        translator = self._translator
        lmnr_provider = self._lmnr_tracer_provider

        # Detach translator from Laminar's provider.
        if lmnr_provider is not None and translator is not None:
            _success = remove_span_processor(lmnr_provider, translator)

        # Detach translator + laminar span processor from every Langfuse
        # provider we attached them to.
        if self._provider_attachment is not None:
            self._provider_attachment.detach_all()

        self._translator = None
        self._lmnr_span_processor = None
        self._lmnr_tracer_provider = None
        self._provider_attachment = None
        self._litellm_bridge = None

    @override
    def _uninstrument(self, **kwargs: Any) -> None:  # pyright: ignore[reportAny, reportExplicitAny]
        self._teardown()


# Module-level singleton, mirroring `get_tracer_wrapper()` in
# `tracing/__init__.py`. `BaseInstrumentor.__new__` already makes
# `LangfuseInstrumentor()` a per-class singleton, but routing every caller
# through one accessor keeps the intent explicit and matches the idiom used
# for `TracerWrapper`.
_instrumentor = LangfuseInstrumentor()


def get_langfuse_instrumentor() -> LangfuseInstrumentor:
    return _instrumentor

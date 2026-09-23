"""Bookkeeping for attaching Laminar's translator + span processor to every
Langfuse-owned `TracerProvider`, and for keeping future Langfuse clients
covered (the `LangfuseResourceManager._initialize_instance` monkey-patch).

Split out of the original monolithic `LangfuseInstrumentor` — this class owns
exactly the "which providers have we touched, and how" concern; it carries no
knowledge of LiteLLM or of the `BaseInstrumentor` lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import TracerProvider

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse.processor import (
    LangfuseAttributeTranslator,
    prepend_span_processor,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class ProviderAttachment:
    """Attaches / detaches `(translator, lmnr_span_processor)` to Langfuse
    `TracerProvider`s, and keeps future Langfuse clients covered.

    Constructed fresh per `LangfuseInstrumentor._instrument()` call, holding
    references to the translator and Laminar span processor it needs to
    attach for the lifetime of that install.
    """

    def __init__(
        self,
        translator: LangfuseAttributeTranslator | None,
        lmnr_span_processor: SpanProcessor | None,
    ) -> None:
        self._translator: LangfuseAttributeTranslator | None = translator
        self._lmnr_span_processor: SpanProcessor | None = lmnr_span_processor
        self._handled_providers: set[int] = set()
        #: Providers we attached the translator / span processor to, keyed by id().
        #: `detach_all` walks this map to detach what we added. The reference is
        #: only held for the duration of an install; `detach_all` clears it
        #: immediately so the instrumentor never pins a provider long-term.
        self._attached_providers: dict[int, TracerProvider] = {}
        self._original_initialize_instance: Callable[..., None] | None = None

    def mark_handled(self, provider: TracerProvider) -> None:
        """Pre-register `provider`'s id so `attach()` short-circuits if it's
        ever offered this same provider (e.g. Laminar's own provider)."""
        self._handled_providers.add(id(provider))

    def attach(self, provider: TracerProvider | None) -> None:
        if provider is None:
            return
        pid = id(provider)
        if pid in self._handled_providers:
            return
        self._handled_providers.add(pid)

        # Skip the Laminar provider itself — our processor and translator are
        # already attached there.
        from lmnr.opentelemetry_lib.tracing import get_tracer_wrapper

        lmnr_wrapper = get_tracer_wrapper()
        if lmnr_wrapper is not None and lmnr_wrapper.tracer_provider is provider:
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
            # `detach_all` could never detach the orphaned translator,
            # letting a reinstall stack a second one. `_remove_span_processor`
            # tolerates a processor that was never attached, so recording
            # eagerly is safe.
            if self._translator is not None:
                _success = prepend_span_processor(provider, self._translator)
                self._attached_providers[pid] = provider
            if self._lmnr_span_processor is not None:
                add = getattr(provider, "add_span_processor", None)
                if callable(add):
                    _success = add(self._lmnr_span_processor)
                    self._attached_providers[pid] = provider
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Failed to attach Laminar processor to Langfuse TracerProvider: %s",
                exc,
            )
            # Roll back the partial attach and un-mark the provider so a later
            # attempt (e.g. the resource-manager re-init hook) can retry. Left
            # as-is, `pid` would stay in `_handled_providers` and every future
            # `attach()` call would short-circuit, so dual-export to Laminar
            # would never run for this provider.
            from .processor import remove_span_processor

            if self._translator is not None:
                _removed = remove_span_processor(provider, self._translator)
            if self._lmnr_span_processor is not None:
                _removed = remove_span_processor(provider, self._lmnr_span_processor)
            self._handled_providers.discard(pid)
            _provider = self._attached_providers.pop(pid, None)

    def detach_all(self) -> None:
        from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse.processor import (
            remove_span_processor,
        )

        translator = self._translator
        lmnr_processor = self._lmnr_span_processor
        for provider in list(self._attached_providers.values()):
            if translator is not None:
                _removed = remove_span_processor(provider, translator)
            if lmnr_processor is not None:
                _removed= remove_span_processor(provider, lmnr_processor)
        self._attached_providers = {}
        self._handled_providers = set()

    def rebind(self, old_processor: SpanProcessor | None, new_processor: SpanProcessor) -> None:
        """Swap `old_processor` for `new_processor` on every provider we
        previously attached a Laminar span processor to."""
        from .processor import remove_span_processor

        self._lmnr_span_processor = new_processor
        for provider in list(self._attached_providers.values()):
            if old_processor is not None:
                _removed = remove_span_processor(provider, old_processor)
            add = getattr(provider, "add_span_processor", None)
            if callable(add):
                try:
                    _added = add(new_processor)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning(
                        "Failed to rebind Laminar processor on Langfuse " +
                        "TracerProvider: %s",
                        exc,
                    )

    # --- future Langfuse-client coverage ---

    def attach_to_existing_langfuse_providers(self) -> None:
        try:
            from langfuse._client.resource_manager import (
                LangfuseResourceManager,
            )
        except Exception:
            # ImportError if not installed. Other exceptions (e.g. pydantic
            # v1 ConfigError on Python 3.14 due to langfuse's own pydantic
            # compat bug) mean the SDK is unusable in this interpreter —
            # treat the same as absent and leave the bridge installed but
            # inert for the resource-manager path.
            return

        instances: dict[str, LangfuseResourceManager] = getattr(LangfuseResourceManager, "_instances", {}) or {}
        for rm in instances.values():
            provider: TracerProvider | None = getattr(rm, "tracer_provider", None)
            self.attach(provider)

    def patch_resource_manager(self) -> None:
        try:
            from langfuse._client.resource_manager import (
                LangfuseResourceManager,
            )
        except Exception:
            # See `attach_to_existing_langfuse_providers` — any import-time
            # failure means Langfuse isn't usable in this interpreter.
            return

        if self._original_initialize_instance is not None:
            return

        original = LangfuseResourceManager._initialize_instance
        self._original_initialize_instance = original
        provider_attachment = self

        def patched(self_rm: LangfuseResourceManager, *args: Any, **kwargs: Any):  # pyright: ignore[reportAny, reportExplicitAny]
            result = original(self_rm, *args, **kwargs)
            try:
                provider_attachment.attach(getattr(self_rm, "tracer_provider", None))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Langfuse post-init attach failed: %s", exc)
            return result

        LangfuseResourceManager._initialize_instance = patched  # pyright: ignore[reportAttributeAccessIssue]

    def unpatch_resource_manager(self) -> None:
        if self._original_initialize_instance is None:
            return
        try:
            from langfuse._client.resource_manager import (
                LangfuseResourceManager,
            )

            LangfuseResourceManager._initialize_instance = (
                self._original_initialize_instance
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # If langfuse can't be imported here it isn't usable in this
            # interpreter, so the patched hook can never be invoked anyway.
            # Clear the bookkeeping regardless so a later reset leaves no
            # half-reset state (a retained `_original_initialize_instance`
            # would make a later `patch_resource_manager` short-circuit).
            logger.debug(
                "Could not restore Langfuse _initialize_instance patch: %s", exc
            )
        finally:
            self._original_initialize_instance = None

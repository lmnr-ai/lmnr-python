"""Bridges LiteLLM's `langfuse_otel` success callback into Laminar.

LiteLLM ships a `langfuse_otel` success callback
(`litellm.integrations.langfuse.langfuse_otel.LangfuseOtelLogger`) that
subclasses LiteLLM's base `OpenTelemetry` integration with
`skip_set_global=True`. It builds its OWN private `TracerProvider`s and
exports OTLP straight to Langfuse Cloud — it never registers with
`LangfuseResourceManager`, so `ProviderAttachment`'s resource-manager
attach/patch path can't see it. The spans it emits carry a hybrid of
`langfuse.*` attrs AND openinference `llm.*` indexed attrs, both of which
`LangfuseAttributeTranslator` already knows how to translate, so the only
work is making those spans flow into Laminar with the right shape. Two things
have to happen, both per-logger (see `_patch_logger`):

  Layer 1 — provider attachment. When the request carries credentials (the
  common `langfuse_otel` case — env-var or dynamic headers), LiteLLM does NOT
  emit through `logger._tracer_provider`. It calls
  `_get_tracer_with_dynamic_headers`, which lazily builds a SEPARATE
  `TracerProvider` per credential set and caches it in
  `logger._tracer_provider_cache`. Attaching Laminar's translator + span
  processor only to `logger._tracer_provider` therefore misses every
  `litellm_request` / `raw_gen_ai_request` span. We wrap
  `_get_tracer_with_dynamic_headers` to dual-attach to each cache provider as
  it appears (and still attach to `_tracer_provider` for the no-credentials
  path).

  Layer 2 — force a primary span. In `_handle_success`, when a parent span is
  active and `USE_OTEL_LITELLM_REQUEST_SPAN` is unset (the default), LiteLLM
  creates NO `litellm_request` span and instead folds `gen_ai.*` /
  openinference attrs onto the parent. With `instruments=[Instruments.LANGFUSE]`
  the active parent is the user's `@observe` root, so the root gets
  mis-marked LLM by app-server's `gen_ai.*` heuristic and no LLM span is
  produced at all. We wrap `_get_span_context` to report `parent_span=None`
  (while keeping the parent CONTEXT, so nesting is preserved), which flips
  LiteLLM's `should_create_primary_span` to True without touching the global
  env var. The wrap is gated: if the parent is already an LLM span (Laminar's
  own `litellm.completion`, present when `Instruments.LITELLM` also runs),
  folding is correct and we leave the parent untouched to avoid a duplicate
  nested LLM span.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from opentelemetry.context import Context
from opentelemetry.trace import Span, Tracer, TracerProvider

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse.provider_attachment import (
    ProviderAttachment,
)
from lmnr.sdk.log import get_default_logger

if TYPE_CHECKING:
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger

logger = get_default_logger(__name__)

GetCtxReturnType = tuple[Context, Span] | tuple[Context, None] | tuple[None, None]

class LiteLLMLangfuseBridge:
    """Attaches Laminar's translator + span processor to LiteLLM's
    `langfuse_otel` loggers, and forces a primary span for them.

    Takes a `ProviderAttachment` (to reuse its `attach()`/id-dedup logic) and
    an `is_llm_span` predicate (from `translate.py`) as constructor args, so
    it has no back-reference into the orchestrator.
    """

    def __init__(
        self,
        provider_attachment: ProviderAttachment,
        is_llm_span: Callable[[Span], bool],
    ) -> None:
        self._provider_attachment: ProviderAttachment = provider_attachment
        self._is_llm_span: Callable[[Span], bool] = is_llm_span
        self._original_litellm_init_logger: Callable[..., Any] | None = None  # pyright: ignore[reportExplicitAny]
        #: LiteLLM `langfuse_otel` loggers we wrapped, keyed by logger id(),
        #: mapping to `(logger, original_get_tracer, original_get_span_context)`.
        #: Lets `unwrap_loggers` restore both bound methods.
        self._wrapped_litellm_loggers: dict[
            int, tuple[LangfuseOtelLogger, Callable[..., Any], Callable[..., Any]]  # pyright: ignore[reportExplicitAny]
        ] = {}

    def attach_to_existing_loggers(self) -> None:
        for logger_obj in self._iter_litellm_langfuse_otel_loggers():
            self._patch_logger(logger_obj)

    @staticmethod
    def _iter_litellm_langfuse_otel_loggers() -> list[LangfuseOtelLogger]:
        """Return every constructed LiteLLM `langfuse_otel` logger instance.

        LiteLLM keeps callback singletons in
        `litellm.litellm_core_utils.litellm_logging._in_memory_loggers`. We
        filter that list for `LangfuseOtelLogger` instances. Any import/attr
        failure (LiteLLM absent, internal layout changed) yields an empty
        list so the bridge stays inert rather than raising.
        """
        try:
            from litellm.integrations.langfuse.langfuse_otel import (
                LangfuseOtelLogger,
            )
            from litellm.litellm_core_utils import (
                litellm_logging,
            )
        except Exception:
            return []
        loggers: list[CustomLogger] = getattr(litellm_logging, "_in_memory_loggers", None) or []
        return [lg for lg in loggers if isinstance(lg, LangfuseOtelLogger)]

    def _patch_logger(self, logger_obj: LangfuseOtelLogger | None) -> None:
        """Apply both bridge layers to a single `langfuse_otel` logger.

        Idempotent per logger (guarded by `_wrapped_litellm_loggers`), so it's
        safe to call from both the existing-logger scan and the factory patch.
        Also attaches to `logger._tracer_provider` for the no-credentials path,
        reusing the id()-guarded `ProviderAttachment.attach`.
        """
        if logger_obj is None:
            return
        lid = id(logger_obj)
        if lid in self._wrapped_litellm_loggers:
            return

        # No-credentials path still emits through `_tracer_provider`.
        self._provider_attachment.attach(getattr(logger_obj, "_tracer_provider", None))

        orig_get_tracer: Callable[[dict[str, str]], Tracer] | None = getattr(logger_obj, "_get_tracer_with_dynamic_headers", None)
        orig_get_ctx: Callable[[dict[str, Any], Span | None], tuple[Context, Span]] | None = getattr(logger_obj, "_get_span_context", None)  # pyright: ignore[reportExplicitAny]
        if not callable(orig_get_tracer) or not callable(orig_get_ctx):
            return

        provider_attachment = self._provider_attachment
        is_llm_span = self._is_llm_span

        def patched_get_tracer(dynamic_headers: dict[str, str], _orig: Callable[[dict[str, str]], Tracer]=orig_get_tracer) -> Tracer:
            tracer = _orig(dynamic_headers)
            # Attach to every cache provider we haven't seen yet. The cache is
            # keyed by credential set, so per-team keys each get a provider;
            # `attach()`'s id() guard makes repeats a no-op.
            try:
                cache: dict[Any, TracerProvider] = getattr(logger_obj, "_tracer_provider_cache", None) or {}  # pyright: ignore[reportExplicitAny]
                for provider in list(cache.values()):
                    provider_attachment.attach(provider)
            except Exception as exc:
                logger.debug("LiteLLM cache-provider attach failed: %s", exc)
            return tracer

        def patched_get_ctx(
            kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
            default_span: Span | None=None,
            _orig: Callable[[dict[str, Any], Span | None], tuple[Context, Span]]=orig_get_ctx  # pyright: ignore[reportExplicitAny]
        ) -> GetCtxReturnType:
            ctx, parent_span = _orig(kwargs, default_span)
            # Only force primary-span creation when folding would corrupt the
            # parent. If the parent is already an LLM span (Laminar's own
            # `litellm.completion`), folding LiteLLM's attrs onto it is the
            # correct, deduplicated shape — leave it alone.
            if parent_span is not None and not is_llm_span(parent_span):  # pyright: ignore[reportUnnecessaryComparison]
                return ctx, None
            return ctx, parent_span

        logger_obj._get_tracer_with_dynamic_headers = patched_get_tracer  # pyright: ignore[reportPrivateUsage]
        logger_obj._get_span_context = patched_get_ctx  # pyright: ignore[reportPrivateUsage]
        self._wrapped_litellm_loggers[lid] = (
            logger_obj,
            orig_get_tracer,
            orig_get_ctx,
        )

    def patch_logger_factory(self) -> None:
        """Wrap LiteLLM's `_init_custom_logger_compatible_class` so any
        `langfuse_otel` logger constructed AFTER the bridge installs also gets
        its private provider dual-attached.

        LiteLLM constructs the callback lazily — the first LLM call (or a
        `litellm.success_callback = ["langfuse_otel"]` assignment that triggers
        `litellm.utils._init_custom_callbacks`) is what builds the logger. All
        call sites import the factory freshly from the module each time (see
        `litellm/utils.py`, `litellm/proxy/...`), so a module-level patch is
        observed by every caller. We attach on the way out, reusing the
        idempotent `ProviderAttachment.attach` (its `_handled_providers` guard
        makes repeat calls for the same provider a no-op).
        """
        try:
            from litellm.integrations.langfuse.langfuse_otel import (
                LangfuseOtelLogger,
            )
            from litellm.litellm_core_utils import (
                litellm_logging,
            )
        except Exception:
            return

        if self._original_litellm_init_logger is not None:
            return

        original = getattr(
            litellm_logging, "_init_custom_logger_compatible_class", None
        )
        if not callable(original):
            return
        self._original_litellm_init_logger = original
        bridge = self

        def patched(*args: Any, **kwargs: Any):  # pyright: ignore[reportExplicitAny, reportAny]
            result = original(*args, **kwargs)
            try:
                # Only the `langfuse_otel` callback should be bridged. The
                # factory builds many OTel-based callbacks (arize, otel, …),
                # all of which carry a private `_tracer_provider`; attaching
                # Laminar's translator + exporter to those would ship
                # unrelated spans into Laminar.
                if isinstance(result, LangfuseOtelLogger):
                    bridge._patch_logger(result)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("LiteLLM post-init attach failed: %s", exc)
            return result

        litellm_logging._init_custom_logger_compatible_class = patched  # pyright: ignore[reportPrivateUsage]

    def unpatch_logger_factory(self) -> None:
        if self._original_litellm_init_logger is None:
            return
        try:
            from litellm.litellm_core_utils import (
                litellm_logging,
            )

            litellm_logging._init_custom_logger_compatible_class = (  # pyright: ignore[reportPrivateUsage]
                self._original_litellm_init_logger
            )
        except Exception as exc:
            logger.debug("Could not restore LiteLLM logger factory patch: %s", exc)
        finally:
            self._original_litellm_init_logger = None

    def unwrap_loggers(self) -> None:
        """Restore the original `_get_tracer_with_dynamic_headers` /
        `_get_span_context` bound methods on every `langfuse_otel` logger we
        wrapped. Mirror of the wrapping in `_patch_logger`.
        """
        for logger_obj, orig_get_tracer, orig_get_ctx in list(
            self._wrapped_litellm_loggers.values()
        ):
            try:
                logger_obj._get_tracer_with_dynamic_headers = orig_get_tracer  # pyright: ignore[reportPrivateUsage]
                logger_obj._get_span_context = orig_get_ctx  # pyright: ignore[reportPrivateUsage]
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Could not restore LiteLLM logger wrap: %s", exc)
        self._wrapped_litellm_loggers = {}

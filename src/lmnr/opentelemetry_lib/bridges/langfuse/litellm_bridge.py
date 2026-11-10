"""Bridges LiteLLM's `langfuse_otel` success callback into Laminar.

`LangfuseOtelLogger` (`litellm/integrations/langfuse/langfuse_otel.py`) builds
its own private `TracerProvider`s that export straight to Langfuse; it never
registers with `LangfuseResourceManager`. With credentials on the request (the
common case) LiteLLM does not emit through `logger._tracer_provider` either:
`_get_tracer_with_dynamic_headers` lazily builds one provider per credential
set in `logger._tracer_provider_cache`. So we attach to `_tracer_provider` and
wrap `_get_tracer_with_dynamic_headers` to attach to each cached provider as it
appears. Nothing LiteLLM sends to Langfuse changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import Tracer, TracerProvider

from lmnr.opentelemetry_lib.bridges.langfuse.provider_attachment import ProviderAttachment
from lmnr.sdk.log import get_default_logger

if TYPE_CHECKING:
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger

logger = get_default_logger(__name__)


class LiteLLMLangfuseBridge:
    def __init__(self, provider_attachment: ProviderAttachment) -> None:
        self._provider_attachment: ProviderAttachment = provider_attachment
        self._original_litellm_init_logger: Callable[..., Any] | None = None
        #: logger id() -> (logger, original `_get_tracer_with_dynamic_headers`)
        self._wrapped_loggers: dict[int, tuple[LangfuseOtelLogger, Callable[..., Any]]] = {}

    def attach_to_existing_loggers(self) -> None:
        for logger_obj in self._iter_litellm_langfuse_otel_loggers():
            self._patch_logger(logger_obj)

    @staticmethod
    def _iter_litellm_langfuse_otel_loggers() -> list[LangfuseOtelLogger]:
        """Every constructed `langfuse_otel` logger, from LiteLLM's callback
        singletons in `litellm_logging._in_memory_loggers`."""
        try:
            from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
            from litellm.litellm_core_utils import litellm_logging
        except Exception:
            return []
        loggers: list[CustomLogger] = getattr(litellm_logging, "_in_memory_loggers", None) or []
        return [lg for lg in loggers if isinstance(lg, LangfuseOtelLogger)]

    def _patch_logger(self, logger_obj: LangfuseOtelLogger | None) -> None:
        if logger_obj is None or id(logger_obj) in self._wrapped_loggers:
            return
        attachment = self._provider_attachment
        attachment.attach(getattr(logger_obj, "_tracer_provider", None), langfuse_owned=False)

        orig_get_tracer: Callable[[Mapping[str, str]], Tracer] | None = getattr(
            logger_obj, "_get_tracer_with_dynamic_headers", None
        )
        if not callable(orig_get_tracer):
            return

        def patched_get_tracer(
            dynamic_headers: Mapping[str, str],
            _orig: Callable[[Mapping[str, str]], Tracer] = orig_get_tracer,
        ) -> Tracer:
            tracer = _orig(dynamic_headers)
            try:
                cache: dict[Any, Any] = getattr(logger_obj, "_tracer_provider_cache", None) or {}
                for entry in list(cache.values()):
                    # Newer LiteLLM wraps each one in a `_CachedTracerProvider`.
                    provider: TracerProvider | None = getattr(entry, "provider", entry)
                    attachment.attach(provider, langfuse_owned=False)
            except Exception:
                logger.debug("LiteLLM cache-provider attach failed", exc_info=True)
            return tracer

        logger_obj._get_tracer_with_dynamic_headers = patched_get_tracer
        self._wrapped_loggers[id(logger_obj)] = (logger_obj, orig_get_tracer)

    def patch_logger_factory(self) -> None:
        """Wrap LiteLLM's `_init_custom_logger_compatible_class` so a
        `langfuse_otel` logger built later (LiteLLM builds callbacks lazily) is
        bridged too. Call sites import the factory from the module each time,
        so the module-level patch is seen everywhere."""
        try:
            from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
            from litellm.litellm_core_utils import litellm_logging
        except Exception:
            return
        if self._original_litellm_init_logger is not None:
            return
        original = getattr(litellm_logging, "_init_custom_logger_compatible_class", None)
        if not callable(original):
            return
        self._original_litellm_init_logger = original
        bridge = self

        def patched(*args: Any, **kwargs: Any):
            result = original(*args, **kwargs)
            try:
                # Only `langfuse_otel`: the factory also builds other OTel
                # callbacks (arize, otel, ...) whose spans aren't Langfuse's.
                if isinstance(result, LangfuseOtelLogger):
                    bridge._patch_logger(result)
            except Exception:
                logger.debug("LiteLLM post-init attach failed", exc_info=True)
            return result

        litellm_logging._init_custom_logger_compatible_class = patched

    def unpatch_logger_factory(self) -> None:
        if self._original_litellm_init_logger is None:
            return
        try:
            from litellm.litellm_core_utils import litellm_logging

            litellm_logging._init_custom_logger_compatible_class = (
                self._original_litellm_init_logger
            )
        except Exception:
            logger.debug("Could not restore LiteLLM logger factory patch", exc_info=True)
        finally:
            self._original_litellm_init_logger = None

    def unwrap_loggers(self) -> None:
        for logger_obj, orig_get_tracer in list(self._wrapped_loggers.values()):
            try:
                logger_obj._get_tracer_with_dynamic_headers = orig_get_tracer
            except Exception:
                logger.debug("Could not restore LiteLLM logger wrap", exc_info=True)
        self._wrapped_loggers = {}

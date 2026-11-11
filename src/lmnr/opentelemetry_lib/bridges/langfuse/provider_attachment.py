"""Attaches a `LangfuseBridgeSpanProcessor` to every Langfuse-owned
`TracerProvider`, including ones created later (the
`LangfuseResourceManager._initialize_instance` patch).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from threading import Lock
from typing import Any, cast

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, SynchronousMultiSpanProcessor
from opentelemetry.trace import TracerProvider

from lmnr.opentelemetry_lib.bridges.langfuse.processor import (
    LangfuseBridgeSpanProcessor,
    langfuse_would_export,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


def remove_span_processor(provider: TracerProvider, processor: SpanProcessor) -> bool:
    """Remove `processor` from `provider`. The SDK has no public API for
    this, so it walks the private `_span_processors` tuple under its lock.
    Every branch logs at debug level, since each one depends on OTel SDK
    internals that may change."""
    active: SynchronousMultiSpanProcessor | None = getattr(provider, "_active_span_processor", None)
    if active is None:
        logger.debug(
            "Not removing the bridge processor: %r has no _active_span_processor", provider
        )
        return False
    lock: Lock | None = getattr(active, "_lock", None)
    current: tuple[SpanProcessor, ...] | None = getattr(active, "_span_processors", None)
    if current is None:
        logger.debug("Not removing the bridge processor: %r has no _span_processors", type(active))
        return False
    try:
        filtered = tuple(p for p in current if p is not processor)
        if len(filtered) == len(current):
            logger.debug("Not removing the bridge processor: it is not on %r", provider)
            return False
        if lock is not None:
            with lock:
                active._span_processors = filtered
        else:
            logger.debug("Removing the bridge processor from %r without a lock", provider)
            active._span_processors = filtered
    except Exception:
        logger.debug("Could not remove the bridge processor from %r", provider, exc_info=True)
        return False
    logger.debug("Removed the bridge processor from %r", provider)
    return True


class ProviderAttachment:
    """Owns which providers carry a bridge processor, so `detach_all` can undo
    exactly what was added."""

    def __init__(
        self,
        lmnr_span_processor: SpanProcessor,
        lmnr_tracer_provider: TracerProvider | None,
        bridge_version: str | None,
        json_encoded_metadata: bool,
    ) -> None:
        self._lmnr_span_processor: SpanProcessor = lmnr_span_processor
        self._lmnr_tracer_provider: TracerProvider | None = lmnr_tracer_provider
        self._bridge_version: str | None = bridge_version
        self._json_encoded_metadata: bool = json_encoded_metadata
        self._lock: Lock = Lock()
        #: provider id() -> (provider, the processor we added to it)
        self._attached: dict[int, tuple[TracerProvider, SpanProcessor]] = {}
        self._original_initialize_instance: Callable[..., None] | None = None

    def attach(self, provider: TracerProvider | None, langfuse_owned: bool = True) -> None:
        """Forward spans that end on `provider` to Laminar.

        `langfuse_owned` providers carry a `LangfuseSpanProcessor`, and only
        spans it exports are forwarded. Other providers (LiteLLM's
        `langfuse_otel` ones) export everything to Langfuse, so everything is
        forwarded.
        """
        if provider is None or provider is self._lmnr_tracer_provider:
            return
        add = getattr(provider, "add_span_processor", None)
        if not callable(add):
            return
        with self._lock:
            if id(provider) in self._attached:
                return
            should_forward: Callable[[ReadableSpan], bool] | None = (
                partial(langfuse_would_export, provider) if langfuse_owned else None
            )
            processor = LangfuseBridgeSpanProcessor(
                self._lmnr_span_processor,
                bridge_version=self._bridge_version,
                should_forward=should_forward,
                json_encoded_metadata=self._json_encoded_metadata,
            )
            try:
                _ = add(processor)
            except Exception:
                logger.warning(
                    "Failed to attach Laminar to a Langfuse TracerProvider", exc_info=True
                )
                return
            self._attached[id(provider)] = (provider, processor)

    def detach_all(self) -> None:
        with self._lock:
            attached = list(self._attached.values())
            self._attached = {}
        for provider, processor in attached:
            _removed = remove_span_processor(provider, processor)

    def attach_to_existing_langfuse_providers(self) -> None:
        try:
            from langfuse._client.resource_manager import LangfuseResourceManager
        except Exception:
            return
        instances = cast(dict[str, Any], getattr(LangfuseResourceManager, "_instances", None) or {})
        for rm in list(instances.values()):
            self.attach(getattr(rm, "tracer_provider", None))

    def patch_resource_manager(self) -> None:
        """Attach to the provider of every Langfuse client created later."""
        try:
            from langfuse._client.resource_manager import LangfuseResourceManager
        except Exception:
            return
        if self._original_initialize_instance is not None:
            return

        original = LangfuseResourceManager._initialize_instance
        self._original_initialize_instance = original
        attachment = self

        def patched(self_rm: Any, *args: Any, **kwargs: Any) -> None:
            result = original(self_rm, *args, **kwargs)
            try:
                attachment.attach(getattr(self_rm, "tracer_provider", None))
            except Exception:
                logger.debug("Langfuse post-init attach failed", exc_info=True)
            return result

        LangfuseResourceManager._initialize_instance = patched  # pyright: ignore[reportAttributeAccessIssue]

    def unpatch_resource_manager(self) -> None:
        if self._original_initialize_instance is None:
            return
        try:
            from langfuse._client.resource_manager import LangfuseResourceManager

            LangfuseResourceManager._initialize_instance = self._original_initialize_instance
        except Exception:
            logger.debug("Could not restore Langfuse _initialize_instance", exc_info=True)
        finally:
            self._original_initialize_instance = None

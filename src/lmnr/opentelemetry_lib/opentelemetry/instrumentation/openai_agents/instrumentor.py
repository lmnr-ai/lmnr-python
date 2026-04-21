"""OpenAIAgentsInstrumentor - BaseInstrumentor for OpenAI Agents SDK."""

from __future__ import annotations

from typing import Collection

from lmnr.sdk.log import get_default_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from .processor import LaminarAgentsTraceProcessor

logger = get_default_logger(__name__)

_instruments = ("openai-agents >= 0.7.0",)


class OpenAIAgentsInstrumentor(BaseInstrumentor):
    """Instrumentor for the OpenAI Agents SDK tracing module."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        try:
            from agents.tracing import add_trace_processor
        except Exception:
            logger.debug("OpenAI Agents SDK not available; skipping instrumentation")
            return

        processor = LaminarAgentsTraceProcessor()
        try:
            add_trace_processor(processor)
        except Exception as exc:
            logger.warning("Failed to register Laminar Agents processor: %s", exc)
            raise
        self._processor = processor
        logger.debug("Laminar OpenAI Agents trace processor registered")

    def _uninstrument(self, **kwargs):
        processor = getattr(self, "_processor", None)
        if processor is None:
            return
        try:
            from agents.tracing import get_trace_provider
            provider = get_trace_provider()
            # The SDK has no remove_trace_processor API, so read the
            # internal list and write back via the public set_processors.
            mp = getattr(provider, "_multi_processor", None)
            if mp is not None:
                current = getattr(mp, "_processors", ())
                provider.set_processors(
                    [p for p in current if p is not processor]
                )
        except Exception:
            pass
        processor.shutdown()
        self._processor = None

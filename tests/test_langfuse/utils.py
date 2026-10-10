"""Shared helpers for the Langfuse bridge tests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.util.types import AttributeValue
from typing_extensions import override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse import (
    get_langfuse_instrumentor,
)
from lmnr.opentelemetry_lib.tracing.instruments import (
    Instruments,
)
from lmnr.sdk.log import get_default_logger

# Raw-provider / framework instrumentors that overlap with what Langfuse emits
# through its own wrappers. The bridge is opt-in, so these must stay enabled in
# the default set even when langfuse is installed (the user only pays the
# double-coverage cost if they explicitly opt into LANGFUSE alongside them).
LANGFUSE_OVERLAP_PROVIDERS = {
    Instruments.ANTHROPIC,
    Instruments.BEDROCK,
    Instruments.COHERE,
    Instruments.GOOGLE_GENAI,
    Instruments.GROQ,
    Instruments.LANGCHAIN,
    Instruments.MISTRAL,
    Instruments.OPENAI,
}


def langfuse_sdk_importable() -> bool:
    """Probe whether the real `langfuse` SDK can be imported on this
    interpreter. Langfuse pins pydantic v1, which fails to build some of
    langfuse's own generated API models on Python 3.14
    (`pydantic.v1.errors.ConfigError: unable to infer type`). The bridge
    itself still loads (it only uses OTel), so the translator/attach-path
    unit tests stay green — the SDK-backed integration tests skip instead.
    """
    try:
        pass
    except Exception:
        return False
    return True


LANGFUSE_IMPORTABLE = langfuse_sdk_importable()
langfuse_sdk_required = pytest.mark.skipif(
    not LANGFUSE_IMPORTABLE,
    reason="langfuse SDK cannot be imported on this interpreter "
    + "(known pydantic v1 incompatibility on Python 3.14)",
)
logger = get_default_logger(__name__)


def reset_langfuse_bridge_state() -> None:
    """Reset the `get_langfuse_instrumentor()` singleton to a clean,
    uninstalled state.

    Routes through the real `_teardown()` (detaches any processors it still
    has attached, restores the resource-manager / LiteLLM-factory patches,
    and drops `_provider_attachment` / `_litellm_bridge`), then clears the
    `BaseInstrumentor`-owned instrumented flag directly — `_teardown()` only
    covers the instance state this module introduced, not the flag
    `instrument()`/`uninstrument()` manage. Safe to call regardless of
    whether the bridge is currently installed (every step is a no-op on
    already-clean state).
    """
    instrumentor = get_langfuse_instrumentor()
    instrumentor._teardown()
    instrumentor._is_instrumented_by_opentelemetry = False


class FakeSpan:
    """Minimal stand-in for `opentelemetry.sdk.trace.Span` + `ReadableSpan`."""

    def __init__(self, attributes: dict[str, AttributeValue | dict[str, int]], scope_name: str = "langfuse-sdk"):
        self._attributes: dict[str, AttributeValue] = cast(Any, dict(attributes))
        scope = MagicMock()
        scope.name = scope_name
        self.instrumentation_scope: MagicMock | None = scope

    @property
    def attributes(self):
        return self._attributes

    def set_attribute(self, key: str, value: AttributeValue):
        self._attributes[key] = value


def silence_langfuse_background_threads(instances: Iterable[Any]):
    """Neutralize the background threads a Langfuse client spins up so they
    don't slow the test suite down.

    Two distinct costs are eliminated:

    1. The OTLP span exporter (would POST to a dead host on every flush) is
       swapped for a no-op. Laminar's own SimpleSpanProcessor already exports
       Langfuse spans synchronously into the InMemorySpanExporter, so the tests
       never need Langfuse's exporter to run.

    2. The prompt-cache refresh thread (`PromptCacheTaskManager`) blocks on a
       HARDCODED 1s `queue.get(timeout=1)` and registers its OWN
       `atexit.register(shutdown)` that `client.shutdown()` never reaches. Left
       alone, every test's manager accumulates an atexit handler, and at
       interpreter teardown each one `join()`s a thread mid-`get()` — ~1s
       apiece, which is the multi-second freeze after the suite finishes. We
       pause the consumers and unregister their atexit hook; the threads are
       daemons, so the interpreter reaps them without a blocking join.
    """
    import atexit


    class _NoopExporter(SpanExporter):
        @override
        def export(self, spans: Sequence[ReadableSpan]):
            return SpanExportResult.SUCCESS

        @override
        def shutdown(self):
            pass

    for instance in instances:
        if getattr(instance, "tracer_provider", None) is not None:
            active = instance.tracer_provider._active_span_processor
            for proc in getattr(active, "_span_processors", ()):
                if hasattr(proc, "_batch_processor"):
                    proc._batch_processor._exporter = _NoopExporter()

        task_manager = getattr(
            getattr(instance, "prompt_cache", None), "_task_manager", None
        )
        if task_manager is not None:
            atexit.unregister(task_manager.shutdown)
            for consumer in task_manager._consumers:
                consumer.pause()


def litellm_langfuse_otel_importable() -> bool:
    try:
        pass
    except Exception:
        return False
    return True


LITELLM_IMPORTABLE = litellm_langfuse_otel_importable()
litellm_required = pytest.mark.skipif(
    not LITELLM_IMPORTABLE,
    reason="litellm (with langfuse_otel) not importable on this interpreter",
)

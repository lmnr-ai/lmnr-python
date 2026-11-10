"""Fixtures for the Langfuse bridge tests."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from typing import Any, cast

import pytest
from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typing_extensions import override

from lmnr.opentelemetry_lib.bridges import Bridge, BridgeContext, get_bridge
from lmnr.opentelemetry_lib.bridges.langfuse import LangfuseBridge
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor

_new_val = os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
_new_val = os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-test")
_new_val = os.environ.setdefault("LANGFUSE_HOST", "http://127.0.0.1:1")
# The media-upload consumer blocks `client.shutdown()` for ~1s per test.
_new_val = os.environ.setdefault("LANGFUSE_MEDIA_UPLOAD_ENABLED", "False")
logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.exporter.otlp.proto.http").setLevel(logging.CRITICAL)


class RecordingExporter(SpanExporter):
    """Stands in for Langfuse's OTLP exporter. Stores attribute snapshots taken
    at export time, so later mutation of a span would show up as a diff."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []

    @override
    def export(self, spans: Any) -> SpanExportResult:
        for span in spans:
            readable: ReadableSpan = span
            self.spans.append(
                {
                    "name": readable.name,
                    "attributes": dict(readable.attributes or {}),
                    "events": [e.name for e in readable.events],
                    "status": readable.status.status_code.name,
                }
            )
        return SpanExportResult.SUCCESS

    @override
    def shutdown(self) -> None:
        pass


class Laminar:
    """Laminar's side of the bridge: its span processor and what it exported."""

    def __init__(self) -> None:
        self.exporter: InMemorySpanExporter = InMemorySpanExporter()
        self.processor: LaminarSpanProcessor = LaminarSpanProcessor(
            api_key="test", exporter=self.exporter, disable_batch=True
        )
        self.provider: TracerProvider = TracerProvider()
        self.provider.add_span_processor(self.processor)
        self.context: BridgeContext = BridgeContext(
            span_processor=self.processor, tracer_provider=self.provider
        )

    def spans(self) -> dict[str, ReadableSpan]:
        return {s.name: s for s in self.exporter.get_finished_spans()}


@pytest.fixture
def laminar() -> Generator[Laminar, None, None]:
    lmnr = Laminar()
    yield lmnr
    lmnr.processor.shutdown()


@pytest.fixture
def bridge() -> Generator[LangfuseBridge, None, None]:
    bridge = get_bridge(Bridge.LANGFUSE)
    assert isinstance(bridge, LangfuseBridge)
    bridge.uninstall()
    yield bridge
    bridge.uninstall()


@pytest.fixture
def langfuse_export() -> RecordingExporter:
    return RecordingExporter()


@pytest.fixture
def langfuse_client(langfuse_export: RecordingExporter) -> Generator[Any, None, None]:
    """A real Langfuse client on its own provider, exporting to
    `langfuse_export` synchronously."""
    LangfuseResourceManager._instances.clear()
    client = Langfuse(tracer_provider=TracerProvider())
    _replace_langfuse_exporter(client, langfuse_export)
    yield client
    client.shutdown()
    LangfuseResourceManager._instances.clear()


def _replace_langfuse_exporter(client: Any, exporter: SpanExporter) -> None:
    """Swap the OTLP exporter inside `LangfuseSpanProcessor` (a
    `BatchSpanProcessor`) for `exporter`. Its batching is kept; tests flush."""
    for rm in LangfuseResourceManager._instances.values():
        active = cast(Any, rm.tracer_provider)._active_span_processor
        for proc in getattr(active, "_span_processors", ()):
            batch = getattr(proc, "_batch_processor", None)
            if batch is not None:
                batch._exporter = exporter
    _ = client

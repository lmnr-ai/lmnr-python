import pytest
from unittest.mock import patch
from lmnr import Laminar
from lmnr.openllmetry_sdk import TracerManager
from lmnr.openllmetry_sdk.tracing.tracing import TracerWrapper
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def exporter() -> SpanExporter:
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    # Set up a partial mock of TracerManager.init to inject our processor
    orig_tracermanager_init = TracerManager.init

    def mock_tracermanager_init(*args, **kwargs):
        kwargs["processor"] = processor
        orig_tracermanager_init(*args, **kwargs)

    with patch(
        "lmnr.openllmetry_sdk.TracerManager.init",
        side_effect=mock_tracermanager_init,
    ):
        Laminar.initialize(
            project_api_key="test_key",
        )

    return exporter


@pytest.fixture(scope="function", autouse=True)
def clear_exporter(exporter: InMemorySpanExporter):
    exporter.clear()
    TracerWrapper.clear()

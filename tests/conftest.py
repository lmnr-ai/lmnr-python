import pytest

from lmnr import Laminar
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    Laminar.initialize(
        project_api_key="test_key",
        _processor=processor,
    )
    return exporter


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()

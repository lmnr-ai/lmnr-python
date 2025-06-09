import pytest
from unittest.mock import patch
from lmnr import Laminar
from lmnr.opentelemetry_lib import TracerManager
from lmnr.opentelemetry_lib.tracing import TracerWrapper
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SpanExporter

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def exporter() -> SpanExporter:
    exporter = InMemorySpanExporter()

    # Set up a partial mock of TracerManager.init to inject our exporter
    orig_tracermanager_init = TracerManager.init

    def mock_tracermanager_init(*args, **kwargs):
        new_kwargs = kwargs.copy()
        new_kwargs["exporter"] = exporter
        orig_tracermanager_init(*args, **new_kwargs)

    with patch(
        "lmnr.opentelemetry_lib.TracerManager.init",
        side_effect=mock_tracermanager_init,
    ):
        Laminar.initialize(
            project_api_key="test_key",
            disable_batch=True,
        )

    return exporter


@pytest.fixture(scope="function", autouse=True)
def clear_exporter(exporter: InMemorySpanExporter):
    exporter.clear()
    TracerWrapper.clear()


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            "authorization",
            "api-key",
            "x-goog-api-key",
        ]
    }

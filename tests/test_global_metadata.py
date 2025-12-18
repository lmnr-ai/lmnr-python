from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.decorators import observe


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Reset Laminar state before each test."""
    # Save the current state
    original_initialized = Laminar._Laminar__initialized
    original_base_http_url = Laminar._Laminar__base_http_url
    original_project_api_key = Laminar._Laminar__project_api_key

    # Reset the initialized state for the test
    Laminar._Laminar__initialized = False
    Laminar._Laminar__base_http_url = None
    Laminar._Laminar__project_api_key = None

    yield

    # Restore the original state after test
    Laminar._Laminar__initialized = original_initialized
    Laminar._Laminar__base_http_url = original_base_http_url
    Laminar._Laminar__project_api_key = original_project_api_key


def test_global_metadata_no_trace_metadata(span_exporter: InMemorySpanExporter):
    Laminar.initialize(project_api_key="test_key", metadata={"foo": "bar"})
    span = Laminar.start_span("test")
    span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert spans[0].name == "test"


def test_global_metadata_no_trace_metadata_start_span_merge(
    span_exporter: InMemorySpanExporter,
):
    Laminar.initialize(
        project_api_key="test_key", metadata={"foo": "bar", "replace": "me"}
    )
    span = Laminar.start_span("test", metadata={"baz": "qux", "replace": "new"})
    span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert spans[0].attributes["lmnr.association.properties.metadata.baz"] == "qux"
    assert spans[0].attributes["lmnr.association.properties.metadata.replace"] == "new"
    assert spans[0].name == "test"


def test_global_metadata_no_trace_metadata_start_as_current_span_merge(
    span_exporter: InMemorySpanExporter,
):
    Laminar.initialize(
        project_api_key="test_key", metadata={"foo": "bar", "replace": "me"}
    )
    with Laminar.start_as_current_span(
        "test", metadata={"baz": "qux", "replace": "new"}
    ):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert spans[0].attributes["lmnr.association.properties.metadata.baz"] == "qux"
    assert spans[0].attributes["lmnr.association.properties.metadata.replace"] == "new"
    assert spans[0].name == "test"


def test_global_metadata_no_trace_metadata_start_active_span_merge(
    span_exporter: InMemorySpanExporter,
):
    Laminar.initialize(
        project_api_key="test_key", metadata={"foo": "bar", "replace": "me"}
    )
    span = Laminar.start_active_span("test", metadata={"baz": "qux", "replace": "new"})
    span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert spans[0].attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert spans[0].attributes["lmnr.association.properties.metadata.baz"] == "qux"
    assert spans[0].attributes["lmnr.association.properties.metadata.replace"] == "new"


def test_global_metadata_no_trace_metadata_observe(span_exporter: InMemorySpanExporter):
    Laminar.initialize(project_api_key="test_key", metadata={"foo": "bar"})

    @observe()
    def test():
        return "test"

    test()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert spans[0].name == "test"


@pytest.mark.asyncio
async def test_global_metadata_no_trace_metadata_observe_async(
    span_exporter: InMemorySpanExporter,
):
    Laminar.initialize(project_api_key="test_key", metadata={"foo": "bar"})

    @observe()
    async def test():
        return "test"

    await test()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert spans[0].name == "test"


def test_global_metadata_no_trace_metadata_two_traces(
    span_exporter: InMemorySpanExporter,
):
    Laminar.initialize(project_api_key="test_key", metadata={"foo": "bar"})

    actual_span = Laminar.start_span("test", metadata={"baz": "qux"})
    actual_span.end()

    actual_span2 = Laminar.start_span("test2")
    actual_span2.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span = [s for s in spans if s.name == "test"][0]
    span2 = [s for s in spans if s.name == "test2"][0]

    assert span.attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert span.attributes["lmnr.association.properties.metadata.baz"] == "qux"

    assert span2.attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert span2.attributes.get("lmnr.association.properties.metadata.baz") is None

    assert span.parent is None or span.parent.span_id == 0
    assert span2.parent is None or span2.parent.span_id == 0
    assert span.get_span_context().trace_id != span2.get_span_context().trace_id

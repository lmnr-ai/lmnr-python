import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.sdk.decorators import observe
from lmnr.sdk.laminar import Laminar


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Reset Laminar state before each test."""
    # Save the current state
    original_initialized = Laminar._Laminar__initialized  # pyright:ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    original_base_http_url = Laminar._Laminar__base_http_url  # pyright:ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    original_project_api_key = Laminar._Laminar__project_api_key  # pyright:ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]

    # Reset the initialized state for the test
    Laminar._Laminar__initialized = False  # pyright:ignore[reportAttributeAccessIssue]
    Laminar._Laminar__base_http_url = None  # pyright:ignore[reportAttributeAccessIssue]
    Laminar._Laminar__project_api_key = None  # pyright:ignore[reportAttributeAccessIssue]

    yield

    # Restore the original state after test
    Laminar._Laminar__initialized = original_initialized  # pyright:ignore[reportAttributeAccessIssue]
    Laminar._Laminar__base_http_url = original_base_http_url  # pyright:ignore[reportAttributeAccessIssue]
    Laminar._Laminar__project_api_key = original_project_api_key  # pyright:ignore[reportAttributeAccessIssue]


def test_global_metadata_no_trace_metadata(span_exporter: InMemorySpanExporter):
    Laminar.initialize(project_api_key="test_key", metadata={"foo": "bar"})
    span = Laminar.start_span("test")
    span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert (spans[0].attributes or {})["lmnr.association.properties.metadata.foo"] == "bar"
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
    attributes = spans[0].attributes or {}
    assert attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert attributes["lmnr.association.properties.metadata.baz"] == "qux"
    assert attributes["lmnr.association.properties.metadata.replace"] == "new"
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
    attributes = spans[0].attributes or {}
    assert attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert attributes["lmnr.association.properties.metadata.baz"] == "qux"
    assert attributes["lmnr.association.properties.metadata.replace"] == "new"
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
    attributes = spans[0].attributes or {}
    assert attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert attributes["lmnr.association.properties.metadata.baz"] == "qux"
    assert attributes["lmnr.association.properties.metadata.replace"] == "new"


def test_global_metadata_no_trace_metadata_observe(span_exporter: InMemorySpanExporter):
    Laminar.initialize(project_api_key="test_key", metadata={"foo": "bar"})

    @observe()
    def test():
        return "test"

    _result = test()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = spans[0].attributes or {}
    assert attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert spans[0].name == "test"


@pytest.mark.asyncio
async def test_global_metadata_no_trace_metadata_observe_async(
    span_exporter: InMemorySpanExporter,
):
    Laminar.initialize(project_api_key="test_key", metadata={"foo": "bar"})

    @observe()
    async def test():
        return "test"

    _result = await test()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = spans[0].attributes or {}
    assert attributes["lmnr.association.properties.metadata.foo"] == "bar"
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
    span = next(s for s in spans if s.name == "test")
    span2 = next(s for s in spans if s.name == "test2")

    attributes = span.attributes or {}
    assert attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert attributes["lmnr.association.properties.metadata.baz"] == "qux"

    span2_attributes = span2.attributes or {}
    assert span2_attributes["lmnr.association.properties.metadata.foo"] == "bar"
    assert span2_attributes.get("lmnr.association.properties.metadata.baz") is None

    assert span.parent is None or span.parent.span_id == 0
    assert span2.parent is None or span2.parent.span_id == 0
    ctx = span.get_span_context()
    ctx2 = span2.get_span_context()
    assert ctx is not None
    assert ctx2 is not None
    assert ctx.trace_id != ctx2.trace_id

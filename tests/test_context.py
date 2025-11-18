from opentelemetry.trace.span import INVALID_SPAN_ID
from lmnr import Laminar, observe
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_clear_context_observe(span_exporter: InMemorySpanExporter):
    @observe()
    def inner():
        Laminar.set_trace_user_id("test_user_id_2")
        return "inner"

    @observe()
    def outer():
        Laminar.set_trace_user_id("test_user_id_1")
        Laminar.force_flush()
        return inner()

    outer()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    outer_span = [s for s in spans if s.name == "outer"][0]
    inner_span = [s for s in spans if s.name == "inner"][0]
    assert (
        outer_span.attributes["lmnr.association.properties.user_id"] == "test_user_id_1"
    )
    assert (
        inner_span.attributes["lmnr.association.properties.user_id"] == "test_user_id_2"
    )

    assert inner_span.parent is None or inner_span.parent.span_id == INVALID_SPAN_ID
    assert (
        inner_span.get_span_context().trace_id != outer_span.get_span_context().trace_id
    )


def test_clear_context_start_as_current_span(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("outer"):
        Laminar.set_trace_user_id("test_user_id_1")
        Laminar.force_flush()
        with Laminar.start_as_current_span("inner"):
            Laminar.set_trace_user_id("test_user_id_2")
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    outer_span = [s for s in spans if s.name == "outer"][0]
    inner_span = [s for s in spans if s.name == "inner"][0]
    assert (
        outer_span.attributes["lmnr.association.properties.user_id"] == "test_user_id_1"
    )
    assert (
        inner_span.attributes["lmnr.association.properties.user_id"] == "test_user_id_2"
    )

    assert inner_span.parent is None or inner_span.parent.span_id == INVALID_SPAN_ID
    assert (
        inner_span.get_span_context().trace_id != outer_span.get_span_context().trace_id
    )


def test_clear_context_start_active_span(span_exporter: InMemorySpanExporter):
    span = Laminar.start_active_span("outer")
    Laminar.set_trace_user_id("test_user_id_1")
    Laminar.force_flush()
    span2 = Laminar.start_active_span("inner")
    Laminar.set_trace_user_id("test_user_id_2")
    span2.end()
    span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    outer_span = [s for s in spans if s.name == "outer"][0]
    inner_span = [s for s in spans if s.name == "inner"][0]
    assert (
        outer_span.attributes["lmnr.association.properties.user_id"] == "test_user_id_1"
    )
    assert (
        inner_span.attributes["lmnr.association.properties.user_id"] == "test_user_id_2"
    )

    assert inner_span.parent is None or inner_span.parent.span_id == INVALID_SPAN_ID
    assert (
        inner_span.get_span_context().trace_id != outer_span.get_span_context().trace_id
    )

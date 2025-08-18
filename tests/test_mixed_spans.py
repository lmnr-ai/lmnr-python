import json

from lmnr import Laminar, observe
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_start_as_current_span_inside_observe(span_exporter: InMemorySpanExporter):
    @observe()
    def foo():
        with Laminar.start_as_current_span("test", input="my_input"):
            Laminar.set_span_output("foo")
            pass

    foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    outer_span = next(span for span in spans if span.name == "foo")
    inner_span = next(span for span in spans if span.name == "test")
    assert json.loads(inner_span.attributes["lmnr.span.output"]) == "foo"
    assert json.loads(inner_span.attributes["lmnr.span.input"]) == "my_input"
    assert (
        inner_span.get_span_context().trace_id == outer_span.get_span_context().trace_id
    )
    assert inner_span.parent.span_id == outer_span.get_span_context().span_id
    assert outer_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert outer_span.attributes["lmnr.span.path"] == ("foo",)
    assert inner_span.attributes["lmnr.span.path"] == ("foo", "test")

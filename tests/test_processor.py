from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from lmnr.sdk.decorators import observe


def test_span_processor_cleanup(span_exporter: InMemorySpanExporter):
    processor: LaminarSpanProcessor = TracerWrapper()._span_processor

    @observe()
    def foo():
        assert processor._LaminarSpanProcessor__span_id_lists  # not empty
        assert processor._LaminarSpanProcessor__span_id_to_path  # not empty
        return "bar"

    foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert not processor._LaminarSpanProcessor__span_id_lists
    assert not processor._LaminarSpanProcessor__span_id_to_path

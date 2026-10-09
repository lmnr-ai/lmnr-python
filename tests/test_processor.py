from typing import cast

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.tracing import get_tracer_wrapper
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from lmnr.sdk.decorators import observe


def test_span_processor_cleanup(span_exporter: InMemorySpanExporter):
    wrapper = get_tracer_wrapper()
    assert wrapper is not None
    processor = cast(LaminarSpanProcessor, wrapper.span_processor)

    @observe()
    def foo() -> str:
        assert processor._LaminarSpanProcessor__span_id_lists  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        assert processor._LaminarSpanProcessor__span_id_to_path  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        return "bar"

    _bar = foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert not processor._LaminarSpanProcessor__span_id_lists  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert not processor._LaminarSpanProcessor__span_id_to_path  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

from datetime import datetime
import time
import uuid

from opentelemetry import trace, context
from opentelemetry.trace import Span, SpanContext, NonRecordingSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from lmnr import Laminar, observe


TRACE_ID = 369
SPAN_ID = 963


class MyBrokenSpanContext(SpanContext):
    @property
    def trace_flags(self) -> int:
        return 1


def test_broken_span(exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("outer"):
        trace_id = trace.get_current_span().get_span_context().trace_id
        span = NonRecordingSpan(MyBrokenSpanContext(trace_id, SPAN_ID, False))
        ctx = trace.set_span_in_context(span, context.get_current())
        ctx_token = context.attach(ctx)

        with Laminar.start_as_current_span("inner"):
            pass
        pass

    span.end()
    time.sleep(0.5)

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    inner_span = [span for span in spans if span.name == "inner"][0]
    outer_span = [span for span in spans if span.name == "outer"][0]
    assert outer_span.name == "outer"
    assert outer_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert outer_span.attributes["lmnr.span.path"] == ("outer",)
    assert outer_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_span.get_span_context().span_id)),
    )

    assert (
        inner_span.get_span_context().trace_id == outer_span.get_span_context().trace_id
    )

    context.detach(ctx_token)


def test_broken_span_observe(exporter: InMemorySpanExporter):
    @observe()
    def test():
        return 1

    @observe()
    def outer():
        trace_id = trace.get_current_span().get_span_context().trace_id
        span = NonRecordingSpan(MyBrokenSpanContext(trace_id, SPAN_ID, False))
        ctx = trace.set_span_in_context(span, context.get_current())
        context.attach(ctx)

        result = test()
        return result

    result = outer()
    time.sleep(0.5)
    assert result == 1

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    inner_span = [span for span in spans if span.name == "test"][0]
    outer_span = [span for span in spans if span.name == "outer"][0]
    assert inner_span.name == "test"
    assert inner_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert inner_span.attributes["lmnr.span.path"] == ("test",)
    assert inner_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=inner_span.get_span_context().span_id)),
    )
    assert inner_span.parent.span_id == SPAN_ID
    assert inner_span.parent.trace_flags.sampled

    assert outer_span.name == "outer"
    assert outer_span.attributes["lmnr.span.instrumentation_source"] == "python"
    assert outer_span.attributes["lmnr.span.path"] == ("outer",)
    assert outer_span.attributes["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_span.get_span_context().span_id)),
    )

    assert (
        inner_span.get_span_context().trace_id == outer_span.get_span_context().trace_id
    )

import threading
import time

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext

from lmnr import Laminar, observe


def _ctx(span: ReadableSpan) -> SpanContext:
    ctx = span.get_span_context()
    assert ctx is not None
    return ctx


def _parent(span: ReadableSpan) -> SpanContext:
    assert span.parent is not None
    return span.parent


def sleep_and_print(do_observe: bool = False, span_name: str = "sleep_and_print"):
    if do_observe:
        with Laminar.start_as_current_span(span_name):
            time.sleep(0.2)
            print("done")
    else:

        time.sleep(0.2)
        print("done")


def test_threading_works_on_start():
    t = threading.Thread(target=sleep_and_print)
    t.start()
    assert hasattr(t, "_lmnr_otel_context")
    t.join()


def test_threading_works_on_run():
    t = threading.Thread(target=sleep_and_print)
    t.run()
    assert hasattr(t, "_lmnr_otel_context")


def test_threading_start_preserves_context(span_exporter: InMemorySpanExporter):
    @observe()
    def parent():
        t1 = threading.Thread(
            target=sleep_and_print, kwargs={"do_observe": True, "span_name": "t1"}
        )
        t2 = threading.Thread(
            target=sleep_and_print, kwargs={"do_observe": True, "span_name": "t2"}
        )
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    @observe()
    def sibling():
        t = threading.Thread(
            target=sleep_and_print,
            kwargs={"do_observe": True, "span_name": "thread_sibling"},
        )
        t.start()
        t.join()
        assert hasattr(t, "_lmnr_otel_context")

    parent()
    sibling()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 5
    parent_span = next(s for s in spans if s.name == "parent")
    sibling_span = next(s for s in spans if s.name == "sibling")
    t1_span = next(s for s in spans if s.name == "t1")
    t2_span = next(s for s in spans if s.name == "t2")
    thread_sibling_span = next(s for s in spans if s.name == "thread_sibling")

    assert _parent(t1_span).span_id == _ctx(parent_span).span_id
    assert _parent(t2_span).span_id == _ctx(parent_span).span_id
    assert (
        _ctx(t1_span).trace_id == _ctx(parent_span).trace_id
    )
    assert (
        _ctx(t2_span).trace_id == _ctx(parent_span).trace_id
    )

    assert (
        _ctx(sibling_span).trace_id
        != _ctx(parent_span).trace_id
    )
    assert (
        _ctx(thread_sibling_span).trace_id
        == _ctx(sibling_span).trace_id
    )
    assert _parent(thread_sibling_span).span_id == _ctx(sibling_span).span_id


def test_threading_run_preserves_context(span_exporter: InMemorySpanExporter):
    @observe()
    def parent():
        t1 = threading.Thread(
            target=sleep_and_print, kwargs={"do_observe": True, "span_name": "t1"}
        )
        t2 = threading.Thread(
            target=sleep_and_print, kwargs={"do_observe": True, "span_name": "t2"}
        )
        t1.run()
        t2.run()

    @observe()
    def sibling():
        t = threading.Thread(
            target=sleep_and_print,
            kwargs={"do_observe": True, "span_name": "thread_sibling"},
        )
        t.run()
        assert hasattr(t, "_lmnr_otel_context")

    parent()
    sibling()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 5
    parent_span = next(s for s in spans if s.name == "parent")
    sibling_span = next(s for s in spans if s.name == "sibling")
    t1_span = next(s for s in spans if s.name == "t1")
    t2_span = next(s for s in spans if s.name == "t2")
    thread_sibling_span = next(s for s in spans if s.name == "thread_sibling")

    assert _parent(t1_span).span_id == _ctx(parent_span).span_id
    assert _parent(t2_span).span_id == _ctx(parent_span).span_id
    assert (
        _ctx(t1_span).trace_id == _ctx(parent_span).trace_id
    )
    assert (
        _ctx(t2_span).trace_id == _ctx(parent_span).trace_id
    )

    assert (
        _ctx(sibling_span).trace_id
        != _ctx(parent_span).trace_id
    )
    assert (
        _ctx(thread_sibling_span).trace_id
        == _ctx(sibling_span).trace_id
    )
    assert _parent(thread_sibling_span).span_id == _ctx(sibling_span).span_id

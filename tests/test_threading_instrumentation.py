import threading
import time
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr import Laminar, observe


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
    parent_span = [s for s in spans if s.name == "parent"][0]
    sibling_span = [s for s in spans if s.name == "sibling"][0]
    t1_span = [s for s in spans if s.name == "t1"][0]
    t2_span = [s for s in spans if s.name == "t2"][0]
    thread_sibling_span = [s for s in spans if s.name == "thread_sibling"][0]

    assert t1_span.parent.span_id == parent_span.get_span_context().span_id
    assert t2_span.parent.span_id == parent_span.get_span_context().span_id
    assert (
        t1_span.get_span_context().trace_id == parent_span.get_span_context().trace_id
    )
    assert (
        t2_span.get_span_context().trace_id == parent_span.get_span_context().trace_id
    )

    assert (
        sibling_span.get_span_context().trace_id
        != parent_span.get_span_context().trace_id
    )
    assert (
        thread_sibling_span.get_span_context().trace_id
        == sibling_span.get_span_context().trace_id
    )
    assert thread_sibling_span.parent.span_id == sibling_span.get_span_context().span_id


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
    parent_span = [s for s in spans if s.name == "parent"][0]
    sibling_span = [s for s in spans if s.name == "sibling"][0]
    t1_span = [s for s in spans if s.name == "t1"][0]
    t2_span = [s for s in spans if s.name == "t2"][0]
    thread_sibling_span = [s for s in spans if s.name == "thread_sibling"][0]

    assert t1_span.parent.span_id == parent_span.get_span_context().span_id
    assert t2_span.parent.span_id == parent_span.get_span_context().span_id
    assert (
        t1_span.get_span_context().trace_id == parent_span.get_span_context().trace_id
    )
    assert (
        t2_span.get_span_context().trace_id == parent_span.get_span_context().trace_id
    )

    assert (
        sibling_span.get_span_context().trace_id
        != parent_span.get_span_context().trace_id
    )
    assert (
        thread_sibling_span.get_span_context().trace_id
        == sibling_span.get_span_context().trace_id
    )
    assert thread_sibling_span.parent.span_id == sibling_span.get_span_context().span_id

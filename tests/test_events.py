import pytest

from lmnr import Laminar, observe
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode


def test_simple_event(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        Laminar.event("test_event")

    observed_foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "test_event"


def test_event_outside_span_creates_span(span_exporter: InMemorySpanExporter):
    def foo():
        Laminar.event("test_event")

    foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test_event"
    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "test_event"


def test_event_with_attributes(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        Laminar.event("test_event", attributes={"key": "value"})

    observed_foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "test_event"
    assert events[0].attributes == {"key": "value"}


def test_event_with_attributes_and_session_id(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        Laminar.set_trace_session_id("test_session_id")
        Laminar.set_trace_user_id("test_user_id")
        Laminar.event("test_event", attributes={"key": "value"})

    observed_foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "test_event"
    assert events[0].attributes == {
        "key": "value",
        "lmnr.event.session_id": "test_session_id",
        "lmnr.event.user_id": "test_user_id",
    }


def test_event_overrides_session_id_and_user_id(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        Laminar.set_trace_session_id("test_session_id")
        Laminar.set_trace_user_id("test_user_id")
        Laminar.event(
            "test_event",
            attributes={"key": "value"},
            user_id="another_user_id",
            session_id="another_session_id",
        )

    observed_foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "test_event"
    assert events[0].attributes == {
        "key": "value",
        "lmnr.event.session_id": "another_session_id",
        "lmnr.event.user_id": "another_user_id",
    }


def test_event_with_attributes_and_session_id_observe(
    span_exporter: InMemorySpanExporter,
):
    @observe(session_id="test_session_id", user_id="test_user_id")
    def observed_foo():
        Laminar.event("test_event", attributes={"key": "value"})

    observed_foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "test_event"
    assert events[0].attributes == {
        "key": "value",
        "lmnr.event.session_id": "test_session_id",
        "lmnr.event.user_id": "test_user_id",
    }


def test_event_with_attributes_and_session_id_different_span(
    span_exporter: InMemorySpanExporter,
):
    @observe()
    def inner():
        Laminar.event("test_event", attributes={"key": "value"})

    @observe()
    def observed_foo():
        Laminar.set_trace_session_id("test_session_id")
        Laminar.set_trace_user_id("test_user_id")
        inner()

    observed_foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    inner_span = next(span for span in spans if span.name == "inner")
    assert len(inner_span.events) == 1
    assert inner_span.events[0].name == "test_event"
    assert inner_span.events[0].attributes == {
        "key": "value",
        "lmnr.event.session_id": "test_session_id",
        "lmnr.event.user_id": "test_user_id",
    }


def test_event_with_attributes_and_session_id_different_trace(
    span_exporter: InMemorySpanExporter,
):
    @observe()
    def observed_bar():
        Laminar.event("test_event", attributes={"key": "value"})

    @observe()
    def observed_foo():
        Laminar.set_trace_session_id("test_session_id")
        Laminar.set_trace_user_id("test_user_id")

    observed_foo()
    observed_bar()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    foo_span = next(span for span in spans if span.name == "observed_foo")
    bar_span = next(span for span in spans if span.name == "observed_bar")
    assert len(foo_span.events) == 0
    assert len(bar_span.events) == 1
    assert bar_span.events[0].name == "test_event"
    assert bar_span.events[0].attributes == {
        "key": "value",
    }


def test_exception_with_attributes_and_session_id_observe(
    span_exporter: InMemorySpanExporter,
):
    @observe(session_id="test_session_id", user_id="test_user_id")
    def observed_foo():
        raise ValueError("test_error")

    with pytest.raises(ValueError, match="test_error"):
        observed_foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].status.status_code == StatusCode.ERROR
    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert events[0].attributes.get("exception.type") == "ValueError"
    assert events[0].attributes.get("exception.message") == "test_error"
    assert events[0].attributes.get("exception.stacktrace") is not None
    assert events[0].attributes.get("lmnr.event.session_id") == "test_session_id"
    assert events[0].attributes.get("lmnr.event.user_id") == "test_user_id"


def test_exception_with_attributes_and_session_id(
    span_exporter: InMemorySpanExporter,
):
    @observe()
    def observed_foo():
        Laminar.set_trace_session_id("test_session_id")
        Laminar.set_trace_user_id("test_user_id")
        raise ValueError("test_error")

    with pytest.raises(ValueError, match="test_error"):
        observed_foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert spans[0].status.status_code == StatusCode.ERROR
    events = spans[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert events[0].attributes.get("exception.type") == "ValueError"
    assert events[0].attributes.get("exception.message") == "test_error"
    assert events[0].attributes.get("exception.stacktrace") is not None
    assert events[0].attributes.get("lmnr.event.session_id") == "test_session_id"
    assert events[0].attributes.get("lmnr.event.user_id") == "test_user_id"

import pytest

from lmnr import Laminar
from lmnr.opentelemetry_lib.tracing.span import LaminarSpan


@pytest.fixture
def uninitialized(monkeypatch):
    monkeypatch.setattr(Laminar, "is_initialized", classmethod(lambda cls: False))


def _exercise(span):
    assert isinstance(span, LaminarSpan)
    assert not span.is_recording()
    span.set_input({"a": 1})
    span.set_output("x")
    span.set_tags(["t"])
    span.add_tags(["u"])
    span.set_trace_session_id("s")
    span.set_trace_user_id("u")
    span.set_trace_metadata({"k": "v"})
    span.set_attribute("k", "v")
    assert span.name == ""
    assert span.attributes == {}
    assert span.get_span_context().trace_id != 0
    span.end()


def test_start_span(uninitialized):
    _exercise(Laminar.start_span("x"))


def test_start_active_span(uninitialized):
    _exercise(Laminar.start_active_span("x"))


def test_start_as_current_span(uninitialized):
    with Laminar.start_as_current_span("x") as span:
        _exercise(span)


def test_use_span(uninitialized):
    with Laminar.use_span(Laminar.start_span("x"), end_on_exit=True) as span:
        _exercise(span)

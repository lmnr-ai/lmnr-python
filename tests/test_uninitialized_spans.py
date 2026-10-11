from collections.abc import Mapping
from typing import Any, cast

import pytest
from opentelemetry.sdk.trace import Span as SDKSpan
from opentelemetry.trace import Span

from lmnr import Laminar
from lmnr.opentelemetry_lib.tracing.span import LaminarSpan


@pytest.fixture
def uninitialized(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Laminar, "is_initialized", classmethod(lambda cls: False))


def _exercise(span: Span):
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


def test_start_span(uninitialized: None):
    _exercise(Laminar.start_span("x"))


def test_start_active_span(uninitialized: None):
    _exercise(Laminar.start_active_span("x"))


def test_start_as_current_span(uninitialized: None):
    with Laminar.start_as_current_span("x") as span:
        _exercise(span)


def test_use_span(uninitialized: None):
    with Laminar.use_span(Laminar.start_span("x"), end_on_exit=True) as span:
        _exercise(span)


def test_use_span_wraps_the_original_sdk_span(uninitialized: None):
    from opentelemetry.sdk.trace import TracerProvider

    original = TracerProvider().get_tracer("t").start_span("orig")
    with Laminar.use_span(original) as span:
        assert isinstance(span, LaminarSpan)
        assert span.span is original
        span.set_attribute("k", "v")
    assert (cast(SDKSpan, original).attributes or {})["k"] == "v"
    original.end()


def test_use_span_forwards_writes_to_a_foreign_span(uninitialized: None):
    from opentelemetry.trace import Span, SpanContext

    class Foreign(Span):
        def __init__(self):
            self.attrs: dict[str, Any] = {}
            self.ctx: SpanContext = SpanContext(1, 2, False)

        def get_span_context(self) -> SpanContext:
            return self.ctx

        def set_attribute(self, key: str, value: Any):
            self.attrs[key] = value

        def set_attributes(self, attributes: Mapping[str, Any]):
            self.attrs.update(attributes)

        def add_event(self, *a: Any, **k: Any): ...
        def add_link(self, *a: Any, **k: Any): ...
        def update_name(self, name: str): ...
        def is_recording(self) -> bool:
            return True

        def end(self, end_time: int | None = None): ...
        def set_status(self, *a: Any, **k: Any): ...
        def record_exception(self, *a: Any, **k: Any): ...

    foreign = Foreign()
    with Laminar.use_span(foreign) as span:
        assert isinstance(span, LaminarSpan)
        assert span.is_recording()
        span.set_attribute("k", "v")
    assert foreign.attrs == {"k": "v"}

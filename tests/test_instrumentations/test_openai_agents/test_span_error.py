"""Error propagation from Agents SDK spans to Laminar spans.

app-server sets a span's status to `error` based on the presence of an
`exception` event, not on the OTel status code, so `set_status` alone leaves a
failed tool call rendered as a success.
"""

import json

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.span_data import (
    apply_span_error,
)


class _FakeAgentsSpan:
    """Stands in for an `agents.tracing.Span` carrying a `SpanError`."""

    def __init__(self, error):
        self.error = error


def _exception_events(span):
    return [e for e in span.events if e.name == "exception"]


def _only_span(span_exporter, name):
    return next(s for s in span_exporter.get_finished_spans() if s.name == name)


def test_span_error_dict_emits_exception_event(span_exporter):
    span_exporter.clear()

    # The shape the Agents SDK actually produces: `SpanError` is a TypedDict,
    # so both fields are dict keys at runtime.
    error = {
        "message": "Error running tool (non-fatal)",
        "data": {"tool_name": "lookup_paper", "error": "MCP tool failure"},
    }

    with Laminar.start_as_current_span(name="tool-that-failed") as span:
        apply_span_error(span, _FakeAgentsSpan(error))

    events = _exception_events(_only_span(span_exporter, "tool-that-failed"))
    assert len(events) == 1
    attributes = dict(events[0].attributes)
    assert attributes["exception.type"] == "Error running tool (non-fatal)"
    assert json.loads(attributes["exception.message"]) == error["data"]


def test_span_error_object_emits_exception_event(span_exporter):
    """Newer SDKs may hand us an object rather than a plain dict."""
    span_exporter.clear()

    class _ObjectError:
        message = "Max turns exceeded"
        data = None

    with Laminar.start_as_current_span(name="agent-that-failed") as span:
        apply_span_error(span, _FakeAgentsSpan(_ObjectError()))

    events = _exception_events(_only_span(span_exporter, "agent-that-failed"))
    assert len(events) == 1
    attributes = dict(events[0].attributes)
    assert attributes["exception.type"] == "Max turns exceeded"
    # With no `data` the label doubles as the detail rather than being dropped.
    assert attributes["exception.message"] == "Max turns exceeded"


def test_span_without_error_emits_no_event(span_exporter):
    span_exporter.clear()

    with Laminar.start_as_current_span(name="tool-that-succeeded") as span:
        apply_span_error(span, _FakeAgentsSpan(None))

    assert _exception_events(_only_span(span_exporter, "tool-that-succeeded")) == []

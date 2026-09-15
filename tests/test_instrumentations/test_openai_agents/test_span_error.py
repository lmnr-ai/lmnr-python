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
    # Reported through `record_exception`, like every other instrumentor —
    # `exception.escaped` only comes from that path.
    assert attributes["exception.escaped"] == "False"
    # Nothing was raised, so no misleading synthetic stacktrace.
    assert attributes["exception.stacktrace"] == ""


def test_span_error_event_carries_context_attributes(span_exporter):
    """`record_exception` is called with `get_event_attributes_from_context()`."""
    span_exporter.clear()

    with Laminar.start_as_current_span(
        name="tool-with-session", session_id="session-123", user_id="user-456"
    ) as span:
        apply_span_error(span, _FakeAgentsSpan({"message": "Max turns exceeded"}))

    events = _exception_events(_only_span(span_exporter, "tool-with-session"))
    attributes = dict(events[0].attributes)
    assert attributes["lmnr.event.session_id"] == "session-123"
    assert attributes["lmnr.event.user_id"] == "user-456"


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

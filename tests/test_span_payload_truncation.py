"""Oversized span payloads must be truncated at the boundary, not replaced (LAM-2050)."""

from lmnr import Laminar
from lmnr.opentelemetry_lib.tracing.attributes import SPAN_INPUT, SPAN_OUTPUT
from lmnr.opentelemetry_lib.tracing.span import (
    MAX_MANUAL_SPAN_PAYLOAD_SIZE,
    TRUNCATION_SUFFIX,
)


def test_oversized_input_is_truncated_not_replaced(span_exporter):
    oversized = "a" * (MAX_MANUAL_SPAN_PAYLOAD_SIZE + 1000)
    with Laminar.start_as_current_span("test") as span:
        span.set_input({"blob": oversized})

    recorded = span_exporter.get_finished_spans()[0].attributes[SPAN_INPUT]
    assert len(recorded) == MAX_MANUAL_SPAN_PAYLOAD_SIZE
    assert recorded.endswith(TRUNCATION_SUFFIX)
    # The leading bytes of the real payload survive — the whole point of truncating.
    assert recorded.startswith('{"blob":"aaa')
    assert "too large to record" not in recorded


def test_oversized_output_is_truncated_not_replaced(span_exporter):
    oversized = "b" * (MAX_MANUAL_SPAN_PAYLOAD_SIZE + 1000)
    with Laminar.start_as_current_span("test") as span:
        span.set_output({"blob": oversized})

    recorded = span_exporter.get_finished_spans()[0].attributes[SPAN_OUTPUT]
    assert len(recorded) == MAX_MANUAL_SPAN_PAYLOAD_SIZE
    assert recorded.endswith(TRUNCATION_SUFFIX)
    assert "too large to record" not in recorded


def test_payload_at_the_limit_is_untouched(span_exporter):
    with Laminar.start_as_current_span("test") as span:
        span.set_input("x")

    recorded = span_exporter.get_finished_spans()[0].attributes[SPAN_INPUT]
    assert recorded == '"x"'
    assert TRUNCATION_SUFFIX not in recorded

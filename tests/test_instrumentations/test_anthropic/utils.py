from __future__ import annotations

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def successful_spans(span_exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    """The finished spans, minus the one from the test's intentional bad call.

    `messages.create(unknown_parameter=...)` raises a TypeError inside the SDK.
    Its span must still end, with the `exception` event app-server reads as
    the error marker.
    """
    spans = span_exporter.get_finished_spans()
    failed = [span for span in spans if any(e.name == "exception" for e in span.events)]
    assert [span.name for span in failed] == ["anthropic.chat"]
    assert (failed[0].attributes or {}).get("error.type") == "TypeError"
    return [span for span in spans if span not in failed]

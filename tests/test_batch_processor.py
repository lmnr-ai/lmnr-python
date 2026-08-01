"""Tests for the size-limited batch span processor.

These build their own `TracerProvider` + recording exporter rather than using
the session-scoped `span_exporter` fixture, because they need to control
`max_export_batch_size` / `schedule_delay_millis` to isolate the byte-based
flush trigger from the count- and time-based ones.
"""

import threading

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from lmnr.opentelemetry_lib.tracing.batch_processor import (
    DEFAULT_MAX_EXPORT_BATCH_SIZE,
    DEFAULT_MAX_EXPORT_BATCH_SIZE_BYTES,
    SizeLimitedBatchSpanProcessor,
    approximate_span_size,
)
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor

# Long enough that the schedule-delay trigger can never fire during a test.
_NEVER_MILLIS = 600_000


class RecordingExporter(SpanExporter):
    def __init__(self):
        self.batches: list[list] = []
        self._lock = threading.Lock()

    def export(self, spans) -> SpanExportResult:
        with self._lock:
            self.batches.append(list(spans))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    @property
    def batch_sizes(self) -> list[int]:
        with self._lock:
            return [len(batch) for batch in self.batches]

    @property
    def span_count(self) -> int:
        with self._lock:
            return sum(len(batch) for batch in self.batches)


def _make_processor(
    max_export_batch_size_bytes: int,
    max_export_batch_size: int = 1000,
):
    """A processor whose ONLY reachable flush trigger is the byte limit."""
    exporter = RecordingExporter()
    processor = SizeLimitedBatchSpanProcessor(
        exporter,
        max_export_batch_size=max_export_batch_size,
        max_queue_size=2048,
        schedule_delay_millis=_NEVER_MILLIS,
        max_export_batch_size_bytes=max_export_batch_size_bytes,
    )
    provider = TracerProvider()
    provider.add_span_processor(processor)
    return exporter, processor, provider.get_tracer(__name__)


def _emit(tracer, name: str, payload_size: int = 0):
    with tracer.start_as_current_span(name) as span:
        if payload_size:
            span.set_attribute("gen_ai.input.messages", "x" * payload_size)


def test_byte_limit_flushes_before_count_and_time_limits():
    # 4 KiB-ish spans against a 10 KB limit: every third span trips it.
    exporter, processor, tracer = _make_processor(max_export_batch_size_bytes=10_000)

    for i in range(10):
        _emit(tracer, f"span-{i}", payload_size=4000)

    # Without the byte limit nothing would have been exported yet: the item
    # limit is 1000 and the schedule delay is 10 minutes.
    assert exporter.batch_sizes == [2, 2, 2, 2]

    processor.force_flush()
    assert exporter.span_count == 10


def test_small_spans_never_trip_the_byte_limit():
    exporter, processor, tracer = _make_processor(
        max_export_batch_size_bytes=DEFAULT_MAX_EXPORT_BATCH_SIZE_BYTES
    )

    for i in range(200):
        _emit(tracer, f"span-{i}", payload_size=5)

    assert exporter.batch_sizes == []

    processor.force_flush()
    assert exporter.span_count == 200


def test_span_larger_than_the_whole_limit_is_exported_alone():
    # A single span over the limit must not wedge the processor: it is enqueued
    # after the preceding batch flushes, and the next span flushes it in turn.
    exporter, processor, tracer = _make_processor(max_export_batch_size_bytes=1000)

    for i in range(3):
        _emit(tracer, f"huge-{i}", payload_size=50_000)

    processor.force_flush()
    assert exporter.batch_sizes == [1, 1, 1]


def test_pending_size_resyncs_after_an_external_flush():
    # The worker thread and force_flush() both drain the queue without telling
    # us. An emptied queue must reset the running total, or a stale total would
    # flush a nearly-empty buffer on the very next span.
    exporter, processor, tracer = _make_processor(max_export_batch_size_bytes=10_000)

    _emit(tracer, "first", payload_size=9000)
    processor.force_flush()
    assert exporter.batch_sizes == [1]

    _emit(tracer, "second", payload_size=9000)
    assert exporter.batch_sizes == [1], "stale pending total forced an early flush"

    processor.force_flush()
    assert exporter.batch_sizes == [1, 1]


def test_unsampled_spans_are_not_counted():
    exporter, processor, _ = _make_processor(max_export_batch_size_bytes=1000)

    class _Unsampled:
        name = "unsampled"
        attributes = {"gen_ai.input.messages": "x" * 50_000}
        events = ()
        links = ()
        status = None

        class context:
            class trace_flags:
                sampled = False

    processor.on_end(_Unsampled())

    assert exporter.batch_sizes == []
    assert processor._pending_size_bytes == 0


def test_shutdown_exports_everything_buffered():
    exporter, processor, tracer = _make_processor(max_export_batch_size_bytes=10_000)

    for i in range(5):
        _emit(tracer, f"span-{i}", payload_size=4000)

    processor.shutdown()
    assert exporter.span_count == 5


def test_concurrent_emitters_lose_no_spans():
    exporter, processor, tracer = _make_processor(max_export_batch_size_bytes=20_000)

    def worker(worker_id: int):
        for i in range(50):
            _emit(tracer, f"w{worker_id}-{i}", payload_size=2000)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    processor.force_flush()
    assert exporter.span_count == 8 * 50


def test_approximate_span_size_counts_attribute_keys_and_values():
    _, _, tracer = _make_processor(max_export_batch_size_bytes=10**9)

    with tracer.start_as_current_span("named") as span:
        span.set_attribute("k", "v" * 100)
        span.set_attribute("count", 7)
        span.set_attribute("tags", ["ab", "cd"])
        readable = span._readable_span()

    size = approximate_span_size(readable)
    # "named"(5) + "k"(1) + 100 + "count"(5) + 8 + "tags"(4) + 4
    assert size == 127


def test_approximate_span_size_includes_events_and_status_description():
    _, _, tracer = _make_processor(max_export_batch_size_bytes=10**9)

    with tracer.start_as_current_span("s") as span:
        span.add_event("boom", attributes={"detail": "d" * 50})
        readable = span._readable_span()

    # "s"(1) + "boom"(4) + "detail"(6) + 50
    assert approximate_span_size(readable) == 61


def test_laminar_span_processor_uses_the_size_limited_batch_processor():
    processor = LaminarSpanProcessor(
        base_url="https://api.lmnr.ai",
        api_key="test",
        exporter=RecordingExporter(),
    )

    assert isinstance(processor.instance, SizeLimitedBatchSpanProcessor)
    assert (
        processor.instance._max_export_batch_size_bytes
        == DEFAULT_MAX_EXPORT_BATCH_SIZE_BYTES
    )
    # The 64-span default is Laminar's, applied here rather than left to OTel's
    # own default of 512.
    assert (
        processor.instance._batch_processor._max_export_batch_size
        == DEFAULT_MAX_EXPORT_BATCH_SIZE
    )


def test_laminar_span_processor_forwards_both_limits():
    processor = LaminarSpanProcessor(
        base_url="https://api.lmnr.ai",
        api_key="test",
        exporter=RecordingExporter(),
        max_export_batch_size=7,
        max_export_batch_size_bytes=1234,
    )

    assert processor.instance._max_export_batch_size_bytes == 1234
    assert processor.instance._batch_processor._max_export_batch_size == 7


def test_disable_batch_still_uses_the_simple_processor():
    processor = LaminarSpanProcessor(
        base_url="https://api.lmnr.ai",
        api_key="test",
        exporter=RecordingExporter(),
        disable_batch=True,
    )

    assert isinstance(processor.instance, SimpleSpanProcessor)


def test_force_reinit_preserves_both_limits():
    processor = LaminarSpanProcessor(
        base_url="https://api.lmnr.ai",
        api_key="test",
        max_export_batch_size=7,
        max_export_batch_size_bytes=1234,
    )

    processor.force_reinit()

    assert isinstance(processor.instance, SizeLimitedBatchSpanProcessor)
    assert processor.instance._max_export_batch_size_bytes == 1234
    assert processor.instance._batch_processor._max_export_batch_size == 7

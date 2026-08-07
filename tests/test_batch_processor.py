"""Tests for the size-limited batch span processor.

The size trigger is opt-in (`Laminar.initialize(flush_by_size=True)`), so the
`LaminarSpanProcessor` wiring tests below cover both transports: the plain
upstream `BatchSpanProcessor` by default and `SizeLimitedBatchSpanProcessor`
when the flag is set.

The behavioral tests build their own `TracerProvider` + recording exporter
rather than using the session-scoped `span_exporter` fixture, because they need
to control `max_export_batch_size` / `schedule_delay_millis` to isolate the
byte-based flush trigger from the count- and time-based ones.
"""

import threading
import time

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from lmnr.opentelemetry_lib.tracing.batch_processor import (
    DEFAULT_MAX_EXPORT_BATCH_SIZE,
    DEFAULT_MAX_EXPORT_BATCH_SIZE_BYTES,
    SizeLimitedBatchSpanProcessor,
    approximate_span_size,
    utf8_size,
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


_built_processors: list = []


@pytest.fixture(autouse=True)
def _shutdown_processors():
    """Shut down every processor a test builds.

    Every batch processor starts a daemon worker thread that lives until
    shutdown. Without this, each test leaks one and a dozen threads keep
    exporting on their own schedules for the rest of the session. Only this
    module builds processors directly, so the fixture stays here rather than in
    `conftest.py`.
    """
    yield
    while _built_processors:
        try:
            _built_processors.pop().shutdown()
        except Exception:
            pass


def _make_laminar_processor(**kwargs) -> LaminarSpanProcessor:
    """A `LaminarSpanProcessor` registered for shutdown; it wraps a batch
    processor and so owns a worker thread of its own.
    """
    processor = LaminarSpanProcessor(**kwargs)
    _built_processors.append(processor)
    return processor


def _make_processor(
    max_export_batch_size_bytes: int,
    max_export_batch_size: int = 1000,
    schedule_delay_millis: float = _NEVER_MILLIS,
):
    """A processor whose only reachable flush trigger is the byte limit, unless
    a caller deliberately lowers the item count or the schedule delay.
    """
    exporter = RecordingExporter()
    processor = SizeLimitedBatchSpanProcessor(
        exporter,
        max_export_batch_size=max_export_batch_size,
        max_queue_size=2048,
        schedule_delay_millis=schedule_delay_millis,
        max_export_batch_size_bytes=max_export_batch_size_bytes,
    )
    _built_processors.append(processor)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    return exporter, processor, provider.get_tracer(__name__)


def _emit(tracer, name: str, payload_size: int = 0):
    with tracer.start_as_current_span(name) as span:
        if payload_size:
            span.set_attribute("gen_ai.input.messages", "x" * payload_size)


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    """Poll until the upstream worker thread has exported. Its flushes are
    asynchronous, so the count- and time-limit tests cannot assert immediately.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_byte_limit_flushes_before_count_and_time_limits():
    # 4 KiB-ish spans against a 10 KB limit: every third span trips it.
    exporter, processor, tracer = _make_processor(max_export_batch_size_bytes=10_000)

    for i in range(10):
        _emit(tracer, f"span-{i}", payload_size=4000)

    # Without the byte limit nothing would have been exported yet: the item
    # limit is 1000 and the schedule delay is 10 minutes. Four batches went out
    # on the byte trigger; the trailing two spans need the explicit flush.
    processor.force_flush()
    assert exporter.batch_sizes == [2, 2, 2, 2, 2]


def test_count_limit_flushes_and_resets_the_running_size():
    # The byte limit is out of reach, so only the item count can fire. The
    # upstream worker drains the queue without telling us, so the running total
    # is stale until the next span observes the empty queue and resyncs it —
    # otherwise that stale total would flush a nearly-empty buffer early.
    exporter, processor, tracer = _make_processor(
        max_export_batch_size_bytes=10**9,
        max_export_batch_size=3,
    )

    for i in range(3):
        _emit(tracer, f"span-{i}", payload_size=1000)

    assert _wait_for(lambda: exporter.batch_sizes == [3]), (
        f"count limit did not flush: {exporter.batch_sizes}"
    )

    _emit(tracer, "after-flush", payload_size=500)

    # Resynced to just the new span, not carrying the exported three.
    assert processor._pending_size_bytes < 1000
    assert exporter.batch_sizes == [3], "resync should not have caused a flush"


def test_time_limit_flushes_and_resets_the_running_size():
    # Same invariant, reached via the schedule delay instead of the item count.
    exporter, processor, tracer = _make_processor(
        max_export_batch_size_bytes=10**9,
        schedule_delay_millis=100,
    )

    for i in range(2):
        _emit(tracer, f"span-{i}", payload_size=1000)

    assert _wait_for(lambda: exporter.batch_sizes == [2]), (
        f"schedule delay did not flush: {exporter.batch_sizes}"
    )

    _emit(tracer, "after-flush", payload_size=500)

    assert processor._pending_size_bytes < 1000


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


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("Hello, how can I help you today? " * 40, id="ascii"),
        pytest.param("Café naïve façade Zürich " * 50, id="latin1"),
        pytest.param("Привет, как я могу помочь? " * 50, id="cyrillic"),
        pytest.param("مرحبا كيف يمكنني مساعدتك " * 50, id="arabic"),
        pytest.param("你好，我今天能为您做些什么？" * 90, id="cjk"),
        pytest.param("Nice 👍🏽 great 🎉 done ✅ " * 50, id="emoji"),
        pytest.param('{"role":"user","content":"分析 and почему"}' * 40, id="mixed"),
        # gen_ai.input.messages opens with ASCII JSON scaffolding before any
        # non-ASCII content, so a prefix-only sample would read this as ASCII.
        pytest.param('[{"role":"system","content":"' + "你好" * 5000, id="ascii-prefix"),
        pytest.param("你好" * 5000 + "a" * 2000, id="non-ascii-prefix"),
    ],
)
def test_utf8_size_is_within_five_percent_of_the_real_encoding(text: str):
    actual = len(text.encode("utf-8"))
    estimate = utf8_size(text)

    assert abs(estimate - actual) / actual < 0.05, (
        f"estimated {estimate}, actual {actual}"
    )


@pytest.mark.parametrize(
    "text",
    ["", "a", "ascii only", "Hello, world! " * 500],
)
def test_utf8_size_is_exact_for_ascii(text: str):
    # str.isascii() is O(1), so the common case is exact rather than sampled.
    assert utf8_size(text) == len(text.encode("utf-8")) == len(text)


@pytest.mark.parametrize(
    "text",
    ["你好世界", "Привет мир", "👍🏽🎉✅"],
)
def test_utf8_size_is_exact_for_short_non_ascii(text: str):
    # Below the sampling budget the whole string is encoded, so no estimation.
    assert utf8_size(text) == len(text.encode("utf-8"))


@pytest.mark.parametrize(
    "text",
    [
        "hello \ud800 world",
        "\ud800" * 20,
        "你好" * 5000 + "\ud800",
    ],
)
def test_utf8_size_survives_lone_surrogates(text: str):
    # Bad JSON decoding upstream can put lone surrogates in a span attribute. A
    # strict encode raises UnicodeEncodeError, and this runs inside on_end, so
    # raising would break span export entirely.
    assert utf8_size(text) > 0


def test_span_size_counts_multibyte_attributes_above_their_character_count():
    _, _, tracer = _make_processor(max_export_batch_size_bytes=10**9)

    with tracer.start_as_current_span("s") as span:
        span.set_attribute("gen_ai.input.messages", "你好" * 2000)
        readable = span._readable_span()

    size = approximate_span_size(readable)
    # 4000 CJK characters are 12000 UTF-8 bytes; counting code points would
    # report ~4000 and let the batch grow to ~3x the configured limit.
    assert size > 11_000


def test_default_is_the_plain_upstream_batch_processor():
    # flush_by_size is opt-in: by default the size trigger must not be wired in
    # at all, so no user thread can ever block on an export.
    processor = _make_laminar_processor(
        base_url="https://api.lmnr.ai",
        api_key="test",
        exporter=RecordingExporter(),
    )

    assert isinstance(processor.instance, BatchSpanProcessor)
    assert not isinstance(processor.instance, SizeLimitedBatchSpanProcessor)
    # The 64-span default is Laminar's, applied here rather than left to OTel's
    # own default of 512 — and it must hold on the default path too.
    assert (
        processor.instance._batch_processor._max_export_batch_size
        == DEFAULT_MAX_EXPORT_BATCH_SIZE
    )


def test_flush_by_size_opts_into_the_size_limited_batch_processor():
    processor = _make_laminar_processor(
        base_url="https://api.lmnr.ai",
        api_key="test",
        exporter=RecordingExporter(),
        flush_by_size=True,
    )

    assert isinstance(processor.instance, SizeLimitedBatchSpanProcessor)
    assert (
        processor.instance._max_export_batch_size_bytes
        == DEFAULT_MAX_EXPORT_BATCH_SIZE_BYTES
    )
    assert (
        processor.instance._batch_processor._max_export_batch_size
        == DEFAULT_MAX_EXPORT_BATCH_SIZE
    )


def test_laminar_span_processor_forwards_both_limits():
    processor = _make_laminar_processor(
        base_url="https://api.lmnr.ai",
        api_key="test",
        exporter=RecordingExporter(),
        max_export_batch_size=7,
        max_export_batch_size_bytes=1234,
        flush_by_size=True,
    )

    assert processor.instance._max_export_batch_size_bytes == 1234
    assert processor.instance._batch_processor._max_export_batch_size == 7


def test_size_limit_is_ignored_without_the_flag():
    # Passing a byte limit without opting in must not silently switch transports.
    processor = _make_laminar_processor(
        base_url="https://api.lmnr.ai",
        api_key="test",
        exporter=RecordingExporter(),
        max_export_batch_size_bytes=1234,
    )

    assert not isinstance(processor.instance, SizeLimitedBatchSpanProcessor)


def test_disable_batch_still_uses_the_simple_processor():
    processor = _make_laminar_processor(
        base_url="https://api.lmnr.ai",
        api_key="test",
        exporter=RecordingExporter(),
        disable_batch=True,
    )

    assert isinstance(processor.instance, SimpleSpanProcessor)


def test_disable_batch_wins_over_flush_by_size():
    processor = _make_laminar_processor(
        base_url="https://api.lmnr.ai",
        api_key="test",
        exporter=RecordingExporter(),
        disable_batch=True,
        flush_by_size=True,
    )

    assert isinstance(processor.instance, SimpleSpanProcessor)


def test_force_reinit_preserves_both_limits():
    processor = _make_laminar_processor(
        base_url="https://api.lmnr.ai",
        api_key="test",
        max_export_batch_size=7,
        max_export_batch_size_bytes=1234,
        flush_by_size=True,
    )

    processor.force_reinit()

    assert isinstance(processor.instance, SizeLimitedBatchSpanProcessor)
    assert processor.instance._max_export_batch_size_bytes == 1234
    assert processor.instance._batch_processor._max_export_batch_size == 7


def test_force_reinit_preserves_the_default_transport():
    processor = _make_laminar_processor(
        base_url="https://api.lmnr.ai",
        api_key="test",
        max_export_batch_size=7,
    )

    processor.force_reinit()

    assert not isinstance(processor.instance, SizeLimitedBatchSpanProcessor)
    assert processor.instance._batch_processor._max_export_batch_size == 7

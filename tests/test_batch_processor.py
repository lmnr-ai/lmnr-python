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

import os
import select
import signal
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

# Byte limit and span size used by the flush-behaviour tests. The ratio matters:
# a span at or above half the limit takes the inline path (see
# `SizeLimitedBatchSpanProcessor.on_end`), so a test that wants to exercise the
# asynchronous handoff needs spans well under that. 4 KB against 100 KB is a
# 4% span, which mirrors the real shape — a ~500 KB GenAI span against the
# 16 MiB default is 3%.
_BYTE_LIMIT = 100_000
_SPAN_BYTES = 4000
# Spans per batch once the limit trips. A span measures slightly more than its
# payload (name, attribute key, the ids the processor stamps), so this is
# floor(limit / measured size) = 24 that fit under the limit, plus the one that
# trips it -- which the asynchronous flush cannot exclude, because `on_end`
# enqueues it before the flush thread runs.
_SPANS_PER_BATCH = 25


class RecordingExporter(SpanExporter):
    def __init__(self):
        self.batches: list[list] = []
        # Stands in for network time. Tests that assert on *where* an export
        # runs need it to take long enough to be observable.
        self.export_delay_s = 0.0
        self._lock = threading.Lock()

    def export(self, spans) -> SpanExportResult:
        spans = list(spans)
        if self.export_delay_s:
            time.sleep(self.export_delay_s)
        with self._lock:
            self.batches.append(spans)
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
    def batch_count(self) -> int:
        with self._lock:
            return len(self.batches)

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
    exporter, processor, tracer = _make_processor(
        max_export_batch_size_bytes=_BYTE_LIMIT
    )

    n_spans = _SPANS_PER_BATCH * 2 + 1
    for i in range(n_spans):
        _emit(tracer, f"span-{i}", payload_size=_SPAN_BYTES)
        # The trigger hands the export to a background thread, so a tight loop
        # would outrun it and land everything in one batch. Real producers do
        # work between spans; waiting for the request to be picked up stands in
        # for that. The backpressure test below covers the tight-loop case.
        assert _wait_for(lambda: not processor._flush_requested.is_set())

    # Without the byte limit nothing would have been exported yet: the item
    # limit is 1000 and the schedule delay is 10 minutes.
    assert _wait_for(lambda: exporter.batch_count >= 2)
    processor.force_flush()
    assert exporter.batch_sizes == [_SPANS_PER_BATCH, _SPANS_PER_BATCH, 1]
    assert exporter.span_count == n_spans


def test_byte_limit_flush_does_not_block_the_ending_thread():
    """The soft trigger must hand the export off, not run it inline.

    This is the property that makes the byte limit cheap enough to enable by
    default: the previous implementation called `force_flush()` from `on_end`,
    which the load tests measured at a p99 of 664 ms.
    """
    exporter, processor, tracer = _make_processor(
        max_export_batch_size_bytes=_BYTE_LIMIT
    )
    exporter.export_delay_s = 0.5

    latencies = []
    for i in range(_SPANS_PER_BATCH * 2 + 1):
        span = tracer.start_span(f"span-{i}")
        span.set_attribute("gen_ai.input.messages", "x" * _SPAN_BYTES)
        started = time.perf_counter()
        span.end()
        latencies.append(time.perf_counter() - started)
        assert _wait_for(lambda: not processor._flush_requested.is_set())

    # Exports of 500 ms each fired, yet no `end()` waited on one.
    assert exporter.batch_count >= 2


def test_a_producer_outrunning_the_flush_thread_is_back_pressured():
    """A tight loop must not let the buffer grow without bound.

    An unconditional handoff makes the byte limit only a hint: the flush thread
    never gets scheduled, so the export ends up bounded by
    `max_export_batch_size` instead. Measured before the inline fallback existed:
    95 MiB exports against a 16 MiB limit at a 512-span count limit.
    """
    exporter, processor, tracer = _make_processor(
        max_export_batch_size_bytes=_BYTE_LIMIT
    )
    exporter.export_delay_s = 0.2

    # No pause between spans: the flush thread cannot keep up, so `on_end` has
    # to export inline to keep the batch bounded.
    n_spans = _SPANS_PER_BATCH * 4
    for i in range(n_spans):
        _emit(tracer, f"span-{i}", payload_size=_SPAN_BYTES)

    processor.force_flush()

    # Without backpressure this is one batch of every span emitted. The bound is
    # not exact -- spans keep arriving during the inline export -- so this
    # asserts the order of magnitude, not a precise count.
    assert max(exporter.batch_sizes) < n_spans, exporter.batch_sizes
    assert max(exporter.batch_sizes) <= _SPANS_PER_BATCH * 2, exporter.batch_sizes
    assert exporter.span_count == n_spans


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


def test_spans_at_half_the_limit_do_not_pair_up():
    """A span at least half the limit must be exported alone, not batched.

    The asynchronous flush cannot exclude the span that triggered it, so without
    this two spans just over half the limit would land in one export of ~1.5x the
    limit and keep growing with the threshold. This is what bounds the worst-case
    export size, and the loopback suite caught it the hard way: with the guard set
    at the whole limit instead, 10-30 MiB spans against a 16 MiB limit paired into
    34 MiB exports and tripled the 413 count.
    """
    # Spans at ~60% of the limit: over half, comfortably under the whole thing.
    exporter, processor, tracer = _make_processor(
        max_export_batch_size_bytes=_BYTE_LIMIT
    )
    payload = int(_BYTE_LIMIT * 0.6)

    for i in range(6):
        _emit(tracer, f"big-{i}", payload_size=payload)

    processor.force_flush()
    assert exporter.batch_sizes == [1, 1, 1, 1, 1, 1]
    assert exporter.span_count == 6


def test_spans_below_half_the_limit_are_handed_off():
    """The converse: the threshold must not drag ordinary spans onto the inline
    path.

    Asserted on *where* the export runs, not on batch size — the inline path
    batches identically, it just charges the producer for the network. A span at
    30% of the limit is under the threshold, so `on_end` must return without
    exporting.
    """
    exporter, processor, tracer = _make_processor(
        max_export_batch_size_bytes=_BYTE_LIMIT
    )
    exporter.export_delay_s = 0.4
    payload = int(_BYTE_LIMIT * 0.3)

    latencies = []
    for i in range(6):
        span = tracer.start_span(f"mid-{i}")
        span.set_attribute("gen_ai.input.messages", "x" * payload)
        started = time.perf_counter()
        span.end()
        latencies.append(time.perf_counter() - started)
        assert _wait_for(lambda: not processor._flush_requested.is_set())

    # The byte limit fired, but on the flush thread: no `end()` saw the 400 ms.
    assert _wait_for(lambda: exporter.batch_count >= 1)

    processor.force_flush()
    assert exporter.span_count == 6


def test_pending_size_resyncs_after_an_external_flush():
    # The worker thread and force_flush() both drain the queue without telling
    # us. An emptied queue must reset the running total, or a stale total would
    # flush a nearly-empty buffer on the very next span.
    exporter, processor, tracer = _make_processor(max_export_batch_size_bytes=_BYTE_LIMIT)

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
    exporter, processor, tracer = _make_processor(max_export_batch_size_bytes=_BYTE_LIMIT)

    for i in range(5):
        _emit(tracer, f"span-{i}", payload_size=_SPAN_BYTES)

    processor.shutdown()
    assert exporter.span_count == 5


def test_shutdown_stops_the_flush_thread():
    """The flush thread must not outlive shutdown.

    It calls `force_flush()`, which upstream turns into a no-op once shut down —
    so a surviving thread would spin on a dead processor and, being a daemon,
    keep a reference to the exporter alive for the rest of the process.
    """
    _, processor, tracer = _make_processor(max_export_batch_size_bytes=_BYTE_LIMIT)
    for i in range(5):
        _emit(tracer, f"span-{i}", payload_size=_SPAN_BYTES)

    thread = processor._flush_thread
    assert thread is not None and thread.is_alive()

    processor.shutdown()

    assert _wait_for(lambda: not thread.is_alive()), "flush thread outlived shutdown"


def test_shutdown_is_idempotent():
    # TracerProvider.shutdown and an explicit call both land here; the second
    # must not raise on an already-joined thread.
    _, processor, tracer = _make_processor(max_export_batch_size_bytes=_BYTE_LIMIT)
    _emit(tracer, "span", payload_size=_SPAN_BYTES)
    processor.shutdown()
    processor.shutdown()


def test_spans_ending_during_an_export_are_not_lost():
    """A span ended while the flush thread is mid-export must still be exported.

    The flush request is cleared *before* the export, so a span arriving during
    it can set the request again and get its own flush. Clearing afterwards
    would swallow that request and leave the span buffered until the schedule
    delay — which the tests set to 10 minutes.
    """
    exporter, processor, tracer = _make_processor(
        max_export_batch_size_bytes=_BYTE_LIMIT
    )
    exporter.export_delay_s = 0.3

    # Enough spans to trip the limit twice, paced so later spans arrive while the
    # first export is still in flight.
    n_spans = _SPANS_PER_BATCH * 2
    for i in range(n_spans):
        _emit(tracer, f"span-{i}", payload_size=_SPAN_BYTES)
        time.sleep(0.01)

    assert _wait_for(lambda: exporter.span_count >= _SPANS_PER_BATCH)
    processor.force_flush()
    assert exporter.span_count == n_spans


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


_fork_required = pytest.mark.skipif(
    not hasattr(os, "fork"), reason="os.fork is unavailable on this platform"
)


def _run_in_fork(child, timeout: float = 30.0) -> str:
    """Run `child(write_result)` in a forked process and return what it reported.

    The child cannot assert — a failed assertion there would exit non-zero
    without telling the parent why, and pytest only sees the parent. So it
    reports a string over a pipe and the parent asserts on that.

    The parent enforces `timeout` itself and SIGKILLs a child that overruns.
    That is essential rather than tidy: the failure these tests exist to catch
    is a child deadlocked on an inherited lock, which never writes to the pipe
    and never exits, so a bare blocking `os.read`/`os.waitpid` would hang the
    whole suite instead of failing it. Returns a `TIMEOUT` sentinel so the
    caller's assertion reports something legible.
    """
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            child(lambda text: os.write(write_fd, text.encode()))
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            os.write(write_fd, f"EXCEPTION {type(exc).__name__}: {exc}".encode())
        finally:
            os.close(write_fd)
            os._exit(0)
    os.close(write_fd)
    chunks: list[bytes] = []
    timed_out = False
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            # select, not a blocking read: a deadlocked child holds the write
            # end open forever, so the read would never return on its own.
            readable, _, _ = select.select([read_fd], [], [], remaining)
            if not readable:
                timed_out = True
                break
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(read_fd)
        if timed_out:
            # SIGKILL, not SIGTERM: the child may be blocked on a lock that will
            # never be released, where a handler would never run.
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        os.waitpid(pid, 0)
    if timed_out:
        partial = b"".join(chunks).decode(errors="replace")
        return f"TIMEOUT after {timeout}s (child killed); partial output: {partial!r}"
    return b"".join(chunks).decode()


@_fork_required
def test_forked_child_can_still_flush_with_locks_held_at_fork():
    """A fork while another thread holds the processor's locks must not wedge it.

    `fork` clones only the calling thread, so a lock another thread held at fork
    time is inherited permanently locked and there is nobody left to release it.
    `on_end` takes `_pending_size_lock` on every span, so without a fork handler
    the child deadlocks on its very first span — and it cannot recover lazily,
    because it blocks before reaching any pid check.
    """
    exporter, processor, tracer = _make_processor(
        max_export_batch_size_bytes=_BYTE_LIMIT,
        max_export_batch_size=2000,
    )

    holding = threading.Event()
    release = threading.Event()

    def hold_both_locks():
        with processor._pending_size_lock:
            with processor._flush_thread_lock:
                holding.set()
                release.wait(30)

    holder = threading.Thread(target=hold_both_locks, daemon=True)
    holder.start()
    assert holding.wait(5), "helper never acquired the locks"

    def child(report):
        for i in range(_SPANS_PER_BATCH + 5):
            _emit(tracer, f"child-{i}", payload_size=_SPAN_BYTES)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and exporter.span_count == 0:
            time.sleep(0.05)
        alive = any(
            t.name == "LmnrSizeLimitedSpanFlush" for t in threading.enumerate()
        )
        report(f"exported={exporter.span_count} flush_thread_alive={alive}")

    try:
        result = _run_in_fork(child)
    finally:
        release.set()
        holder.join(timeout=5)

    assert result.startswith("exported="), result
    exported = int(result.split("exported=")[1].split()[0])
    # The byte trigger fired in the child and the flush thread serviced it, so
    # the size limit is still enforced there rather than silently disabled.
    assert exported > 0, result
    assert "flush_thread_alive=True" in result, result


@_fork_required
def test_fork_reinit_replaces_inherited_locks_and_resets_state():
    exporter, processor, tracer = _make_processor(
        max_export_batch_size_bytes=_BYTE_LIMIT
    )
    _emit(tracer, "before-fork", payload_size=_SPAN_BYTES)
    assert processor._pending_size_bytes > 0

    before = (
        id(processor._pending_size_lock),
        id(processor._flush_thread_lock),
        id(processor._flush_requested),
    )

    processor._at_fork_reinit()

    after = (
        id(processor._pending_size_lock),
        id(processor._flush_thread_lock),
        id(processor._flush_requested),
    )
    # Every synchronization primitive is a fresh object, not the inherited one.
    assert all(b != a for b, a in zip(before, after))
    # The inherited queue is drained by upstream's fork handler, so the running
    # total no longer describes anything.
    assert processor._pending_size_bytes == 0
    assert processor._flush_thread is not None
    assert processor._flush_thread.is_alive()
    assert processor._flush_thread_pid == os.getpid()
    del exporter


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

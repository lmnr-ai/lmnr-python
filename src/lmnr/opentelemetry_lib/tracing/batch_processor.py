import logging
import os
import threading

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.util.types import AttributeValue

logger = logging.getLogger(__name__)

# Lower than OTel's own default of 512: GenAI spans are far heavier than the
# spans that default was chosen for.
DEFAULT_MAX_EXPORT_BATCH_SIZE = 64

# GenAI spans carry whole prompts and completions as string attributes, so a
# batch of only a few dozen of them can be tens of megabytes. 16 MiB keeps a
# single export comfortably under the ingest limits while still batching.
DEFAULT_MAX_EXPORT_BATCH_SIZE_BYTES = 16 * 1024 * 1024

# Non-string attribute values (ints, floats, bools) are counted as a flat 8
# bytes instead of being measured — they never dominate a GenAI payload.
_FIXED_ATTRIBUTE_VALUE_SIZE = 8

# Strings longer than this are measured by sampling rather than encoded whole.
# 1 KiB of samples keeps the worst observed error near 2% while costing ~3us
# even on a 200 KB value; encoding such a value outright costs ~125us.
_UTF8_SAMPLE_BUDGET = 1024


def utf8_size(value: str) -> int:
    """Byte length of `value` encoded as UTF-8 — exact for ASCII, sampled above
    `_UTF8_SAMPLE_BUDGET` characters.

    `len()` alone counts code points, which undercounts CJK by ~3x and
    Cyrillic/Arabic/emoji by ~2x. On GenAI payloads that put a CJK user's real
    16 MiB batch at ~41 MiB — the exact oversized-export case this limit exists
    to prevent.

    `str.isascii()` is O(1) (CPython caches the flag on the string), so the
    common case is both exact and free. Beyond that we encode a *strided*
    sample: prefix sampling would be fooled by `gen_ai.input.messages`, which
    opens with ASCII JSON scaffolding (`[{"role":"system","content":"`) before
    reaching non-ASCII content, and would report near-ASCII sizes for it.

    `errors="replace"` is required, not defensive: attribute values reaching
    here can contain lone surrogates (bad JSON decoding upstream produces
    them), and a strict encode raises `UnicodeEncodeError`. This runs inside
    `on_end`, so raising would break span export.
    """
    if value.isascii():
        return len(value)
    length = len(value)
    if length <= _UTF8_SAMPLE_BUDGET:
        return len(value.encode("utf-8", errors="replace"))
    sample = value[:: length // _UTF8_SAMPLE_BUDGET + 1]
    bytes_per_char = len(sample.encode("utf-8", errors="replace")) / len(sample)
    return int(length * bytes_per_char)


def _value_size(value: AttributeValue) -> int:
    if isinstance(value, str):
        return utf8_size(value)
    if isinstance(value, (list, tuple)):
        return sum(_value_size(item) for item in value)
    return _FIXED_ATTRIBUTE_VALUE_SIZE


def _attributes_size(attributes) -> int:
    if not attributes:
        return 0
    return sum(utf8_size(key) + _value_size(value) for key, value in attributes.items())


def approximate_span_size(span: ReadableSpan) -> int:
    """Approximate the exported size of a span, in bytes.

    Deliberately cheap, and an underestimate: protobuf framing, ids and
    timestamps are ignored, and non-string values are counted flat.
    Underestimating small spans is fine — the item count and schedule delay
    limits fire first for those. What matters is that a span carrying a large
    prompt or completion is measured close to its real weight, which is what
    `utf8_size` is for.
    """
    total = utf8_size(span.name or "")
    total += _attributes_size(span.attributes)
    for event in span.events or ():
        total += utf8_size(event.name or "") + _attributes_size(event.attributes)
    for link in span.links or ():
        total += _attributes_size(link.attributes)
    if span.status is not None and span.status.description:
        total += utf8_size(span.status.description)
    return total


class SizeLimitedBatchSpanProcessor(BatchSpanProcessor):
    """`BatchSpanProcessor` with a third, size-based flush trigger.

    Opt-in via `Laminar.initialize(flush_by_size=True)`; the default transport
    is the plain upstream `BatchSpanProcessor`.

    Upstream flushes on whichever comes first: `max_export_batch_size` spans
    buffered, or `schedule_delay_millis` elapsed. Neither bounds the *payload*,
    so a handful of large GenAI spans can produce an export big enough for the
    backend to reject. This adds `max_export_batch_size_bytes`: when the span
    being ended would push the buffer past the limit, the buffer is flushed
    first and the span then starts a fresh batch.

    The flush normally runs on a dedicated background thread owned by this
    processor, so `on_end` does not block the thread that ended the span. That
    thread does nothing but wait for a trigger and call the public
    `force_flush()`.

    Why not hand the work to the upstream worker thread instead? Because there
    is no public way to ask it for one. `BatchProcessor.worker` picks its export
    strategy from whether its sleep was interrupted, and the only lever a
    subclass has — setting `_worker_awaken` — selects
    `EXPORT_WHILE_BATCH_EXCEEDS_THRESHOLD`, which explicitly declines to export
    a queue shorter than `max_export_batch_size`. That is precisely the case the
    byte limit creates. The strategy that *would* work,
    `EXPORT_AT_LEAST_ONE_BATCH`, lives in `opentelemetry.sdk._shared_internal`
    and is not reachable without either importing a private module or
    reimplementing `worker()`. One extra mostly-idle thread is a better trade
    than depending on private control flow.

    An asynchronous flush alone would turn the byte limit from a bound into a
    hint: a producer in a tight loop enqueues faster than the flush thread can
    drain, so the export that eventually goes out is sized by
    `max_export_batch_size`, not by the byte limit. Measured, with 500 KB spans
    against a 16 MiB limit: 95 MiB exports at a 512-span count limit — worse
    than the problem this class exists to solve.

    So the handoff is conditional: `on_end` exports on its own thread when the
    previous request has not been picked up yet (the producer is outrunning the
    network, and backpressure is the only thing keeping the buffer bounded), or
    when the span alone exceeds the limit and is therefore its own export
    regardless. Everything else is handed off. Steady-state workloads only ever
    take the handoff path.

    Load testing (see `loadtest/RESULTS.md`) measured the fully synchronous
    implementation at a p99 `on_end` of 664 ms on a realistic GenAI mix, versus
    0.2 ms for the plain upstream processor. Keeping the common case off the
    calling thread is what makes the byte limit cheap enough to consider
    enabling by default.
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        max_export_batch_size_bytes: int | None = None,
        **kwargs,
    ):
        super().__init__(span_exporter, **kwargs)
        self._max_export_batch_size_bytes = (
            max_export_batch_size_bytes or DEFAULT_MAX_EXPORT_BATCH_SIZE_BYTES
        )
        self._pending_size_bytes = 0
        self._pending_size_lock = threading.Lock()

        # Set by on_end to request a flush; cleared by the flush thread once it
        # has one in progress. An Event rather than a queue because requests
        # coalesce: two triggers arriving before the thread wakes need one
        # flush, not two.
        self._flush_requested = threading.Event()
        self._flush_shutdown = False
        self._flush_thread: threading.Thread | None = None
        self._flush_thread_lock = threading.Lock()
        self._flush_thread_pid: int | None = None
        self._start_flush_thread()

    def _start_flush_thread(self) -> None:
        """Start the flush thread, or restart it after a fork.

        A forked child inherits no threads, so `_flush_thread` would name a
        thread that does not exist there and flushes would silently stop. The
        pid is recorded so `on_end` can detect that and respawn. Upstream
        handles the same problem for its own worker via `os.register_at_fork`.
        """
        with self._flush_thread_lock:
            if self._flush_shutdown:
                return
            pid = os.getpid()
            if self._flush_thread is not None and self._flush_thread_pid == pid:
                return
            # After a fork the inherited Event may be set with no thread to
            # service it, and the lock state is undefined. Start clean.
            self._flush_requested = threading.Event()
            self._flush_thread = threading.Thread(
                name="LmnrSizeLimitedSpanFlush",
                target=self._flush_loop,
                daemon=True,
            )
            self._flush_thread_pid = pid
            self._flush_thread.start()

    def _flush_loop(self) -> None:
        while True:
            self._flush_requested.wait()
            if self._flush_shutdown:
                return
            # Clear before flushing, not after: a span ended *during* the export
            # belongs to the next batch and must be able to request its own
            # flush. Clearing afterwards would swallow that request.
            self._flush_requested.clear()
            try:
                self.force_flush()
            except Exception:
                # A failed export must not kill the thread — that would silently
                # disable the byte limit for the rest of the process.
                logger.debug("Size-triggered span flush failed", exc_info=True)

    def on_end(self, span: ReadableSpan) -> None:
        if not (span.context and span.context.trace_flags.sampled):
            return

        size = approximate_span_size(span)
        with self._pending_size_lock:
            # The worker thread drains the buffer on its own schedule, without
            # telling us, so the running total is an estimate that resyncs
            # whenever the queue is observed empty. Between resyncs it can drift
            # either way — over-counting after a partial drain, under-counting
            # when a concurrent emitter resyncs in the window between our
            # bookkeeping and our enqueue below. Both self-correct at the next
            # resync, and both only shift when a flush happens, never whether
            # spans are exported.
            if not self._batch_processor._queue:
                self._pending_size_bytes = 0
            should_flush = (
                self._pending_size_bytes > 0
                and self._pending_size_bytes + size > self._max_export_batch_size_bytes
            )
            # This span lands in the buffer either way, and the flush below
            # takes out only what preceded it.
            self._pending_size_bytes = (
                size if should_flush else self._pending_size_bytes + size
            )

        if should_flush:
            if self._flush_thread_pid != os.getpid():
                self._start_flush_thread()
            # Two cases have to be exported on this thread rather than handed
            # off, because an asynchronous flush cannot exclude the span that
            # triggered it: `on_end` returns and enqueues before the flush thread
            # runs, so the triggering span rides along in the batch it caused.
            # Overshooting by one ordinary span is fine — the limit is an
            # estimate anyway — but not in these two cases.
            #
            # 1. A request is still pending from last time, so the flush thread
            #    has not even picked it up. The producer is outrunning the
            #    network, and without backpressure the byte limit degrades into a
            #    hint: the export size ends up bounded by `max_export_batch_size`
            #    instead, measured at 95 MiB against a 16 MiB limit at a 512-span
            #    count limit.
            # 2. This span alone is at or over the limit. It will be its own
            #    export no matter what, so there is no batching to win, and
            #    letting it pair with another oversized span would double the
            #    largest export in the workload that can least afford it.
            oversized = size >= self._max_export_batch_size_bytes
            if self._flush_requested.is_set() or oversized:
                self.force_flush()
            else:
                self._flush_requested.set()

        super().on_end(span)

    def shutdown(self) -> None:
        """Stop the flush thread, then hand off to upstream's shutdown.

        Order matters. Upstream's `shutdown` sets a flag that makes
        `force_flush` a no-op and joins its own worker, so a flush thread still
        running at that point would either race the exporter's teardown or
        silently drop the spans it was asked to ship. Stopping first, then
        letting upstream drain the queue itself, means nothing is lost: its
        shutdown exports whatever is buffered.
        """
        self._flush_shutdown = True
        self._flush_requested.set()
        thread = self._flush_thread
        is_self = thread is threading.current_thread()
        if thread is not None and thread.is_alive() and not is_self:
            # Bounded: the thread only ever waits on the Event or sits in one
            # export, and upstream's shutdown will drain anything left over.
            thread.join(timeout=30)
        super().shutdown()

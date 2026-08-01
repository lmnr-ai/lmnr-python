import threading

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.util.types import AttributeValue

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


def _value_size(value: AttributeValue) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(_value_size(item) for item in value)
    return _FIXED_ATTRIBUTE_VALUE_SIZE


def _attributes_size(attributes) -> int:
    if not attributes:
        return 0
    return sum(len(key) + _value_size(value) for key, value in attributes.items())


def approximate_span_size(span: ReadableSpan) -> int:
    """Approximate the exported size of a span, in bytes.

    Deliberately cheap and deliberately an underestimate: string lengths are
    character counts (not UTF-8 byte counts) and protobuf framing, ids and
    timestamps are ignored. Underestimating small spans is fine — the item
    count and schedule delay limits fire first for those. What matters is that
    a span carrying a large prompt or completion is measured close to its real
    weight.
    """
    total = len(span.name or "")
    total += _attributes_size(span.attributes)
    for event in span.events or ():
        total += len(event.name or "") + _attributes_size(event.attributes)
    for link in span.links or ():
        total += _attributes_size(link.attributes)
    if span.status is not None and span.status.description:
        total += len(span.status.description)
    return total


class SizeLimitedBatchSpanProcessor(BatchSpanProcessor):
    """`BatchSpanProcessor` with a third, size-based flush trigger.

    Upstream flushes on whichever comes first: `max_export_batch_size` spans
    buffered, or `schedule_delay_millis` elapsed. Neither bounds the *payload*,
    so a handful of large GenAI spans can produce an export big enough for the
    backend to reject. This adds `max_export_batch_size_bytes`: when the span
    being ended would push the buffer past the limit, the buffer is flushed
    first and the span then starts a fresh batch.

    That flush is synchronous on the ending thread — the upstream worker thread
    cannot be asked to export a buffer shorter than `max_export_batch_size`
    (see `BatchExportStrategy.EXPORT_WHILE_BATCH_EXCEEDS_THRESHOLD`), and
    spawning a second worker is not worth the complexity. The cost is bounded:
    it happens once per `max_export_batch_size_bytes` of span data, and only
    for workloads large enough that the alternative is a rejected export.
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
            self.force_flush()

        super().on_end(span)

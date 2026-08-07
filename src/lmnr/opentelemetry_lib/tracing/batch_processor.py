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

    That flush is synchronous on the ending thread — the upstream worker thread
    cannot be asked to export a buffer shorter than `max_export_batch_size`
    (see `BatchExportStrategy.EXPORT_WHILE_BATCH_EXCEEDS_THRESHOLD`), and
    spawning a second worker is not worth the complexity. Blocking a user's
    thread on an export is the reason this is opt-in rather than the default.
    The cost is bounded: it happens once per `max_export_batch_size_bytes` of
    span data.
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

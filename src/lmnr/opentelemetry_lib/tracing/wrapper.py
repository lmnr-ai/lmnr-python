import atexit
from collections.abc import Sequence
from typing import cast

from opentelemetry import trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider

from lmnr.opentelemetry_lib.tracing.context import clear_context
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import SessionRecordingOptions

# This module is the leaf that holds the tracing singleton. It must NOT import
# `tracing/__init__` or `tracing/instruments` (and so none of the instrumentors),
# otherwise `tracing/tracer.py` -> here would close an import cycle.

TRACER_NAME = "lmnr.tracer"

LOG = get_default_logger(__name__)


def default_session_recording_options() -> SessionRecordingOptions:
    return {"mask_input_options": None}


def _detach_span_processor(provider: TracerProvider, processor: SpanProcessor) -> None:
    """Remove `processor` from `provider`, touching the private
    `_span_processors` tuple under its lock (same approach as the Langfuse
    bridge's `_remove_span_processor`)."""
    active = getattr(provider, "_active_span_processor", None)
    current = getattr(active, "_span_processors", None)
    if current is None:
        return
    filtered = tuple(p for p in cast(Sequence[SpanProcessor], current) if p is not processor)
    lock = getattr(active, "_lock", None)
    if active is None:
        return
    if lock is not None:
        with lock:
            active._span_processors = filtered
    else:
        active._span_processors = filtered


def _detach_log_processor(
    provider: LoggerProvider, processor: BatchLogRecordProcessor
) -> None:
    """`_detach_span_processor`'s counterpart for the logs pipeline."""
    multi = getattr(provider, "_multi_log_record_processor", None)
    current = getattr(multi, "_log_record_processors", None)
    if current is None:
        return
    filtered = tuple(p for p in cast(Sequence[BatchLogRecordProcessor], current) if p is not processor)
    lock = getattr(multi, "_lock", None)
    if multi is None:
        return
    if lock is not None:
        with lock:
            multi._log_record_processors = filtered
    else:
        multi._log_record_processors = filtered


class TracerWrapper:
    """Holds the OpenTelemetry plumbing for one process lifetime."""

    def __init__(
        self,
        *,
        resource: Resource,
        span_processor: SpanProcessor,
        tracer_provider: TracerProvider,
        logger_provider: LoggerProvider,
        log_processor: BatchLogRecordProcessor,
        async_client: AsyncLaminarClient | None,
    ) -> None:
        self.resource: Resource = resource
        self.span_processor: SpanProcessor = span_processor
        self.tracer_provider: TracerProvider = tracer_provider
        self.logger_provider: LoggerProvider = logger_provider
        self.log_processor: BatchLogRecordProcessor = log_processor
        self.async_client: AsyncLaminarClient | None = async_client

    def get_tracer(self) -> trace.Tracer:
        return self.tracer_provider.get_tracer(TRACER_NAME)

    def flush(self) -> bool:
        span_result = self.span_processor.force_flush()
        log_result = self.log_processor.force_flush()
        return span_result and log_result

    def shutdown(self) -> None:
        """Detach and shut down this run's processors.

        The providers deliberately OUTLIVE the wrapper. OTel refuses to
        override an already-set global `TracerProvider`, so a later
        `init_tracing()` can never publish a replacement — and instrumentors
        that captured a tracer at `_instrument()` time (MCP, pydantic_ai, the
        traceloop-derived ones) are not re-instrumented either, since
        `BaseInstrumentor.instrument()` no-ops when already instrumented.
        Shutting the provider down here would therefore strand them on a dead
        provider for the rest of the process.
        """
        _detach_span_processor(self.tracer_provider, self.span_processor)
        _detach_log_processor(self.logger_provider, self.log_processor)
        self.span_processor.shutdown()
        self.log_processor.shutdown()

    def force_reinit_processor(self) -> bool:
        if not isinstance(self.span_processor, LaminarSpanProcessor):
            LOG.warning("Not using LaminarSpanProcessor, cannot force reinit")
            return False
        spans_flush = self.span_processor.force_flush()
        spans_reinit = self.span_processor.force_reinit()
        logs_flush = self.log_processor.force_flush()
        # Clear the isolated context to prevent subsequent invocations
        # (e.g., in Lambda) from continuing traces from previous invocations
        clear_context()
        return spans_flush and spans_reinit and logs_flush

    def clear(self) -> None:
        """Reset per-run state. Used in between tests."""
        if isinstance(self.span_processor, LaminarSpanProcessor):
            self.span_processor.clear()
        # Clear the isolated context state for clean test state
        clear_context()

    def exit_handler(self) -> None:
        # Force flushes for debug environments (e.g. local development)
        if isinstance(self.span_processor, LaminarSpanProcessor):
            self.span_processor.clear()
        _success = self.flush()


# The singleton lives here, not on the class, so that `TracerWrapper(...)` can
# never double as an accessor that silently initializes tracing.
_tracer_wrapper: TracerWrapper | None = None
# Kept outside the wrapper because the browser utils read it before (and
# without) initialization.
_session_recording_options: SessionRecordingOptions | None = None


def publish_tracer_wrapper(wrapper: TracerWrapper) -> None:
    global _tracer_wrapper
    _tracer_wrapper = wrapper


def set_session_recording_options(options: SessionRecordingOptions) -> None:
    global _session_recording_options
    _session_recording_options = options


def get_tracer_wrapper() -> TracerWrapper | None:
    """The initialized wrapper, or None if `init_tracing` has not run.

    Deliberately lock-free; see `is_tracing_initialized`.
    """
    return _tracer_wrapper


def is_tracing_initialized() -> bool:
    # This does not take `_lock`, but it is fine to return False from here even
    # if initialization is going on.
    #
    # If we tried to acquire the lock here, it could deadlock if an automatic
    # instrumentation is importing a file that (at the top level) has a
    # function annotated with Laminar's `observe` decorator.
    # The decorator is evaluated at import time, inside `init_instrumentations`,
    # which is called by `init_tracing` while holding the lock.
    # Without the lock here, we will simply return False, which will cause
    # the decorator to return the original function. This is fine, at runtime,
    # the next import statement will re-evaluate the decorator, and Laminar will
    # have been initialized by that time.
    return _tracer_wrapper is not None


def flush_tracing() -> bool:
    if _tracer_wrapper is None:
        LOG.debug("Laminar tracing is not initialized, cannot flush")
        return False
    return _tracer_wrapper.flush()


def force_reinit_processor() -> bool:
    if _tracer_wrapper is None:
        return False
    return _tracer_wrapper.force_reinit_processor()


def clear_tracing_state() -> None:
    if _tracer_wrapper is None:
        return
    _tracer_wrapper.clear()


def shutdown_tracing() -> None:
    global _tracer_wrapper
    wrapper = _tracer_wrapper
    if wrapper is None:
        return
    # Unregister first: atexit holds a strong reference, so without this every
    # initialize()/shutdown() cycle pins a retired wrapper (and its exporter
    # threads) alive for the life of the process.
    atexit.unregister(wrapper.exit_handler)
    _tracer_wrapper = None
    wrapper.shutdown()


def reset_tracing() -> None:
    """Drop the tracing singleton. Test hook.

    Goes through `shutdown_tracing` so the retired processors are detached
    from the (reused) providers rather than left accumulating on them.
    """
    global _session_recording_options
    shutdown_tracing()
    _session_recording_options = None


def get_session_recording_options() -> SessionRecordingOptions:
    """The session recording options set during initialization.

    Callable before `init_tracing` — the browser utils read it unconditionally.
    """
    return _session_recording_options or default_session_recording_options()

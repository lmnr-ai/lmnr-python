import atexit
import logging
import sys
import threading
from collections.abc import Sequence

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

from lmnr.opentelemetry_lib.tracing.context import (
    clear_context,
    setup_thread_context_inheritance,
)
from lmnr.opentelemetry_lib.tracing.exporter import LaminarLogExporter
from lmnr.opentelemetry_lib.tracing.instruments import (
    Instruments,
    init_instrumentations,
)
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import SessionRecordingOptions

# instead of importing from opentelemetry.instrumentation.threading,
# we import from our modified copy to use Laminar's isolated context.
from ..opentelemetry.instrumentation.threading import ThreadingInstrumentor

TRACER_NAME = "lmnr.tracer"

LOG = get_default_logger(__name__)


def _default_session_recording_options() -> SessionRecordingOptions:
    return {"mask_input_options": None}


class TracerWrapper:
    """Holds the OpenTelemetry plumbing for one process lifetime.
    """

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
        self.resource = resource
        self.span_processor = span_processor
        self.tracer_provider = tracer_provider
        self.logger_provider = logger_provider
        self.log_processor = log_processor
        self.async_client = async_client

    def get_tracer(self) -> trace.Tracer:
        return self.tracer_provider.get_tracer(TRACER_NAME)

    def flush(self) -> bool:
        span_result = self.span_processor.force_flush()
        log_result = self.log_processor.force_flush()
        return span_result and log_result

    def shutdown(self) -> None:
        self.tracer_provider.shutdown()
        self.logger_provider.shutdown()

    def force_reinit_processor(self) -> bool:
        if not isinstance(self.span_processor, LaminarSpanProcessor):
            LOG.warning("Not using LaminarSpanProcessor, cannot force reinit")
            return False
        self.span_processor.force_flush()
        self.span_processor.force_reinit()
        self.log_processor.force_flush()
        # Clear the isolated context to prevent subsequent invocations
        # (e.g., in Lambda) from continuing traces from previous invocations
        clear_context()
        return True

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
        self.flush()


# The singleton lives here, not on the class, so that `TracerWrapper(...)` can
# never double as an accessor that silently initializes tracing.
_lock = threading.Lock()
_tracer_wrapper: TracerWrapper | None = None
# Kept outside the wrapper because the browser utils read it before (and
# without) initialization.
_session_recording_options: SessionRecordingOptions | None = None


def init_tracing(
    app_name: str | None = sys.argv[0],
    disable_batch: bool = False,
    exporter: SpanExporter | None = None,
    resource_attributes: (
        dict[
            str,
            str
            | int
            | float
            | bool
            | Sequence[str]
            | Sequence[int | float]
            | Sequence[bool],
        ]
        | None
    ) = None,
    instruments: set[Instruments] | None = None,
    block_instruments: set[Instruments] | None = None,
    base_url: str | None = None,
    port: int = 8443,
    http_port: int = 443,
    project_api_key: str | None = None,
    max_export_batch_size: int | None = None,
    max_export_batch_size_bytes: int | None = None,
    flush_by_size: bool = False,
    force_http: bool = False,
    timeout_seconds: int | None = None,
    set_global_tracer_provider: bool = True,
    otel_logger_level: int = logging.ERROR,
    session_recording_options: SessionRecordingOptions | None = None,
) -> TracerWrapper:
    """Initialize Laminar tracing. Idempotent: a second call returns the
    existing wrapper and ignores the new arguments."""
    global _tracer_wrapper, _session_recording_options

    # Silence some opentelemetry warnings
    logging.getLogger("opentelemetry.trace").setLevel(otel_logger_level)

    base_http_url = f"{base_url}:{http_port}" if base_url else None
    timeout = timeout_seconds if timeout_seconds is not None else 30

    with _lock:
        if _tracer_wrapper is not None:
            return _tracer_wrapper

        _session_recording_options = (
            session_recording_options or _default_session_recording_options()
        )

        resource = Resource(
            attributes={**(resource_attributes or {}), SERVICE_NAME: app_name or "default_app"}
        )

        async_client = (
            AsyncLaminarClient(
                base_url=base_http_url or "https://api.lmnr.ai",
                project_api_key=project_api_key,
            )
            if project_api_key
            else None
        )

        span_processor = LaminarSpanProcessor(
            base_url=base_url,
            api_key=project_api_key,
            http_port=http_port,
            grpc_port=port,
            exporter=exporter,
            max_export_batch_size=max_export_batch_size,
            max_export_batch_size_bytes=max_export_batch_size_bytes,
            flush_by_size=flush_by_size,
            timeout_seconds=timeout,
            force_http=force_http,
            disable_batch=disable_batch,
        )

        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(span_processor)
        global_provider = trace.get_tracer_provider()
        if set_global_tracer_provider and isinstance(
            global_provider, trace.ProxyTracerProvider
        ):
            trace.set_tracer_provider(tracer_provider)

        # Setup LoggerProvider for OTel logs
        log_exporter = LaminarLogExporter(
            base_url=base_url,
            port=http_port if force_http else port,
            api_key=project_api_key,
            timeout_seconds=timeout,
            force_http=force_http,
        )
        log_processor = BatchLogRecordProcessor(log_exporter)
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(log_processor)

        # Set global logger provider (follows same flag as tracer provider)
        if set_global_tracer_provider:
            set_logger_provider(logger_provider)

        wrapper = TracerWrapper(
            resource=resource,
            span_processor=span_processor,
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            log_processor=log_processor,
            async_client=async_client,
        )

        # Setup threading context inheritance
        setup_thread_context_inheritance()

        # This is not a real instrumentation and does not generate telemetry
        # data, but it is required to ensure that OpenTelemetry context
        # propagation is enabled.
        # See the README at:
        # https://pypi.org/project/opentelemetry-instrumentation-threading/
        ThreadingInstrumentor().instrument()

        init_instrumentations(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            instruments=instruments,
            block_instruments=block_instruments,
            async_client=async_client,
            lmnr_span_processor=span_processor,
        )

        # Publish LAST, after init_instrumentations. Some instrumentors import
        # modules that evaluate Laminar's `observe` decorator at import time;
        # those must see tracing as *not yet* initialized (see the note on
        # is_tracing_initialized below). The Langfuse bridge relies on the same
        # ordering to avoid double-attaching to Laminar's own tracer provider.
        _tracer_wrapper = wrapper

        atexit.register(wrapper.exit_handler)

        return wrapper


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
    """Drop the singleton without shutting it down. Test hook."""
    global _tracer_wrapper, _session_recording_options
    if _tracer_wrapper is not None:
        atexit.unregister(_tracer_wrapper.exit_handler)
    _tracer_wrapper = None
    _session_recording_options = None


def get_session_recording_options() -> SessionRecordingOptions:
    """The session recording options set during initialization.

    Callable before `init_tracing` — the browser utils read it unconditionally.
    """
    return _session_recording_options or _default_session_recording_options()

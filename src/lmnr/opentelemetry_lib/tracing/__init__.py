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
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

from lmnr.opentelemetry_lib.tracing.context import setup_thread_context_inheritance
from lmnr.opentelemetry_lib.tracing.exporter import LaminarLogExporter
from lmnr.opentelemetry_lib.tracing.instruments import (
    Instruments,
    init_instrumentations,
)
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from lmnr.opentelemetry_lib.tracing.wrapper import (
    TRACER_NAME,
    TracerWrapper,
    clear_tracing_state,
    default_session_recording_options,
    flush_tracing,
    force_reinit_processor,
    get_session_recording_options,
    get_tracer_wrapper,
    is_tracing_initialized,
    publish_tracer_wrapper,
    reset_tracing,
    set_session_recording_options,
    shutdown_tracing,
)
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import SessionRecordingOptions

# instead of importing from opentelemetry.instrumentation.threading,
# we import from our modified copy to use Laminar's isolated context.
from ..opentelemetry.instrumentation.threading import ThreadingInstrumentor

__all__ = [
    "TRACER_NAME",
    "Instruments",
    "TracerWrapper",
    "clear_tracing_state",
    "flush_tracing",
    "force_reinit_processor",
    "get_session_recording_options",
    "get_tracer_wrapper",
    "init_tracing",
    "is_tracing_initialized",
    "reset_tracing",
    "shutdown_tracing",
]

LOG = get_default_logger(__name__)

# Serializes `init_tracing`. The singleton itself lives in `tracing/wrapper.py`.
_lock = threading.Lock()
# The providers are built ONCE per process and reused across
# initialize()/shutdown() cycles — see TracerWrapper.shutdown.
_tracer_provider: TracerProvider | None = None
_logger_provider: LoggerProvider | None = None


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
    global _tracer_provider, _logger_provider

    # Silence some opentelemetry warnings
    logging.getLogger("opentelemetry.trace").setLevel(otel_logger_level)

    base_http_url = f"{base_url}:{http_port}" if base_url else None
    timeout = timeout_seconds if timeout_seconds is not None else 30

    with _lock:
        existing = get_tracer_wrapper()
        if existing is not None:
            return existing

        set_session_recording_options(
            session_recording_options or default_session_recording_options()
        )

        resource = Resource(
            attributes={
                **(resource_attributes or {}),
                SERVICE_NAME: app_name or "default_app",
            }
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

        if _tracer_provider is None:
            _tracer_provider = TracerProvider(resource=resource)
            global_provider = trace.get_tracer_provider()
            if set_global_tracer_provider and isinstance(
                global_provider, trace.ProxyTracerProvider
            ):
                trace.set_tracer_provider(_tracer_provider)
        elif _tracer_provider.resource != resource:
            LOG.warning(
                "Reusing the tracer provider from a previous Laminar " +
                "initialization; its resource attributes (including app_name) " +
                "are fixed at first initialization and will not be updated."
            )
        tracer_provider = _tracer_provider
        tracer_provider.add_span_processor(span_processor)

        # Setup LoggerProvider for OTel logs
        log_exporter = LaminarLogExporter(
            base_url=base_url,
            port=http_port if force_http else port,
            api_key=project_api_key,
            timeout_seconds=timeout,
            force_http=force_http,
        )
        log_processor = BatchLogRecordProcessor(log_exporter)
        if _logger_provider is None:
            _logger_provider = LoggerProvider(resource=resource)
            # Set global logger provider (follows same flag as tracer provider)
            if set_global_tracer_provider:
                set_logger_provider(_logger_provider)
        logger_provider = _logger_provider
        logger_provider.add_log_record_processor(log_processor)

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
        publish_tracer_wrapper(wrapper)

        _handler = atexit.register(wrapper.exit_handler)

        return wrapper



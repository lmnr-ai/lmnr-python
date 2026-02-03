import atexit
import logging
import threading

from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from lmnr.opentelemetry_lib.tracing.exporter import LaminarLogExporter
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.types import SessionRecordingOptions
from lmnr.sdk.log import VerboseColorfulFormatter
from lmnr.opentelemetry_lib.tracing.instruments import (
    Instruments,
    init_instrumentations,
)
from lmnr.opentelemetry_lib.tracing.context import (
    attach_context,
    clear_context,
    pop_span_context as ctx_pop_span_context,
    get_current_context,
    get_token_stack,
    push_span_context as ctx_push_span_context,
    set_token_stack,
)

from opentelemetry import trace
from opentelemetry.context import Context

# instead of importing from opentelemetry.instrumentation.threading,
# we import from our modified copy to use Laminar's isolated context.
from ..opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

TRACER_NAME = "lmnr.tracer"

MAX_EVENTS_OR_ATTRIBUTES_PER_SPAN = 5000


class TracerWrapper(object):
    resource_attributes: dict = {}
    enable_content_tracing: bool = True
    session_recording_options: SessionRecordingOptions = {}
    _lock = threading.Lock()
    _tracer_provider: TracerProvider | None = None
    _logger_provider: LoggerProvider | None = None
    _logger: logging.Logger
    _async_client: AsyncLaminarClient
    _resource: Resource
    _span_processor: SpanProcessor
    _log_processor: BatchLogRecordProcessor | None = None
    _original_thread_init = None

    def __new__(
        cls,
        disable_batch=False,
        exporter: SpanExporter | None = None,
        instruments: set[Instruments] | None = None,
        block_instruments: set[Instruments] | None = None,
        base_url: str | None = None,
        port: int = 8443,
        http_port: int = 443,
        project_api_key: str | None = None,
        max_export_batch_size: int | None = None,
        force_http: bool = False,
        timeout_seconds: int = 30,
        set_global_tracer_provider: bool = True,
        otel_logger_level: int = logging.ERROR,
        session_recording_options: SessionRecordingOptions | None = None,
    ) -> "TracerWrapper":
        # Silence some opentelemetry warnings
        logging.getLogger("opentelemetry.trace").setLevel(otel_logger_level)

        base_http_url = f"{base_url}:{http_port}" if base_url else None
        with cls._lock:
            if not hasattr(cls, "instance"):
                cls._initialize_logger(cls)
                obj = super(TracerWrapper, cls).__new__(cls)

                # Store session recording options
                cls.session_recording_options = session_recording_options or {}

                if project_api_key:
                    obj._async_client = AsyncLaminarClient(
                        base_url=base_http_url or "https://api.lmnr.ai",
                        project_api_key=project_api_key,
                    )
                else:
                    obj._async_client = None

                obj._resource = Resource(attributes=TracerWrapper.resource_attributes)

                obj._span_processor = LaminarSpanProcessor(
                    base_url=base_url,
                    api_key=project_api_key,
                    http_port=http_port,
                    grpc_port=port,
                    exporter=exporter,
                    max_export_batch_size=max_export_batch_size,
                    timeout_seconds=timeout_seconds,
                    force_http=force_http,
                    disable_batch=disable_batch,
                )

                lmnr_provider = TracerProvider(resource=obj._resource)
                global_provider = trace.get_tracer_provider()
                if set_global_tracer_provider and isinstance(
                    global_provider, trace.ProxyTracerProvider
                ):
                    trace.set_tracer_provider(lmnr_provider)

                obj._tracer_provider = lmnr_provider

                obj._tracer_provider.add_span_processor(obj._span_processor)

                # Setup LoggerProvider for OTel logs
                log_exporter = LaminarLogExporter(
                    base_url=base_url,
                    port=http_port if force_http else port,
                    api_key=project_api_key,
                    timeout_seconds=timeout_seconds,
                    force_http=force_http,
                )
                obj._log_processor = BatchLogRecordProcessor(log_exporter)
                obj._logger_provider = LoggerProvider(resource=obj._resource)
                obj._logger_provider.add_log_record_processor(obj._log_processor)

                # Set global logger provider (follows same flag as tracer provider)
                if set_global_tracer_provider:
                    set_logger_provider(obj._logger_provider)

                # Setup threading context inheritance
                obj._setup_threading_inheritance()

                # This is not a real instrumentation and does not generate telemetry
                # data, but it is required to ensure that OpenTelemetry context
                # propagation is enabled.
                # See the README at:
                # https://pypi.org/project/opentelemetry-instrumentation-threading/
                ThreadingInstrumentor().instrument()

                init_instrumentations(
                    tracer_provider=obj._tracer_provider,
                    logger_provider=obj._logger_provider,
                    instruments=instruments,
                    block_instruments=block_instruments,
                    async_client=obj._async_client,
                )

                cls.instance = obj

                # Force flushes for debug environments (e.g. local development)
                atexit.register(obj.exit_handler)

            return cls.instance

    def _setup_threading_inheritance(self):
        """Setup threading inheritance for isolated context."""
        if TracerWrapper._original_thread_init is None:
            # Monkey patch Thread.__init__ to capture context inheritance
            TracerWrapper._original_thread_init = threading.Thread.__init__

            def patched_thread_init(thread_self, *args, **kwargs):
                # Capture current isolated context and token stack for inheritance
                current_context = get_current_context()
                current_token_stack = get_token_stack().copy()

                # Get the original target function
                original_target = kwargs.get("target")
                if not original_target and args:
                    original_target = args[0]

                # Only inherit if we have a target function
                if original_target:
                    # Create a wrapper function that sets up context
                    def thread_wrapper(*target_args, **target_kwargs):
                        # Set inherited context and token stack in the new thread
                        attach_context(current_context)
                        set_token_stack(current_token_stack)
                        # Run original target
                        return original_target(*target_args, **target_kwargs)

                    # Replace the target with our wrapper
                    if "target" in kwargs:
                        kwargs["target"] = thread_wrapper
                    elif args:
                        args = (thread_wrapper,) + args[1:]

                # Call original init
                TracerWrapper._original_thread_init(thread_self, *args, **kwargs)

            threading.Thread.__init__ = patched_thread_init

    def exit_handler(self):
        if isinstance(self._span_processor, LaminarSpanProcessor):
            self._span_processor.clear()
        self.flush()

    def _initialize_logger(self):
        self._logger = logging.getLogger(__name__)
        console_log_handler = logging.StreamHandler()
        console_log_handler.setFormatter(VerboseColorfulFormatter())
        self._logger.addHandler(console_log_handler)

    def get_isolated_context(self) -> Context:
        """Get the current isolated context."""
        return get_current_context()

    def push_span_context(self, span: trace.Span) -> Context:
        """Push a new context with the given span onto the stack."""
        current_ctx = get_current_context()
        new_context = trace.set_span_in_context(span, current_ctx)
        ctx_push_span_context(new_context)
        return new_context

    def pop_span_context(self) -> None:
        """Pop the current span context from the stack."""
        ctx_pop_span_context()

    @staticmethod
    def set_static_params(
        resource_attributes: dict,
        enable_content_tracing: bool,
    ) -> None:
        TracerWrapper.resource_attributes = resource_attributes
        TracerWrapper.enable_content_tracing = enable_content_tracing

    @classmethod
    def verify_initialized(cls) -> bool:
        # This is not using lock, but it is fine to return False from here
        # even if initialization is going on.

        # If we try to acquire the lock here, it may deadlock if an automatic
        # instrumentation is importing a file that (at the top level) has a
        # function annotated with Laminar's `observe` decorator.
        # The decorator is evaluated at import time, inside `init_instrumentations`,
        # which is called by the `TracerWrapper` constructor while holding the lock.
        # Without the lock here, we will simply return False, which will cause
        # the decorator to return the original function. This is fine, at runtime,
        # the next import statement will re-evaluate the decorator, and Laminar will
        # have been initialized by that time.
        return hasattr(cls, "instance") and hasattr(cls.instance, "_span_processor")

    @classmethod
    def clear(cls):
        if not cls.verify_initialized():
            return
        # Any state cleanup. Now used in between tests
        if isinstance(cls.instance._span_processor, LaminarSpanProcessor):
            cls.instance._span_processor.clear()
        # Clear the isolated context state for clean test state
        clear_context()

    def shutdown(self):
        if self._tracer_provider is not None:
            self._tracer_provider.shutdown()
        if self._logger_provider is not None:
            self._logger_provider.shutdown()

    def flush(self):
        if not hasattr(self, "_span_processor"):
            self._logger.warning("TracerWrapper not fully initialized, cannot flush")
            return False
        return self._span_processor.force_flush() and (
            self._log_processor.force_flush() if self._log_processor else True
        )

    def force_reinit_processor(self):
        if isinstance(self._span_processor, LaminarSpanProcessor):
            self._span_processor.force_flush()
            self._span_processor.force_reinit()
            if self._log_processor:
                self._log_processor.force_flush()
            # Clear the isolated context to prevent subsequent invocations
            # (e.g., in Lambda) from continuing traces from previous invocations
            clear_context()
        else:
            self._logger.warning(
                "Not using LaminarSpanProcessor, cannot force reinit processor"
            )

    @classmethod
    def get_session_recording_options(cls) -> SessionRecordingOptions:
        """Get the session recording options set during initialization."""
        return cls.session_recording_options

    def get_tracer(self) -> trace.Tracer:
        if self._tracer_provider is None:
            return trace.get_tracer_provider().get_tracer(TRACER_NAME)
        return self._tracer_provider.get_tracer(TRACER_NAME)

import atexit
import logging
import threading

from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.log import VerboseColorfulFormatter
from lmnr.opentelemetry_lib.tracing.instruments import (
    Instruments,
    init_instrumentations,
)
from lmnr.opentelemetry_lib.tracing.context import (
    attach_context,
    detach_context,
    get_current_context,
    get_token_stack,
    _isolated_token_stack,
    _isolated_token_stack_storage,
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

TRACER_NAME = "lmnr.tracer"

MAX_EVENTS_OR_ATTRIBUTES_PER_SPAN = 5000


class TracerWrapper(object):
    resource_attributes: dict = {}
    enable_content_tracing: bool = True
    _lock = threading.Lock()
    _tracer_provider: TracerProvider | None = None
    _logger: logging.Logger
    _client: LaminarClient
    _async_client: AsyncLaminarClient
    _resource: Resource
    _span_processor: SpanProcessor
    _original_thread_init = None

    def __new__(
        cls,
        disable_batch=False,
        exporter: SpanExporter | None = None,
        instruments: set[Instruments] | None = None,
        block_instruments: set[Instruments] | None = None,
        base_url: str = "https://api.lmnr.ai",
        port: int = 8443,
        http_port: int = 443,
        project_api_key: str | None = None,
        max_export_batch_size: int | None = None,
        force_http: bool = False,
        timeout_seconds: int = 30,
        set_global_tracer_provider: bool = True,
        otel_logger_level: int = logging.ERROR,
    ) -> "TracerWrapper":
        # Silence some opentelemetry warnings
        logging.getLogger("opentelemetry.trace").setLevel(otel_logger_level)

        base_http_url = f"{base_url}:{http_port}"
        with cls._lock:
            if not hasattr(cls, "instance"):
                cls._initialize_logger(cls)
                obj = super(TracerWrapper, cls).__new__(cls)

                obj._client = LaminarClient(
                    base_url=base_http_url,
                    project_api_key=project_api_key,
                )
                obj._async_client = AsyncLaminarClient(
                    base_url=base_http_url,
                    project_api_key=project_api_key,
                )

                obj._resource = Resource(attributes=TracerWrapper.resource_attributes)

                obj._span_processor = LaminarSpanProcessor(
                    base_url=base_url,
                    api_key=project_api_key,
                    port=http_port if force_http else port,
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
                    instruments=instruments,
                    block_instruments=block_instruments,
                    client=obj._client,
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
        token = attach_context(new_context)

        # Store the token for later detachment - tokens are much lighter than contexts
        current_stack = get_token_stack().copy()
        current_stack.append(token)
        set_token_stack(current_stack)

        return new_context

    def pop_span_context(self) -> None:
        """Pop the current span context from the stack."""
        current_stack = get_token_stack().copy()
        if current_stack:
            token = current_stack.pop()
            set_token_stack(current_stack)
            detach_context(token)

    @staticmethod
    def set_static_params(
        resource_attributes: dict,
        enable_content_tracing: bool,
    ) -> None:
        TracerWrapper.resource_attributes = resource_attributes
        TracerWrapper.enable_content_tracing = enable_content_tracing

    @classmethod
    def verify_initialized(cls) -> bool:
        with cls._lock:
            return hasattr(cls, "instance") and hasattr(cls.instance, "_span_processor")

    @classmethod
    def clear(cls):
        if not cls.verify_initialized():
            return
        # Any state cleanup. Now used in between tests
        if isinstance(cls.instance._span_processor, LaminarSpanProcessor):
            cls.instance._span_processor.clear()
        # Clear the isolated context state for clean test state
        try:
            _isolated_token_stack.set([])
        except LookupError:
            pass
        if hasattr(_isolated_token_stack_storage, "token_stack"):
            _isolated_token_stack_storage.token_stack = []
        # Reset the isolated context to a fresh state
        attach_context(Context())

    def shutdown(self):
        if self._tracer_provider is None:
            return
        self._tracer_provider.shutdown()

    def flush(self):
        if not hasattr(self, "_span_processor"):
            self._logger.warning("TracerWrapper not fully initialized, cannot flush")
            return False
        return self._span_processor.force_flush()

    def get_tracer(self):
        if self._tracer_provider is None:
            return trace.get_tracer_provider().get_tracer(TRACER_NAME)
        return self._tracer_provider.get_tracer(TRACER_NAME)

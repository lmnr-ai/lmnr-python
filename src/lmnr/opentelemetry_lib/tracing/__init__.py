import atexit
import logging
import threading
from contextlib import contextmanager
from typing import Optional

from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.log import VerboseColorfulFormatter
from lmnr.opentelemetry_lib.tracing.instruments import (
    Instruments,
    init_instrumentations,
)

from opentelemetry import trace
from opentelemetry import context as context_api
from opentelemetry.context import Context
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter

TRACER_NAME = "lmnr.tracer"

MAX_EVENTS_OR_ATTRIBUTES_PER_SPAN = 5000

# Thread-local storage for isolated context stack
_isolated_context_storage = threading.local()


def _get_context_stack():
    """Get the context stack for the current thread."""
    if not hasattr(_isolated_context_storage, "context_stack"):
        # Initialize with empty context as the base
        _isolated_context_storage.context_stack = [context_api.Context()]
    return _isolated_context_storage.context_stack


def _push_context(context: Context):
    """Push a new context onto the stack."""
    stack = _get_context_stack()
    stack.append(context)


def _pop_context() -> Context:
    """Pop a context from the stack, but never pop the base context."""
    stack = _get_context_stack()
    if len(stack) > 1:
        return stack.pop()
    return stack[0]  # Return base context if stack is at minimum


def _current_context() -> Context:
    """Get the current context from the top of the stack."""
    stack = _get_context_stack()
    return stack[-1]


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
        timeout_seconds: int = 10,
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
        """Get the current isolated context from the context stack."""
        return _current_context()

    def push_span_context(self, span: trace.Span) -> None:
        """Push a new context with the given span onto the context stack."""
        current_ctx = _current_context()
        new_context = trace.set_span_in_context(span, current_ctx)
        _push_context(new_context)

    def pop_span_context(self) -> None:
        """Pop the current span context from the context stack."""
        _pop_context()

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
        # Clear the context stack for clean test state
        if hasattr(_isolated_context_storage, "context_stack"):
            _isolated_context_storage.context_stack = [context_api.Context()]

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

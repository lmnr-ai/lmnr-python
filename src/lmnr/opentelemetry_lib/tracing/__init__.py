import atexit
import logging

from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.log import VerboseColorfulFormatter
from lmnr.opentelemetry_lib.tracing.instruments import (
    Instruments,
    init_instrumentations,
)

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter

from typing import Optional, Set


module_logger = logging.getLogger(__name__)
console_log_handler = logging.StreamHandler()
console_log_handler.setFormatter(VerboseColorfulFormatter())
module_logger.addHandler(console_log_handler)


TRACER_NAME = "lmnr.tracer"

MAX_EVENTS_OR_ATTRIBUTES_PER_SPAN = 5000


class TracerWrapper(object):
    resource_attributes: dict = {}
    enable_content_tracing: bool = True
    __tracer_provider: TracerProvider
    __logger: logging.Logger
    __client: LaminarClient
    __async_client: AsyncLaminarClient
    __resource: Resource
    __span_processor: SpanProcessor

    def __new__(
        cls,
        disable_batch=False,
        exporter: Optional[SpanExporter] = None,
        instruments: Optional[Set[Instruments]] = None,
        block_instruments: Optional[Set[Instruments]] = None,
        base_url: str = "https://api.lmnr.ai",
        port: int = 8443,
        http_port: int = 443,
        project_api_key: Optional[str] = None,
        max_export_batch_size: Optional[int] = None,
        force_http: bool = False,
        timeout_seconds: int = 10,
    ) -> "TracerWrapper":
        base_http_url = f"http://{base_url}:{http_port}"
        cls._initialize_logger(cls)
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(TracerWrapper, cls).__new__(cls)

            obj.__client = LaminarClient(
                base_url=base_http_url,
                project_api_key=project_api_key,
            )
            obj.__async_client = AsyncLaminarClient(
                base_url=base_http_url,
                project_api_key=project_api_key,
            )

            obj.__resource = Resource(attributes=TracerWrapper.resource_attributes)
            obj.__tracer_provider = TracerProvider(resource=obj.__resource)

            obj.__span_processor = LaminarSpanProcessor(
                base_url=base_url,
                api_key=project_api_key,
                port=http_port if force_http else port,
                exporter=exporter,
                max_export_batch_size=max_export_batch_size,
                timeout_seconds=timeout_seconds,
                force_http=force_http,
                disable_batch=disable_batch,
            )

            obj.__tracer_provider.add_span_processor(obj.__span_processor)

            init_instrumentations(
                tracer_provider=obj.__tracer_provider,
                instruments=instruments,
                block_instruments=block_instruments,
                client=obj.__client,
                async_client=obj.__async_client,
            )

            # Force flushes for debug environments (e.g. local development)
            atexit.register(obj.exit_handler)

        return cls.instance

    def exit_handler(self):
        if isinstance(self.__span_processor, LaminarSpanProcessor):
            self.__span_processor.clear()
        self.flush()

    def _initialize_logger(self):
        self.__logger = logging.getLogger(__name__)
        console_log_handler = logging.StreamHandler()
        console_log_handler.setFormatter(VerboseColorfulFormatter())
        self.__logger.addHandler(console_log_handler)

    @staticmethod
    def set_static_params(
        resource_attributes: dict,
        enable_content_tracing: bool,
    ) -> None:
        TracerWrapper.resource_attributes = resource_attributes
        TracerWrapper.enable_content_tracing = enable_content_tracing

    @classmethod
    def verify_initialized(cls) -> bool:
        return hasattr(cls, "instance")

    @classmethod
    def clear(cls):
        # Any state cleanup. Now used in between tests
        if isinstance(cls.instance.__span_processor, LaminarSpanProcessor):
            cls.instance.__span_processor.clear()

    def shutdown(self):
        self.__tracer_provider.shutdown()

    def flush(self):
        return self.__span_processor.force_flush()

    def get_tracer(self):
        return self.__tracer_provider.get_tracer(TRACER_NAME)

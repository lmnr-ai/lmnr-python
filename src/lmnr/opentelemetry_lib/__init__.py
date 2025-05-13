import sys

from typing import Optional, Set
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME

from lmnr.opentelemetry_lib.tracing.instruments import Instruments
from lmnr.opentelemetry_lib.tracing import TracerWrapper

MAX_MANUAL_SPAN_PAYLOAD_SIZE = 1024 * 1024  # 1MB


class TracerManager:
    __tracer_wrapper: TracerWrapper

    @staticmethod
    def init(
        app_name: Optional[str] = sys.argv[0],
        disable_batch=False,
        exporter: Optional[SpanExporter] = None,
        resource_attributes: dict = {},
        instruments: Optional[Set[Instruments]] = None,
        base_url: str = "https://api.lmnr.ai",
        port: int = 8443,
        http_port: int = 443,
        project_api_key: Optional[str] = None,
        max_export_batch_size: Optional[int] = None,
        force_http: bool = False,
        timeout_seconds: int = 30,
    ) -> None:
        enable_content_tracing = True

        # Tracer init
        resource_attributes.update({SERVICE_NAME: app_name})
        TracerWrapper.set_static_params(resource_attributes, enable_content_tracing)
        TracerManager.__tracer_wrapper = TracerWrapper(
            disable_batch=disable_batch,
            exporter=exporter,
            instruments=instruments,
            base_url=base_url,
            port=port,
            http_port=http_port,
            project_api_key=project_api_key,
            max_export_batch_size=max_export_batch_size,
            force_http=force_http,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def flush() -> bool:
        return TracerManager.__tracer_wrapper.flush()

    @staticmethod
    def shutdown():
        TracerManager.__tracer_wrapper.shutdown()

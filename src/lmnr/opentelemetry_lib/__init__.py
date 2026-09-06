import logging
import sys
from collections.abc import Sequence

from opentelemetry.sdk.resources import SERVICE_NAME
from opentelemetry.sdk.trace.export import SpanExporter

from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.opentelemetry_lib.tracing.instruments import Instruments
from lmnr.sdk.types import SessionRecordingOptions


class TracerManager:
    __tracer_wrapper: TracerWrapper

    @staticmethod
    def init(
        app_name: str | None = sys.argv[0],
        disable_batch=False,
        exporter: SpanExporter | None = None,
        resource_attributes: dict[str, str | int | float | bool | Sequence[str] | Sequence[int | float] | Sequence[bool]] | None = None,
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
    ) -> None:
        enable_content_tracing = True

        # Tracer init
        attributes = {**(resource_attributes or {}), SERVICE_NAME: app_name}
        TracerWrapper.set_static_params(attributes, enable_content_tracing)
        TracerManager.__tracer_wrapper = TracerWrapper(
            disable_batch=disable_batch,
            exporter=exporter,
            instruments=instruments,
            block_instruments=block_instruments,
            base_url=base_url,
            port=port,
            http_port=http_port,
            project_api_key=project_api_key,
            max_export_batch_size=max_export_batch_size,
            max_export_batch_size_bytes=max_export_batch_size_bytes,
            flush_by_size=flush_by_size,
            force_http=force_http,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else 30,
            set_global_tracer_provider=set_global_tracer_provider,
            otel_logger_level=otel_logger_level,
            session_recording_options=session_recording_options,
        )

    @staticmethod
    def flush() -> bool:
        if not hasattr(TracerManager, "_TracerManager__tracer_wrapper"):
            return False
        return TracerManager.__tracer_wrapper.flush()

    @staticmethod
    def shutdown():
        TracerManager.__tracer_wrapper.shutdown()

    @staticmethod
    def force_reinit_processor() -> bool:
        if not hasattr(TracerManager, "_TracerManager__tracer_wrapper"):
            return False
        return TracerManager.__tracer_wrapper.force_reinit_processor()

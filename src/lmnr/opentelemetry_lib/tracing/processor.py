import logging
import threading
import uuid

from opentelemetry.sdk.trace.export import (
    SpanProcessor,
    SpanExporter,
    BatchSpanProcessor,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace import Span
from opentelemetry.context import Context, get_value

from lmnr.opentelemetry_lib.tracing.attributes import (
    PARENT_SPAN_IDS_PATH,
    PARENT_SPAN_PATH,
    SPAN_IDS_PATH,
    SPAN_INSTRUMENTATION_SOURCE,
    SPAN_LANGUAGE_VERSION,
    SPAN_PATH,
    SPAN_SDK_VERSION,
)
from lmnr.opentelemetry_lib.tracing.exporter import LaminarSpanExporter
from lmnr.sdk.log import get_default_logger
from lmnr.version import PYTHON_VERSION, __version__


class LaminarSpanProcessor(SpanProcessor):
    instance: BatchSpanProcessor | SimpleSpanProcessor
    logger: logging.Logger
    __span_id_to_path: dict[int, list[str]] = {}
    __span_id_lists: dict[int, list[str]] = {}
    max_export_batch_size: int
    _instance_lock: threading.RLock
    _paths_lock: threading.RLock

    def __init__(
        self,
        base_url: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 30,
        force_http: bool = False,
        max_export_batch_size: int = 64,
        disable_batch: bool = False,
        exporter: SpanExporter | None = None,
    ):
        self._instance_lock = threading.RLock()
        self._paths_lock = threading.RLock()
        self.logger = get_default_logger(__name__)
        self.max_export_batch_size = max_export_batch_size
        self.exporter = exporter or LaminarSpanExporter(
            base_url=base_url,
            port=port,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            force_http=force_http,
        )
        self.instance = (
            SimpleSpanProcessor(self.exporter)
            if disable_batch
            else BatchSpanProcessor(
                self.exporter, max_export_batch_size=max_export_batch_size
            )
        )

    def on_start(self, span: Span, parent_context: Context | None = None):
        with self._paths_lock:
            parent_span_path = list(span.attributes.get(PARENT_SPAN_PATH, tuple())) or (
                self.__span_id_to_path.get(span.parent.span_id) if span.parent else None
            )
            parent_span_ids_path = list(
                span.attributes.get(PARENT_SPAN_IDS_PATH, tuple())
            ) or (
                self.__span_id_lists.get(span.parent.span_id, []) if span.parent else []
            )
            span_path = (
                parent_span_path + [span.name] if parent_span_path else [span.name]
            )
            span_ids_path = parent_span_ids_path + [
                str(uuid.UUID(int=span.get_span_context().span_id))
            ]
            span.set_attribute(SPAN_PATH, span_path)
            span.set_attribute(SPAN_IDS_PATH, span_ids_path)
            self.__span_id_to_path[span.get_span_context().span_id] = span_path
            self.__span_id_lists[span.get_span_context().span_id] = span_ids_path

        span.set_attribute(SPAN_INSTRUMENTATION_SOURCE, "python")
        span.set_attribute(SPAN_SDK_VERSION, __version__)
        span.set_attribute(SPAN_LANGUAGE_VERSION, f"python@{PYTHON_VERSION}")

        if span.name == "LangGraph.workflow":
            graph_context = get_value("lmnr.langgraph.graph") or {}
            for key, value in graph_context.items():
                span.set_attribute(f"lmnr.association.properties.{key}", value)

        with self._instance_lock:
            self.instance.on_start(span, parent_context)

    def on_end(self, span: Span):
        with self._instance_lock:
            self.instance.on_end(span)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        with self._instance_lock:
            return self.instance.force_flush(timeout_millis)

    def force_reinit(self):
        if not isinstance(self.exporter, LaminarSpanExporter):
            self.logger.warning(
                "LaminarSpanProcessor is not using LaminarSpanExporter, cannot force reinit"
            )
            return

        # Reinitialize exporter (thread-safe, handles its own locking)
        self.exporter._init_instance()

        with self._instance_lock:
            old_instance = self.instance
            disable_batch = isinstance(old_instance, SimpleSpanProcessor)

            try:
                old_instance.shutdown()
            except Exception as e:
                self.logger.debug(f"Error shutting down old processor instance: {e}")

            self.instance = (
                SimpleSpanProcessor(self.exporter)
                if disable_batch
                else BatchSpanProcessor(
                    self.exporter, max_export_batch_size=self.max_export_batch_size
                )
            )

    def shutdown(self):
        with self._instance_lock:
            self.instance.shutdown()

    def clear(self):
        with self._paths_lock:
            self.__span_id_to_path = {}
            self.__span_id_lists = {}

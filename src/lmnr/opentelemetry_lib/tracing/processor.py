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
    ASSOCIATION_PROPERTIES,
    PARENT_SPAN_IDS_PATH,
    PARENT_SPAN_PATH,
    SESSION_ID,
    SPAN_IDS_PATH,
    SPAN_INSTRUMENTATION_SOURCE,
    SPAN_LANGUAGE_VERSION,
    SPAN_PATH,
    SPAN_SDK_VERSION,
    TRACE_TYPE,
    USER_ID,
)
from lmnr.opentelemetry_lib.tracing.context import (
    CONTEXT_METADATA_KEY,
    CONTEXT_SESSION_ID_KEY,
    CONTEXT_TRACE_TYPE_KEY,
    CONTEXT_USER_ID_KEY,
)
from lmnr.opentelemetry_lib.tracing.exporter import LaminarSpanExporter
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import from_env, is_otel_attribute_value_type, json_dumps
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
        is_disabled = (
            from_env("LMNR_DISABLE_TRACING") or "false"
        ).lower().strip() == "true"

        if is_disabled:
            span.set_attribute("lmnr.internal.disabled", True)

        with self._paths_lock:
            parent_span_path = list(span.attributes.get(PARENT_SPAN_PATH, tuple())) or (
                self.__span_id_to_path.get(span.parent.span_id) if span.parent else None
            )
            parent_span_ids_path = list(
                span.attributes.get(PARENT_SPAN_IDS_PATH, tuple())
            ) or (
                self.__span_id_lists.get(span.parent.span_id, []) if span.parent else []
            )
            span_name_in_path = span.name if not is_disabled else "_"
            span_path = (
                parent_span_path + [span_name_in_path]
                if parent_span_path
                else [span_name_in_path]
            )
            span_ids_path = parent_span_ids_path + [
                str(uuid.UUID(int=span.get_span_context().span_id))
            ]
            span.set_attribute(SPAN_PATH, span_path)
            span.set_attribute(SPAN_IDS_PATH, span_ids_path)
            self.__span_id_to_path[span.get_span_context().span_id] = span_path
            self.__span_id_lists[span.get_span_context().span_id] = span_ids_path

        if is_disabled:
            return

        span.set_attribute(SPAN_INSTRUMENTATION_SOURCE, "python")
        span.set_attribute(SPAN_SDK_VERSION, __version__)
        span.set_attribute(SPAN_LANGUAGE_VERSION, f"python@{PYTHON_VERSION}")

        if parent_context:
            trace_type = get_value(CONTEXT_TRACE_TYPE_KEY, parent_context)
            if trace_type:
                span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{TRACE_TYPE}", trace_type)
            user_id = get_value(CONTEXT_USER_ID_KEY, parent_context)
            if user_id:
                span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{USER_ID}", user_id)
            session_id = get_value(CONTEXT_SESSION_ID_KEY, parent_context)
            if session_id:
                span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{SESSION_ID}", session_id)
            ctx_metadata = get_value(CONTEXT_METADATA_KEY, parent_context)
            if ctx_metadata and isinstance(ctx_metadata, dict):
                span_metadata = {}
                if hasattr(span, "attributes") and hasattr(span.attributes, "items"):
                    for key, value in span.attributes.items():
                        if key.startswith(f"{ASSOCIATION_PROPERTIES}.metadata."):
                            span_metadata[
                                key.replace(f"{ASSOCIATION_PROPERTIES}.metadata.", "")
                            ] = value

                for key, value in {**ctx_metadata, **span_metadata}.items():
                    span.set_attribute(
                        f"{ASSOCIATION_PROPERTIES}.metadata.{key}",
                        (
                            value
                            if is_otel_attribute_value_type(value)
                            else json_dumps(value)
                        ),
                    )

        if span.name == "LangGraph.workflow":
            graph_context = get_value("lmnr.langgraph.graph") or {}
            for key, value in graph_context.items():
                span.set_attribute(f"lmnr.association.properties.{key}", value)

        with self._instance_lock:
            self.instance.on_start(span, parent_context)

    def on_end(self, span: Span):
        if (from_env("LMNR_DISABLE_TRACING") or "false").lower().strip() == "true" or (
            span.attributes and span.attributes.get("lmnr.internal.disabled")
        ):
            return
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

        with self._instance_lock:
            old_instance = self.instance
            disable_batch = isinstance(old_instance, SimpleSpanProcessor)

            # CRITICAL: Shutdown old processor FIRST (uses old exporter)
            # This flushes pending spans and joins the daemon thread.
            # Only then, we can reinitialize the exporter.
            try:
                old_instance.shutdown()
            except Exception as e:
                self.logger.debug(f"Error shutting down old processor instance: {e}")

            # reinitialize the exporter
            # This is thread-safe as it has its own locking
            self.exporter._init_instance()

            # Create new processor with fresh exporter
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

    def set_parent_path_info(
        self,
        parent_span_id: int,
        span_path: list[str],
        span_ids_path: list[str],
    ):
        with self._paths_lock:
            self.__span_id_to_path[parent_span_id] = span_path
            self.__span_id_lists[parent_span_id] = span_ids_path

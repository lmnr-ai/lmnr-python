import logging
import threading
import uuid
from typing import TYPE_CHECKING

from opentelemetry.sdk.trace.export import (
    SpanProcessor,
    SpanExporter,
    BatchSpanProcessor,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace import Span
from opentelemetry.context import Context, get_value

if TYPE_CHECKING:
    from lmnr.sdk.client.synchronous.sync_client import LaminarClient

from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    PARENT_SPAN_IDS_PATH,
    PARENT_SPAN_PATH,
    ROLLOUT_SESSION_ID,
    ROLLOUT_SESSION_ID_ATTR,
    SESSION_ID,
    SPAN_IDS_PATH,
    SPAN_INSTRUMENTATION_SOURCE,
    SPAN_LANGUAGE_VERSION,
    SPAN_PATH,
    SPAN_SDK_VERSION,
    SPAN_TYPE,
    TRACE_TYPE,
    USER_ID,
)
from lmnr.opentelemetry_lib.tracing.context import (
    CONTEXT_METADATA_KEY,
    CONTEXT_ROLLOUT_SESSION_ID_KEY,
    CONTEXT_SESSION_ID_KEY,
    CONTEXT_TRACE_TYPE_KEY,
    CONTEXT_USER_ID_KEY,
)
from lmnr.opentelemetry_lib.tracing.exporter import LaminarSpanExporter
from lmnr.sdk.client.types import SpanStartData
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.rollout_control import is_rollout_mode
from lmnr.sdk.utils import is_otel_attribute_value_type, json_dumps
from lmnr.version import PYTHON_VERSION, __version__


class LaminarSpanProcessor(SpanProcessor):
    instance: BatchSpanProcessor | SimpleSpanProcessor
    logger: logging.Logger
    __span_id_to_path: dict[int, list[str]] = {}
    __span_id_lists: dict[int, list[str]] = {}
    max_export_batch_size: int
    _instance_lock: threading.RLock
    _paths_lock: threading.RLock
    _rollout_client: "LaminarClient | None"

    def __init__(
        self,
        base_url: str | None = None,
        http_port: int | None = None,
        grpc_port: int | None = None,
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
        self._rollout_client = None
        port = http_port if force_http else grpc_port
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

        if is_rollout_mode():
            try:
                from lmnr.sdk.client.synchronous.sync_client import LaminarClient

                self._rollout_client = LaminarClient(
                    base_url=base_url,
                    project_api_key=api_key,
                    port=http_port,
                )
                self.logger.debug("Rollout client initialized for span streaming")
            except Exception as e:
                self.logger.debug(f"Failed to initialize rollout client: {e}")

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
            rollout_session_id = get_value(
                CONTEXT_ROLLOUT_SESSION_ID_KEY, parent_context
            )
            if rollout_session_id:
                # Set as both lmnr.rollout.session_id and lmnr.association.properties.rollout_session_id
                span.set_attribute(ROLLOUT_SESSION_ID_ATTR, rollout_session_id)
                span.set_attribute(
                    f"{ASSOCIATION_PROPERTIES}.{ROLLOUT_SESSION_ID}", rollout_session_id
                )
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

        # Check if in rollout mode and stream span in real-time
        if is_rollout_mode() and self._rollout_client:
            self._send_span_update_for_rollout(span)

        with self._instance_lock:
            self.instance.on_start(span, parent_context)

    def on_end(self, span: Span):

        with self._instance_lock:
            self.instance.on_end(span)

    def _infer_span_type(self, span: Span) -> str:
        """
        Infer the span type from span attributes.

        Args:
            span: The span to infer type for

        Returns:
            Span type: "DEFAULT", "LLM", or "TOOL"
        """
        attrs = span.attributes or {}

        # Check if span type is explicitly set
        if SPAN_TYPE in attrs:
            return str(attrs[SPAN_TYPE])

        # Check for LLM indicators
        if "gen_ai.system" in attrs:
            return "LLM"

        # Check if any attribute starts with gen_ai. or llm.
        if any(k.startswith("gen_ai.") or k.startswith("llm.") for k in attrs.keys()):
            return "LLM"

        return "DEFAULT"

    def _attributes_for_start_event(self, span: Span) -> dict:
        """
        Strip span attributes to only include keys relevant for start events.

        Args:
            span: The span to extract attributes from

        Returns:
            Filtered attributes dict
        """
        attrs = span.attributes or {}

        # Keys to keep in start events
        keys_to_keep = [
            SPAN_PATH,
            SPAN_TYPE,
            SPAN_IDS_PATH,
            PARENT_SPAN_PATH,
            PARENT_SPAN_IDS_PATH,
            SPAN_INSTRUMENTATION_SOURCE,
            SPAN_SDK_VERSION,
            SPAN_LANGUAGE_VERSION,
            "gen_ai.request.model",
            "gen_ai.response.model",
            "gen_ai.system",
            "lmnr.span.original_type",
            "ai.model.id",
        ]

        # Filter to only keep specified keys
        new_attributes = {k: v for k, v in attrs.items() if k in keys_to_keep}

        # If we have request model but no response model, copy it over
        # This helps the UI show the model even before the response
        if "gen_ai.request.model" in attrs and "gen_ai.response.model" not in attrs:
            new_attributes["gen_ai.response.model"] = attrs["gen_ai.request.model"]

        # Always include span type
        new_attributes[SPAN_TYPE] = self._infer_span_type(span)

        return new_attributes

    def _send_span_update_for_rollout(self, span: Span) -> None:
        """
        Stream span data in real-time if in rollout mode.

        This provides live updates to the Laminar UI during rollout sessions.

        Args:
            span: The span to stream
        """
        try:
            from lmnr.sdk.rollout_control import get_rollout_session_id
            import datetime

            if not self._rollout_client:
                return

            session_id = get_rollout_session_id()
            if not session_id:
                return

            if span.name == "cdp_use.session":
                # ignore the signal span that's also ignored in the backend
                return

            start_time_ns = span.start_time
            start_time_seconds = start_time_ns / 1e9
            start_time_dt = datetime.datetime.fromtimestamp(
                start_time_seconds, tz=datetime.timezone.utc
            )

            # Prepare span data
            span_data: SpanStartData = {
                "name": span.name,
                "start_time": start_time_dt,
                "span_id": uuid.UUID(int=span.context.span_id),
                "trace_id": uuid.UUID(int=span.context.trace_id),
                "parent_span_id": (
                    uuid.UUID(int=span.parent.span_id)
                    if span.parent and span.parent.span_id
                    else None
                ),
                "attributes": self._attributes_for_start_event(span),
                "span_type": self._infer_span_type(span),
            }

            self._send_span_start(session_id, span_data)

        except Exception as e:
            # Log but don't raise - streaming failures shouldn't break tracing
            self.logger.debug(f"Failed to stream span for rollout: {e}")

    def _send_span_start(self, session_id: str, span_data: SpanStartData) -> None:
        """
        Stream span data asynchronously without blocking.

        Args:
            session_id: Rollout session ID
            span_data: Span data to stream
        """
        import atexit

        try:

            def stream_task():
                try:
                    self._rollout_client.rollout.update_span_info(session_id, span_data)
                except Exception as e:
                    self.logger.debug(f"Error in span streaming task: {e}")

            # Run in background thread to avoid blocking
            thread = threading.Thread(target=stream_task, daemon=True)
            thread.start()
            atexit.register(thread.join)

        except Exception as e:
            self.logger.debug(f"Failed to start span streaming thread: {e}")

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

    def set_parent_path_info(
        self,
        parent_span_id: int,
        span_path: list[str],
        span_ids_path: list[str],
    ):
        with self._paths_lock:
            self.__span_id_to_path[parent_span_id] = span_path
            self.__span_id_lists[parent_span_id] = span_ids_path

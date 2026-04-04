import logging
import threading
import time
import uuid
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING

from opentelemetry import trace as trace_api
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
    CONTEXT_SPAN_IDS_PATH_KEY,
    CONTEXT_SPAN_PATH_KEY,
    CONTEXT_TRACE_TYPE_KEY,
    CONTEXT_USER_ID_KEY,
)
from lmnr.opentelemetry_lib.tracing.exporter import LaminarSpanExporter
from lmnr.sdk.client.types import SpanStartData
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.rollout_control import is_rollout_mode
from lmnr.sdk.utils import from_env, is_otel_attribute_value_type, json_dumps
from lmnr.version import PYTHON_VERSION, __version__

from typing_extensions import TypedDict

# Maximum number of span-path entries kept in the bounded in-memory fallback.
_MAX_PATH_CACHE_ENTRIES = 50_000

# Traces that have not had any new spans within this many seconds are eligible
# for eviction from the in-memory cache.
_TRACE_IDLE_TIMEOUT_SECONDS = 1800

# Minimum interval (seconds) between stale-trace eviction scans to avoid
# O(T) iteration on every span start.
_EVICTION_INTERVAL_SECONDS = 60


class _PathEntry(TypedDict):
    span_path: list[str]
    span_ids_path: list[str]


class LaminarSpanProcessor(SpanProcessor):
    instance: BatchSpanProcessor | SimpleSpanProcessor
    logger: logging.Logger
    max_export_batch_size: int
    _instance_lock: threading.RLock
    _paths_lock: threading.RLock
    _rollout_client: "LaminarClient | None"

    # Bounded in-memory fallback: span_id (int) -> _PathEntry
    _path_cache: OrderedDict[int, _PathEntry]
    # trace_id (int) -> set of span_ids (int) belonging to this trace
    _trace_spans: dict[int, set[int]]
    # span_id (int) -> trace_id (int) reverse mapping for O(1) eviction
    _span_trace: dict[int, int]
    # trace_id (int) -> last access timestamp (monotonic seconds)
    _trace_last_access: dict[int, float]

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
        self._path_cache = OrderedDict()
        self._trace_spans = {}
        self._span_trace = {}
        self._trace_last_access = {}
        self._last_eviction_time: float = 0.0
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
                self.logger.debug("Debugger client initialized for span streaming")
            except Exception as e:
                self.logger.debug(f"Failed to initialize debugger client: {e}")

    def _evict_stale_traces(self) -> None:
        """Remove cache entries for traces idle longer than the timeout.

        Must be called while holding ``_paths_lock``.  The scan is
        amortized: it only runs at most once every
        ``_EVICTION_INTERVAL_SECONDS`` to avoid an O(T) iteration on
        every span start.
        """
        now = time.monotonic()
        if now - self._last_eviction_time < _EVICTION_INTERVAL_SECONDS:
            return
        self._last_eviction_time = now

        stale_trace_ids = [
            tid
            for tid, ts in self._trace_last_access.items()
            if now - ts > _TRACE_IDLE_TIMEOUT_SECONDS
        ]
        for tid in stale_trace_ids:
            for sid in self._trace_spans.pop(tid, set()):
                self._path_cache.pop(sid, None)
                self._span_trace.pop(sid, None)
            self._trace_last_access.pop(tid, None)

    def _cache_put(
        self, span_id: int, trace_id: int, entry: _PathEntry
    ) -> None:
        """Insert an entry into the bounded path cache.

        Must be called while holding ``_paths_lock``.
        """
        # Lazy eviction of stale traces (amortized, not on every write)
        self._evict_stale_traces()

        # Prune oldest entries if at cap
        while len(self._path_cache) >= _MAX_PATH_CACHE_ENTRIES:
            evicted_sid, _ = self._path_cache.popitem(last=False)
            # O(1) cleanup via reverse mapping
            evicted_tid = self._span_trace.pop(evicted_sid, None)
            if evicted_tid is not None:
                sids = self._trace_spans.get(evicted_tid)
                if sids is not None:
                    sids.discard(evicted_sid)
                    if not sids:
                        del self._trace_spans[evicted_tid]
                        self._trace_last_access.pop(evicted_tid, None)

        self._path_cache[span_id] = entry
        self._trace_spans.setdefault(trace_id, set()).add(span_id)
        self._span_trace[span_id] = trace_id
        self._trace_last_access[trace_id] = time.monotonic()

    def _cache_get(self, span_id: int) -> _PathEntry | None:
        """Look up a path entry from the cache.

        Must be called while holding ``_paths_lock``.
        """
        return self._path_cache.get(span_id)

    def _resolve_parent_path(
        self,
        span: Span,
        parent_context: Context | None,
    ) -> tuple[list[str], list[str]]:
        """Resolve the parent span path and ids path.

        Priority:
          1. OTel Context – read SPAN_PATH / SPAN_IDS_PATH from the parent
             span that lives in the context (works for normal nesting where
             the parent span is still alive).
          2. OTel Context values – read CONTEXT_SPAN_PATH_KEY /
             CONTEXT_SPAN_IDS_PATH_KEY stored directly in the context (covers
             NonRecordingSpan parents, e.g. LMNR_SPAN_CONTEXT env var).
          3. Span attributes – read PARENT_SPAN_PATH / PARENT_SPAN_IDS_PATH
             set by the caller on the *current* span (covers remote / ended
             span contexts passed via LaminarSpanContext).
          4. Bounded in-memory cache – look up by parent span_id (covers the
             edge-case of a raw OTel SpanContext being passed as parent).
        """
        parent_span_path: list[str] = []
        parent_span_ids_path: list[str] = []

        # 1. Try the parent span from OTel context
        if parent_context is not None:
            parent_span = trace_api.get_current_span(parent_context)
            if (
                parent_span is not None
                and parent_span is not trace_api.INVALID_SPAN
                and hasattr(parent_span, "attributes")
                and parent_span.attributes is not None
            ):
                parent_span_path = list(
                    parent_span.attributes.get(SPAN_PATH, ())
                )
                parent_span_ids_path = list(
                    parent_span.attributes.get(SPAN_IDS_PATH, ())
                )

        # 2. Try context values (e.g. from LMNR_SPAN_CONTEXT env var)
        if not parent_span_path and parent_context is not None:
            ctx_path = get_value(CONTEXT_SPAN_PATH_KEY, parent_context)
            if ctx_path:
                parent_span_path = list(ctx_path)
            ctx_ids = get_value(CONTEXT_SPAN_IDS_PATH_KEY, parent_context)
            if ctx_ids:
                parent_span_ids_path = list(ctx_ids)

        # 3. Fall back to span attributes set by the caller
        if not parent_span_path:
            parent_span_path = list(
                span.attributes.get(PARENT_SPAN_PATH, ())
            )
            parent_span_ids_path = list(
                span.attributes.get(PARENT_SPAN_IDS_PATH, ())
            )

        # 4. Bounded in-memory cache (last resort)
        if not parent_span_path and span.parent:
            entry = self._cache_get(span.parent.span_id)
            if entry is not None:
                parent_span_path = list(entry["span_path"])
                parent_span_ids_path = list(entry["span_ids_path"])

        return parent_span_path, parent_span_ids_path

    def on_start(self, span: Span, parent_context: Context | None = None):
        is_disabled = (
            from_env("LMNR_DISABLE_TRACING") or "false"
        ).lower().strip() == "true"

        if is_disabled:
            span.set_attribute("lmnr.internal.disabled", True)

        with self._paths_lock:
            parent_span_path, parent_span_ids_path = self._resolve_parent_path(
                span, parent_context
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
            self._cache_put(
                span.get_span_context().span_id,
                span.get_span_context().trace_id,
                _PathEntry(span_path=span_path, span_ids_path=span_ids_path),
            )

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
        if (from_env("LMNR_DISABLE_TRACING") or "false").lower().strip() == "true" or (
            span.attributes and span.attributes.get("lmnr.internal.disabled")
        ):
            return
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
        try:

            def stream_task():
                try:
                    self._rollout_client.rollout.update_span_info(session_id, span_data)
                except Exception as e:
                    self.logger.debug(f"Error in span streaming task: {e}")

            # Run in background thread to avoid blocking
            thread = threading.Thread(target=stream_task, daemon=True)
            thread.start()

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
            self._path_cache.clear()
            self._trace_spans.clear()
            self._span_trace.clear()
            self._trace_last_access.clear()
            self._last_eviction_time = 0.0

    def set_parent_path_info(
        self,
        parent_span_id: int,
        span_path: list[str],
        span_ids_path: list[str],
    ):
        """Deprecated: path info is now propagated via OTel Context values.

        Callers should store path info in the OTel Context using
        ``CONTEXT_SPAN_PATH_KEY`` / ``CONTEXT_SPAN_IDS_PATH_KEY`` instead.
        """
        warnings.warn(
            "set_parent_path_info is deprecated and no longer stores data. "
            "Use CONTEXT_SPAN_PATH_KEY / CONTEXT_SPAN_IDS_PATH_KEY in the "
            "OTel Context instead.",
            DeprecationWarning,
            stacklevel=2,
        )

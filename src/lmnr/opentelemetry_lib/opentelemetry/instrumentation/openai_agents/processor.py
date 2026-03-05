"""LaminarAgentsTraceProcessor - mirrors OpenAI Agents spans into Laminar."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict

from lmnr import Laminar

try:
    from agents.tracing import TracingProcessor as _Base
except ImportError:  # openai-agents not installed
    _Base = object  # type: ignore[assignment,misc]

from .helpers import map_span_type, span_kind, span_name
from .span_data import _apply_span_data, _apply_span_error


@dataclass
class _SpanEntry:
    lmnr_span: Any
    agents_span: Any = None


@dataclass
class _TraceState:
    root_span: Any = None
    spans: Dict[str, _SpanEntry] = field(default_factory=dict)
    ready: threading.Event = field(default_factory=threading.Event)
    # Tracks in-flight on_span_end calls so _end_trace_state can wait
    # for all child spans to finish before ending the root span.
    pending_ends: int = 0
    pending_ends_done: threading.Event = field(default_factory=threading.Event)
    # Set to True when root span creation fails, so waiting threads
    # know the state is unusable rather than proceeding with root_span=None.
    failed: bool = False

    def __post_init__(self) -> None:
        # Initially no pending ends, so mark as done.
        self.pending_ends_done.set()


class LaminarAgentsTraceProcessor(_Base):
    """TracingProcessor implementation that mirrors OpenAI Agents spans into Laminar."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._traces: Dict[str, _TraceState] = {}
        self._disabled = False

    def on_trace_start(self, trace: Any) -> None:
        if self._disabled:
            return
        trace_id = getattr(trace, "trace_id", None)
        if not trace_id:
            return
        try:
            state = self._get_or_create_trace(trace)
            # If a span arrived first, the root span may have a placeholder
            # name. Update it to the actual trace name.
            trace_name = getattr(trace, "name", None)
            if trace_name and hasattr(state.root_span, "update_name"):
                try:
                    state.root_span.update_name(trace_name)
                except Exception:
                    pass
            self._apply_trace_metadata(state.root_span, trace)
        except Exception:
            pass

    def on_trace_end(self, trace: Any) -> None:
        if self._disabled:
            return
        trace_id = getattr(trace, "trace_id", None)
        if not trace_id:
            return
        with self._lock:
            state = self._traces.pop(trace_id, None)
        if not state:
            return
        self._end_trace_state(state)

    def on_span_start(self, span: Any) -> None:
        if self._disabled:
            return
        trace_id = getattr(span, "trace_id", None)
        if not trace_id:
            return
        try:
            state = self._get_or_create_trace(span)
            parent_id = getattr(span, "parent_id", None)
            with self._lock:
                parent_entry = state.spans.get(parent_id) if parent_id else None
            parent_lmnr_span = parent_entry.lmnr_span if parent_entry else state.root_span

            parent_ctx = None
            if parent_lmnr_span is not None and hasattr(
                parent_lmnr_span, "get_laminar_span_context"
            ):
                parent_ctx = parent_lmnr_span.get_laminar_span_context()

            span_data = getattr(span, "span_data", None)
            span_type = map_span_type(span_data)
            name = span_name(span, span_data)

            lmnr_span = Laminar.start_span(
                name,
                span_type=span_type,
                parent_span_context=parent_ctx,
                tags=["openai-agents"],
            )
            if hasattr(lmnr_span, "set_attribute"):
                lmnr_span.set_attribute("openai.agents.span.type", span_kind(span_data))
                span_id = getattr(span, "span_id", "")
                if span_id:
                    lmnr_span.set_attribute("openai.agents.span.id", span_id)

            # Use span_id as key so parent_id lookups in on_span_start
            # match correctly. The SDK always generates a span_id.
            key = getattr(span, "span_id", None) or str(id(span))
            with self._lock:
                state.spans[key] = _SpanEntry(lmnr_span=lmnr_span, agents_span=span)
        except Exception:
            pass

    def on_span_end(self, span: Any) -> None:
        if self._disabled:
            return
        trace_id = getattr(span, "trace_id", None)
        if not trace_id:
            return

        key = getattr(span, "span_id", None) or str(id(span))

        with self._lock:
            state = self._traces.get(trace_id)
            entry = state.spans.pop(key, None) if state else None
            if entry and state:
                state.pending_ends += 1
                state.pending_ends_done.clear()

        if not entry:
            return

        span_data = getattr(span, "span_data", None)
        try:
            try:
                _apply_span_data(entry.lmnr_span, span_data)
                _apply_span_error(entry.lmnr_span, span)
            except Exception:
                pass
            try:
                entry.lmnr_span.end()
            except Exception:
                pass
        finally:
            with self._lock:
                state.pending_ends -= 1
                if state.pending_ends == 0:
                    state.pending_ends_done.set()

    def shutdown(self) -> None:
        self._disabled = True
        with self._lock:
            states = list(self._traces.values())
            self._traces.clear()
        for state in states:
            self._end_trace_state(state)
        try:
            Laminar.flush()
        except Exception:
            pass

    def force_flush(self) -> bool:
        try:
            return Laminar.flush()
        except Exception:
            return False

    _SHUTDOWN_TIMEOUT = 10.0  # seconds to wait during shutdown/cleanup

    def _end_trace_state(self, state: _TraceState) -> None:
        """End all child spans (LIFO) then the root span for a trace."""
        state.ready.wait(timeout=self._SHUTDOWN_TIMEOUT)
        # Wait for in-flight on_span_end calls, then atomically snapshot
        # remaining spans.  Re-check under the lock to close the window
        # where a new on_span_end increments pending_ends between the
        # wait() return and the lock acquisition.
        for _ in range(3):  # bounded retries
            state.pending_ends_done.wait(timeout=self._SHUTDOWN_TIMEOUT)
            with self._lock:
                if state.pending_ends == 0:
                    remaining = list(state.spans.values())
                    state.spans.clear()
                    break
            # pending_ends changed while we waited; retry
        else:
            # Give up waiting — snapshot whatever is left to avoid hanging.
            with self._lock:
                remaining = list(state.spans.values())
                state.spans.clear()
        for entry in reversed(remaining):
            try:
                if entry.agents_span is not None:
                    span_data = getattr(entry.agents_span, "span_data", None)
                    _apply_span_data(entry.lmnr_span, span_data)
                    _apply_span_error(entry.lmnr_span, entry.agents_span)
            except Exception:
                pass
            try:
                entry.lmnr_span.end()
            except Exception:
                pass
        try:
            state.root_span.end()
        except Exception:
            pass

    def _get_or_create_trace(self, trace_or_span: Any) -> _TraceState:
        trace_id = getattr(trace_or_span, "trace_id", None)
        if not trace_id:
            trace_id = "unknown"
        creator = False
        with self._lock:
            state = self._traces.get(trace_id)
            if state is None:
                state = _TraceState()
                self._traces[trace_id] = state
                creator = True
        if creator:
            try:
                # Use a generic name; on_trace_start will update it
                # to the actual trace name via update_name.
                root_span = Laminar.start_span(
                    "agents.trace",
                    tags=["openai-agents"],
                )
                state.root_span = root_span
            except Exception:
                state.failed = True
                # Remove the broken state so future calls can retry.
                with self._lock:
                    self._traces.pop(trace_id, None)
                raise
            finally:
                state.ready.set()
        else:
            state.ready.wait()
            if state.failed:
                raise RuntimeError("Root span creation failed for this trace")
        return state

    def _apply_trace_metadata(self, root_span: Any, trace: Any) -> None:
        metadata: Dict[str, Any] = {}
        trace_metadata = getattr(trace, "metadata", None)
        if isinstance(trace_metadata, dict):
            metadata.update(trace_metadata)
        trace_id = getattr(trace, "trace_id", None)
        if trace_id:
            metadata["openai.agents.trace_id"] = trace_id
        group_id = getattr(trace, "group_id", None)
        if group_id:
            metadata["openai.agents.group_id"] = group_id
        trace_name = getattr(trace, "name", None)
        if trace_name:
            metadata["openai.agents.trace_name"] = trace_name
        if metadata and hasattr(root_span, "set_trace_metadata"):
            try:
                root_span.set_trace_metadata(metadata)
            except Exception:
                pass
        session_id = metadata.get("session_id") if metadata else None
        user_id = metadata.get("user_id") if metadata else None
        if not session_id:
            session_id = os.getenv("LMNR_SESSION_ID")
        if not user_id:
            user_id = os.getenv("LMNR_USER_ID")
        if session_id and hasattr(root_span, "set_trace_session_id"):
            root_span.set_trace_session_id(session_id)
        if user_id and hasattr(root_span, "set_trace_user_id"):
            root_span.set_trace_user_id(user_id)

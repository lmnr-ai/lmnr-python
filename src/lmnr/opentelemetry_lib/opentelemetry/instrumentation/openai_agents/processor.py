"""LaminarAgentsTraceProcessor - mirrors OpenAI Agents spans into Laminar."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict

from lmnr import Laminar

from .helpers import _map_span_type, _span_kind, _span_name
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


class LaminarAgentsTraceProcessor:
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
        state = self._get_or_create_trace(trace)
        self._apply_trace_metadata(state.root_span, trace)

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
        state = self._get_or_create_trace(span)
        parent_id = getattr(span, "parent_id", None)
        with self._lock:
            parent_entry = state.spans.get(parent_id) if parent_id else None
        parent_lmnr_span = parent_entry.lmnr_span if parent_entry else state.root_span

        parent_ctx = None
        if parent_lmnr_span is not None and hasattr(parent_lmnr_span, "get_laminar_span_context"):
            parent_ctx = parent_lmnr_span.get_laminar_span_context()

        span_data = getattr(span, "span_data", None)
        span_type = _map_span_type(span_data)
        name = _span_name(span, span_data)

        lmnr_span = Laminar.start_span(
            name,
            span_type=span_type,
            parent_span_context=parent_ctx,
            tags=["openai-agents"],
        )
        if hasattr(lmnr_span, "set_attribute"):
            lmnr_span.set_attribute("openai.agents.span.type", _span_kind(span_data))
            span_id = getattr(span, "span_id", "")
            if span_id:
                lmnr_span.set_attribute("openai.agents.span.id", span_id)

        # Use span_id as key, fall back to span_name
        key = getattr(span, "span_id", None) or name
        with self._lock:
            state.spans[key] = _SpanEntry(lmnr_span=lmnr_span, agents_span=span)

    def on_span_end(self, span: Any) -> None:
        if self._disabled:
            return
        trace_id = getattr(span, "trace_id", None)
        if not trace_id:
            return

        # Use consistent key lookup: try span_id first, then span_name
        span_id = getattr(span, "span_id", None)
        span_data = getattr(span, "span_data", None)
        span_name_key = _span_name(span, span_data)
        key = span_id or span_name_key

        with self._lock:
            state = self._traces.get(trace_id)
            entry = state.spans.pop(key, None) if state else None

        if not entry:
            return

        _apply_span_data(entry.lmnr_span, span_data)
        _apply_span_error(entry.lmnr_span, span)
        try:
            entry.lmnr_span.end()
        except Exception:
            pass

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

    def _end_trace_state(self, state: _TraceState) -> None:
        """End all child spans (LIFO) then the root span for a trace."""
        state.ready.wait()
        for entry in reversed(list(state.spans.values())):
            try:
                if entry.agents_span is not None:
                    span_data = getattr(entry.agents_span, "span_data", None)
                    _apply_span_data(entry.lmnr_span, span_data)
                    _apply_span_error(entry.lmnr_span, entry.agents_span)
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
                name = getattr(trace_or_span, "name", None) or "agents.trace"
                root_span = Laminar.start_span(
                    name,
                    tags=["openai-agents"],
                )
                state.root_span = root_span
            finally:
                state.ready.set()
        else:
            state.ready.wait()
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

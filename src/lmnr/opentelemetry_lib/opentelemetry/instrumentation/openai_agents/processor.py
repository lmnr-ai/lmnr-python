"""LaminarAgentsTraceProcessor - mirrors OpenAI Agents spans into Laminar."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opentelemetry.context import get_value, set_value

try:
    from agents.tracing import TracingProcessor as _Base
except ImportError:  # openai-agents not installed
    _Base = object

if TYPE_CHECKING:
    from agents.tracing import Span as AgentsSpan
    from agents.tracing import Trace

    from lmnr.opentelemetry_lib.tracing.span import LaminarSpan
    from lmnr.sdk.types import LaminarSpanContext

from lmnr import Laminar
from lmnr.opentelemetry_lib.tracing.context import get_current_context

from .helpers import (
    DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY,
    name_from_span_data,
    export_span_data,
    map_span_type,
    span_kind,
    span_name,
)
from .span_data import apply_span_data, apply_span_error

logger = logging.getLogger(__name__)


@dataclass
class _SpanEntry:
    lmnr_span: LaminarSpan
    agents_span: AgentsSpan[Any] | None = None


@dataclass
class _TraceState:
    root_span: LaminarSpan | None = None
    spans: dict[str, _SpanEntry] = field(default_factory=dict)
    ready: threading.Event = field(default_factory=threading.Event)
    # Tracks in-flight on_span_end calls so _end_trace_state can wait
    # for all child spans to finish before ending the root span.
    pending_ends: int = 0
    pending_ends_done: threading.Event = field(default_factory=threading.Event)
    # Set to True when root span creation fails, so waiting threads
    # know the state is unusable rather than proceeding with root_span=None.
    failed: bool = False
    # Guards against double-ending from concurrent on_trace_end and shutdown.
    ended: bool = False
    # Maps destination agent name -> handoff lmnr span context so that the
    # subsequent agent span becomes a child of the handoff span.
    pending_handoff_ctxs: dict[str, LaminarSpanContext] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Initially no pending ends, so mark as done.
        self.pending_ends_done.set()


class LaminarAgentsTraceProcessor(_Base):
    """TracingProcessor implementation that mirrors OpenAI Agents spans into Laminar."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._traces: dict[str, _TraceState] = {}
        self._disabled = False

    def on_trace_start(self, trace: Trace) -> None:
        if self._disabled:
            return
        trace_id = trace.trace_id
        if not trace_id:
            return
        try:
            state = self._get_or_create_trace(trace)
            # If a span arrived first, the root span may have a placeholder
            # name. Update it to the actual trace name.
            trace_name = trace.name
            if trace_name and state.root_span is not None:
                try:
                    state.root_span.update_name(trace_name)
                except Exception:
                    pass
            self._apply_trace_metadata(state.root_span, trace)
        except Exception:
            logger.debug("Error in on_trace_start", exc_info=True)

    def on_trace_end(self, trace: Trace) -> None:
        if self._disabled:
            return
        trace_id = trace.trace_id
        if not trace_id:
            return
        with self._lock:
            state = self._traces.get(trace_id)
            if not state or state.ended:
                return
            state.ended = True
        self._end_trace_state(state)
        # Remove after cleanup so concurrent on_span_end calls can still
        # find the state and finish their spans.
        with self._lock:
            self._traces.pop(trace_id, None)

    def on_span_start(self, span: AgentsSpan[Any]) -> None:
        if self._disabled:
            return
        trace_id = span.trace_id
        if not trace_id:
            return
        lmnr_span = None
        try:
            state = self._get_or_create_trace(span)

            parent_ctx: LaminarSpanContext | None = None

            span_data = span.span_data
            span_type = map_span_type(span_data)
            name = span_name(span, span_data)

            # If this is an agent span, check if a handoff targeting this agent
            # is pending. If so, make this span a child of the handoff span so
            # the subagent is nested under the handoff that triggered it.
            if span_kind(span_data) == "agent":
                this_agent = name_from_span_data(
                    export_span_data(span_data).get("name")
                    or getattr(span_data, "name", None)
                )
                with self._lock:
                    handoff_ctx = state.pending_handoff_ctxs.pop(this_agent, None)
                if handoff_ctx is not None:
                    parent_ctx = handoff_ctx

            otel_ctx = get_current_context()
            ctx = set_value(
                DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY, True, otel_ctx
            )

            lmnr_span = Laminar.start_active_span(
                name=name,
                span_type=span_type,
                parent_span_context=parent_ctx,
                context=ctx,
            )
            # Use span_id as key so parent_id lookups in on_span_start
            # match correctly. The SDK always generates a span_id.
            key = span.span_id
            if not key:
                logger.debug("Span missing span_id, cannot track")
                try:
                    lmnr_span.end()
                except Exception:
                    pass
                return
            with self._lock:
                state.spans[key] = _SpanEntry(lmnr_span=lmnr_span, agents_span=span)
        except Exception:
            logger.debug("Error in on_span_start", exc_info=True)
            if lmnr_span is not None:
                try:
                    lmnr_span.end()
                except Exception:
                    pass

    def on_span_end(self, span: AgentsSpan[Any]) -> None:
        if self._disabled:
            return
        trace_id = span.trace_id
        if not trace_id:
            return

        key = span.span_id
        if not key:
            return

        with self._lock:
            state = self._traces.get(trace_id)
            entry = state.spans.pop(key, None) if state else None
            if entry and state:
                state.pending_ends += 1
                state.pending_ends_done.clear()

        if not entry or not state:
            return

        span_data = span.span_data
        try:
            try:
                apply_span_data(entry.lmnr_span, span_data)
                apply_span_error(entry.lmnr_span, span)
            except Exception:
                pass

            # When a handoff span ends, save the *parent* span's context keyed
            # by the destination agent name. on_span_start consumes this so
            # the subsequent agent span becomes a sibling of the handoff span
            # (both children of the handoff's parent).
            if span_kind(span_data) == "handoff":
                try:
                    to_agent = name_from_span_data(
                        export_span_data(span_data).get("to_agent")
                        or getattr(span_data, "to_agent", None)
                    )
                    if to_agent:
                        parent_id = getattr(span, "parent_id", None)
                        with self._lock:
                            parent_entry = (
                                state.spans.get(parent_id) if parent_id else None
                            )
                        parent_lmnr_span = (
                            parent_entry.lmnr_span
                            if parent_entry is not None
                            else state.root_span
                        )
                        if parent_lmnr_span is not None:
                            handoff_ctx = parent_lmnr_span.get_laminar_span_context()
                            with self._lock:
                                state.pending_handoff_ctxs[to_agent] = handoff_ctx
                except Exception:
                    pass

            try:
                entry.lmnr_span.end()
            except Exception:
                pass
        finally:
            with self._lock:
                if state.pending_ends > 0:
                    state.pending_ends -= 1
                if state.pending_ends == 0:
                    state.pending_ends_done.set()

    def shutdown(self) -> None:
        self._disabled = True
        with self._lock:
            states = [s for s in self._traces.values() if not s.ended]
            for s in states:
                s.ended = True
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
        if not state.ready.wait(timeout=self._SHUTDOWN_TIMEOUT) or state.failed:
            return
        # Wait for in-flight on_span_end calls, then atomically snapshot
        # remaining spans. Re-check under the lock to close the window
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
                    apply_span_data(entry.lmnr_span, span_data)
                    apply_span_error(entry.lmnr_span, entry.agents_span)
            except Exception:
                pass
            try:
                entry.lmnr_span.end()
            except Exception:
                pass
        try:
            if state.root_span:
                state.root_span.end()
        except Exception:
            pass

    def _get_or_create_trace(
        self, trace_or_span: Trace | AgentsSpan[Any]
    ) -> _TraceState:
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
                root_span = Laminar.start_active_span(
                    "agents.trace",
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
            if not state.ready.wait(timeout=self._SHUTDOWN_TIMEOUT) or state.failed:
                raise RuntimeError("Root span creation failed for this trace")
        return state

    def _apply_trace_metadata(
        self, root_span: LaminarSpan | None, trace: Trace
    ) -> None:
        if root_span is None:
            return
        metadata: dict[str, Any] = {}
        trace_metadata = getattr(trace, "metadata", None)
        if isinstance(trace_metadata, dict):
            metadata.update(trace_metadata)
        group_id = getattr(trace, "group_id", None)
        if group_id:
            metadata["openai.agents.group_id"] = group_id
        if trace.name:
            metadata["openai.agents.trace_name"] = trace.name
        if metadata:
            try:
                root_span.set_trace_metadata(metadata)
            except Exception:
                pass
        session_id = metadata.get("session_id")
        user_id = metadata.get("user_id")
        if session_id:
            root_span.set_trace_session_id(session_id)
        if user_id:
            root_span.set_trace_user_id(user_id)

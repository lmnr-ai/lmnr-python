"""LaminarAgentsTraceProcessor - mirrors OpenAI Agents spans into Laminar."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opentelemetry.context import set_value

try:
    from agents.tracing import TracingProcessor as _Base
except ImportError:  # openai-agents not installed
    _Base = object

if TYPE_CHECKING:
    from agents.tracing import Span as AgentsSpan
    from agents.tracing import Trace

    from lmnr.opentelemetry_lib.tracing.span import LaminarSpan

from lmnr import Laminar
from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.opentelemetry_lib.tracing.attributes import SPAN_IDS_PATH, SPAN_PATH
from lmnr.opentelemetry_lib.tracing.context import get_current_context
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor

from .helpers import (
    DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY,
    map_span_type,
    span_name,
)
from .span_data import apply_span_data, apply_span_error

logger = logging.getLogger(__name__)

# Fallback name for the root span, used only when the trace name cannot be
# resolved. `lmnr.span.path` is captured at span start, so a placeholder here
# would be baked into the path of every span in the trace.
_ROOT_SPAN_FALLBACK_NAME = "agents.trace"


def _root_span_name(trace_or_span: Trace | AgentsSpan[Any]) -> str:
    """Resolve the trace name up front, whether a trace or a span arrived first."""
    # Only Trace carries `name`; Span does not.
    name = getattr(trace_or_span, "name", None)
    if not name:
        try:
            from agents.tracing import get_current_trace

            current_trace = get_current_trace()
            name = getattr(current_trace, "name", None) if current_trace else None
        except Exception:
            name = None
    return name or _ROOT_SPAN_FALLBACK_NAME


def _rename_span(lmnr_span: LaminarSpan, name: str) -> None:
    """Rename a span and repair the dotted path captured at start."""
    lmnr_span.update_name(name)
    attributes = lmnr_span.attributes or {}
    span_path = list(attributes.get(SPAN_PATH, ()))
    if not span_path:
        return
    span_path[-1] = name
    lmnr_span.set_attribute(SPAN_PATH, span_path)
    # Children resolve their parent path from the processor's cache, not from
    # the attribute, so both have to move.
    processor = TracerWrapper.instance._span_processor
    if isinstance(processor, LaminarSpanProcessor):
        processor.set_parent_path_info(
            lmnr_span.context.span_id,
            span_path,
            list(attributes.get(SPAN_IDS_PATH, ())),
        )


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
            if (
                trace_name
                and state.root_span is not None
                and state.root_span.name != trace_name
            ):
                try:
                    _rename_span(state.root_span, trace_name)
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

            span_data = span.span_data
            span_type = map_span_type(span_data)
            name = span_name(span, span_data)

            otel_ctx = get_current_context()
            ctx = set_value(
                DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY, True, otel_ctx
            )

            # The SDK's own nesting is authoritative. In particular a handoff's
            # destination agent must NOT be reparented onto the handoff or its
            # parent: the SDK finishes the source agent's span before starting
            # the destination's, so those are already closed and the child would
            # start after its parent ended. The handoff stays discoverable via
            # `openai.agents.handoff.{from,to}`.
            lmnr_span = Laminar.start_active_span(
                name=name,
                span_type=span_type,
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
                root_span = Laminar.start_active_span(
                    _root_span_name(trace_or_span),
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

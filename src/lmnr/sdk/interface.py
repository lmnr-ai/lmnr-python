from .context import LaminarSingleton
from .tracing_types import EvaluateEvent, Span, Trace, Event

from typing import Any, Literal, Optional, Union
import datetime
import logging
import uuid


laminar = LaminarSingleton().get()


class ObservationContext:
    observation: Union[Span, Trace] = None
    _parent: "ObservationContext" = None
    _children: dict[uuid.UUID, "ObservationContext"] = {}
    _log = logging.getLogger("laminar.observation_context")

    def __init__(self, observation: Union[Span, Trace], parent: "ObservationContext"):
        self.observation = observation
        self._parent = parent
        self._children = {}

    def _get_parent(self) -> "ObservationContext":
        raise NotImplementedError

    def end(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def span(
        self,
        name: str,
        input: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
        span_type: Literal["DEFAULT", "LLM"] = "DEFAULT",
    ) -> "SpanContext":
        parent = self._get_parent()
        parent_span_id = (
            parent.observation.id if isinstance(parent.observation, Span) else None
        )
        trace_id = (
            parent.observation.traceId
            if isinstance(parent.observation, Span)
            else parent.observation.id
        )
        span = laminar.create_span(
            name=name,
            trace_id=trace_id,
            input=input,
            metadata=metadata,
            attributes=attributes,
            parent_span_id=parent_span_id,
            span_type=span_type,
        )
        span_context = SpanContext(span, self)
        self._children[span.id] = span_context
        return span_context

    def id(self) -> uuid.UUID:
        return self.observation.id


class SpanContext(ObservationContext):
    def _get_parent(self) -> ObservationContext:
        return self._parent

    def end(
        self,
        output: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        check_event_names: Optional[list[str]] = None,
        override: bool = False,
    ) -> "SpanContext":
        if self._children:
            self._log.warning(
                "Ending span %s, but it has children that have not been finalized. Children: %s",
                self.observation.name,
                [child.observation.name for child in self._children.values()],
            )
        self._get_parent()._children.pop(self.observation.id)
        return self._update(
            output=output,
            metadata=metadata,
            evaluate_events=check_event_names,
            override=override,
            finalize=True,
        )

    def update(
        self,
        output: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        check_event_names: Optional[list[str]] = None,
        override: bool = False,
    ) -> "SpanContext":
        return self._update(
            output=output or self.observation.output,
            metadata=metadata or self.observation.metadata,
            evaluate_events=check_event_names or self.observation.evaluateEvents,
            override=override,
            finalize=False,
        )

    def event(
        self,
        name: str,
        value: Optional[Union[str, int, float]] = None,
        timestamp: Optional[datetime.datetime] = None,
    ) -> "SpanContext":
        event = Event(
            name=name,
            span_id=self.observation.id,
            timestamp=timestamp,
            value=value,
        )
        self.observation.add_event(event)
        return self

    def evaluate_event(self, name: str, data: str) -> "SpanContext":
        existing_evaluate_events = self.observation.evaluateEvents
        output = self.observation.output
        self._update(
            output=output,
            evaluate_events=existing_evaluate_events
            + [EvaluateEvent(name=name, data=data)],
            override=False,
        )

    def _update(
        self,
        output: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        evaluate_events: Optional[list[EvaluateEvent]] = None,
        override: bool = False,
        finalize: bool = False,
    ) -> "SpanContext":
        new_metadata = (
            metadata
            if override
            else {**(self.observation.metadata or {}), **(metadata or {})}
        )
        new_evaluate_events = (
            evaluate_events
            if override
            else self.observation.evaluateEvents + (evaluate_events or [])
        )
        self.observation = laminar.update_span(
            span=self.observation,
            end_time=datetime.datetime.now(datetime.timezone.utc),
            output=output,
            metadata=new_metadata,
            evaluate_events=new_evaluate_events,
            finalize=finalize,
        )
        return self


class TraceContext(ObservationContext):
    def _get_parent(self) -> "ObservationContext":
        return self

    def update(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        success: bool = True,
    ) -> "TraceContext":
        return self._update(
            user_id=user_id or self.observation.userId,
            session_id=session_id or self.observation.sessionId,
            release=release or self.observation.release,
            metadata=metadata or self.observation.metadata,
            success=success if success is not None else self.observation.success,
        )

    def end(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        success: bool = True,
    ) -> "TraceContext":
        if self._children:
            self._log.warning(
                "Ending trace id: %s, but it has children that have not been finalized. Children: %s",
                self.observation.id,
                [child.observation.name for child in self._children.values()],
            )
        return self._update(
            user_id=user_id or self.observation.userId,
            session_id=session_id or self.observation.sessionId,
            release=release or self.observation.release,
            metadata=metadata or self.observation.metadata,
            success=success if success is not None else self.observation.success,
            end_time=datetime.datetime.now(datetime.timezone.utc),
        )

    def _update(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        success: bool = True,
        end_time: Optional[datetime.datetime] = None,
    ) -> "TraceContext":
        self.observation = laminar.update_trace(
            id=self.observation.id,
            user_id=user_id,
            start_time=self.observation.startTime,
            session_id=session_id,
            release=release,
            metadata=metadata,
            success=success,
            end_time=end_time,
        )
        return self


def trace(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    release: Optional[str] = None,
) -> TraceContext:
    session_id = session_id or str(uuid.uuid4())
    trace_id = uuid.uuid4()
    trace = laminar.update_trace(
        id=trace_id,
        user_id=user_id,
        session_id=session_id,
        release=release,
        start_time=datetime.datetime.now(datetime.timezone.utc),
    )
    return TraceContext(trace, None)

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
        """Create a span within the current (trace or span) context.

        Args:
            name (str): Span name
            input (Optional[Any], optional): Inputs to the span. Defaults to None.
            metadata (Optional[dict[str, Any]], optional): Any additional metadata. Defaults to None.
            attributes (Optional[dict[str, Any]], optional): Any pre-defined attributes. Must comply to the convention. Defaults to None.
            span_type (Literal[&quot;DEFAULT&quot;, &quot;LLM&quot;], optional): Type of the span. Defaults to "DEFAULT".

        Returns:
            SpanContext: The new span context
        """
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
        """Get the uuid of the current observation

        Returns:
            uuid.UUID: UUID of the observation
        """
        return self.observation.id


class SpanContext(ObservationContext):
    def _get_parent(self) -> ObservationContext:
        return self._parent

    def end(
        self,
        output: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        evaluate_events: Optional[list[EvaluateEvent]] = None,
        override: bool = False,
    ) -> "SpanContext":
        """End the span with the given output and optional metadata and evaluate events.

        Args:
            output (Optional[Any], optional): output of the span. Defaults to None.
            metadata (Optional[dict[str, Any]], optional): any additional metadata to the span. Defaults to None.
            check_event_names (Optional[list[EvaluateEvent]], optional): List of events to evaluate for and tag. Defaults to None.
            override (bool, optional): override existing metadata fully. If False, metadata is merged. Defaults to False.

        Returns:
            SpanContext: the finished span context
        """
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
            evaluate_events=evaluate_events,
            override=override,
            finalize=True,
        )

    def update(
        self,
        output: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        evaluate_events: Optional[list[EvaluateEvent]] = None,
        override: bool = False,
    ) -> "SpanContext":
        """Update the current span with (optionally) the given output and optional metadata and evaluate events, but don't end it.

        Args:
            output (Optional[Any], optional): output of the span. Defaults to None.
            metadata (Optional[dict[str, Any]], optional): any additional metadata to the span. Defaults to None.
            check_event_names (Optional[list[EvaluateEvent]], optional): List of events to evaluate for and tag. Defaults to None.
            override (bool, optional): override existing metadata fully. If False, metadata is merged. Defaults to False.

        Returns:
            SpanContext: the finished span context
        """
        return self._update(
            output=output or self.observation.output,
            metadata=metadata or self.observation.metadata,
            evaluate_events=evaluate_events or self.observation.evaluateEvents,
            override=override,
            finalize=False,
        )

    def event(
        self,
        name: str,
        value: Optional[Union[str, int]] = None,
        timestamp: Optional[datetime.datetime] = None,
    ) -> "SpanContext":
        """Associate an event with the current span

        Args:
            name (str): name of the event. Must be predefined in the Laminar events page.
            value (Optional[Union[str, int]], optional): value of the event. Must match range definition in Laminar events page. Defaults to None.
            timestamp (Optional[datetime.datetime], optional): If you need custom timestamp. If not specified, current time is used. Defaults to None.

        Returns:
            SpanContext: the updated span context
        """
        event = Event(
            name=name,
            span_id=self.observation.id,
            timestamp=timestamp,
            value=value,
        )
        self.observation.add_event(event)
        return self

    def evaluate_event(self, name: str, data: str) -> "SpanContext":
        """Evaluate an event with the given name and data. The event value will be assessed by the Laminar evaluation engine.
        Data is passed as an input to the agent, so you need to specify which data you want to evaluate. Most of the times,
        this is an output of the LLM generation, but sometimes, you may want to evaluate the input or both. In the latter case,
        concatenate the input and output annotating with natural language.

        Args:
            name (str): Name of the event. Must be predefined in the Laminar events page.
            data (str): Data to be evaluated. Typically the output of the LLM generation.

        Returns:
            SpanContext: the updated span context
        """
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
        """Update the current trace with the given metadata and success status.

        Args:
            user_id (Optional[str], optional): Custom user_id of your user. Useful for grouping and further analytics. Defaults to None.
            session_id (Optional[str], optional): Custom session_id for your session. Random UUID is generated on Laminar side, if not specified.
                                                  Defaults to None.
            release (Optional[str], optional): _description_. Release of your application. Useful for grouping and further analytics. Defaults to None.
            metadata (Optional[dict[str, Any]], optional):  any additional metadata to the trace. Defaults to None.
            success (bool, optional): whether this trace ran successfully. Defaults to True.

        Returns:
            TraceContext: updated trace context
        """
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
        """End the current trace with the given metadata and success status.

        Args:
            user_id (Optional[str], optional): Custom user_id of your user. Useful for grouping and further analytics. Defaults to None.
            session_id (Optional[str], optional): Custom session_id for your session. Random UUID is generated on Laminar side, if not specified.
                                                  Defaults to None.
            release (Optional[str], optional): _description_. Release of your application. Useful for grouping and further analytics. Defaults to None.
            metadata (Optional[dict[str, Any]], optional):  any additional metadata to the trace. Defaults to None.
            success (bool, optional): whether this trace ran successfully. Defaults to True.

        Returns:
            TraceContext: context of the ended trace
        """
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
    """Create the initial trace context. All further spans will be created within this context.

    Args:
        user_id (Optional[str], optional): Custom user_id of your user. Useful for grouping and further analytics. Defaults to None.
            session_id (Optional[str], optional): Custom session_id for your session. Random UUID is generated on Laminar side, if not specified.
                                                  Defaults to None.
            release (Optional[str], optional): _description_. Release of your application. Useful for grouping and further analytics. Defaults to None.

    Returns:
        TraceContext: the pointer to the trace context. Use `.span()` to create a new span within this context.
    """
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

from typing import Any, Literal, Optional, Union
import datetime
import pydantic
import uuid

from .constants import CURRENT_TRACING_VERSION
from .utils import to_dict


class EvaluateEvent(pydantic.BaseModel):
    name: str
    data: str


class Span(pydantic.BaseModel):
    version: str = CURRENT_TRACING_VERSION
    spanType: Literal["DEFAULT", "LLM"] = "DEFAULT"
    id: uuid.UUID
    traceId: uuid.UUID
    parentSpanId: Optional[uuid.UUID] = None
    name: str
    # generated at start of span, so required
    startTime: datetime.datetime
    # generated at end of span, optional when span is still running
    endTime: Optional[datetime.datetime] = None
    attributes: dict[str, Any] = {}
    input: Optional[Any] = None
    output: Optional[Any] = None
    metadata: Optional[dict[str, Any]] = None
    evaluateEvents: list[EvaluateEvent] = []
    events: list["Event"] = None

    def __init__(
        self,
        name: str,
        trace_id: uuid.UUID,
        start_time: Optional[datetime.datetime] = None,
        version: str = CURRENT_TRACING_VERSION,
        span_type: Literal["DEFAULT", "LLM"] = "DEFAULT",
        id: Optional[uuid.UUID] = None,
        parent_span_id: Optional[uuid.UUID] = None,
        input: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = {},
        attributes: Optional[dict[str, Any]] = {},
        evaluate_events: list[EvaluateEvent] = [],
    ):
        super().__init__(
            version=version,
            spanType=span_type,
            id=id or uuid.uuid4(),
            traceId=trace_id,
            parentSpanId=parent_span_id,
            name=name,
            startTime=start_time or datetime.datetime.now(datetime.timezone.utc),
            input=input,
            metadata=metadata or {},
            attributes=attributes or {},
            evaluateEvents=evaluate_events,
            events=[],
        )

    def update(
        self,
        end_time: Optional[datetime.datetime],
        output: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
        evaluate_events: Optional[list[EvaluateEvent]] = None,
        override: bool = False,
    ):
        self.endTime = end_time or datetime.datetime.now(datetime.timezone.utc)
        self.output = output
        new_metadata = (
            metadata if override else {**(self.metadata or {}), **(metadata or {})}
        )
        new_attributes = (
            attributes or {}
            if override
            else {**(self.attributes or {}), **(attributes or {})}
        )
        new_evaluate_events = (
            evaluate_events or []
            if override
            else self.evaluateEvents + (evaluate_events or [])
        )
        self.metadata = new_metadata
        self.attributes = new_attributes
        self.evaluateEvents = new_evaluate_events

    def add_event(self, event: "Event"):
        self.events.append(event)

    def to_dict(self) -> dict[str, Any]:
        try:
            obj = self.model_dump()
        except TypeError:
            # if inner values are pydantic models, we need to call model_dump on them
            # see: https://github.com/pydantic/pydantic/issues/7713
            obj = {}
            for key, value in self.__dict__.items():
                obj[key] = (
                    value.model_dump()
                    if isinstance(value, pydantic.BaseModel)
                    else value
                )
        obj = to_dict(obj)
        return obj


class Trace(pydantic.BaseModel):
    id: uuid.UUID
    version: str = CURRENT_TRACING_VERSION
    success: bool = True
    startTime: Optional[datetime.datetime] = None
    endTime: Optional[datetime.datetime] = None
    userId: Optional[str] = None  # provided by user or null
    sessionId: Optional[str] = None  # provided by user or uuid()
    release: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def __init__(
        self,
        success: bool = True,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        id: Optional[uuid.UUID] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        id_ = id or uuid.uuid4()
        super().__init__(
            id=id_,
            startTime=start_time,
            success=success,
            endTime=end_time,
            userId=user_id,
            sessionId=session_id,
            release=release,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        try:
            obj = self.model_dump()
        except TypeError:
            # if inner values are pydantic models, we need to call model_dump on them
            # see: https://github.com/pydantic/pydantic/issues/7713
            obj = {}
            for key, value in self.__dict__.items():
                obj[key] = (
                    value.model_dump()
                    if isinstance(value, pydantic.BaseModel)
                    else value
                )
        obj = to_dict(obj)
        return obj


class Event(pydantic.BaseModel):
    id: uuid.UUID
    templateName: str
    timestamp: datetime.datetime
    spanId: uuid.UUID
    value: Optional[Union[int, str]] = None

    def __init__(
        self,
        name: str,
        span_id: uuid.UUID,
        timestamp: Optional[datetime.datetime] = None,
        value: Optional[Union[int, str]] = None,
    ):
        super().__init__(
            id=uuid.uuid4(),
            templateName=name,
            spanId=span_id,
            timestamp=timestamp or datetime.datetime.now(datetime.timezone.utc),
            value=value,
        )

    def to_dict(self) -> dict[str, Any]:
        try:
            obj = self.model_dump()
        except TypeError:
            # if inner values are pydantic models, we need to call model_dump on them
            # see: https://github.com/pydantic/pydantic/issues/7713
            obj = {}
            for key, value in self.__dict__.items():
                obj[key] = (
                    value.model_dump()
                    if isinstance(value, pydantic.BaseModel)
                    else value
                )
        obj = to_dict(obj)
        return obj

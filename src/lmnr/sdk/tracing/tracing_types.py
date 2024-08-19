from typing import Any, Literal, Optional
import datetime
import pydantic
import uuid

from .constants import LATEST_VERSION
from .utils import to_dict


class Span(pydantic.BaseModel):
    version: str = LATEST_VERSION
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
    checkEventNames: list[str] = None
    events: list["Event"] = None

    def __init__(
        self,
        name: str,
        trace_id: uuid.UUID,
        start_time: Optional[datetime.datetime] = None,
        version: str = LATEST_VERSION,
        span_type: Literal["DEFAULT", "LLM"] = "DEFAULT",
        id: Optional[uuid.UUID] = None,
        parent_span_id: Optional[uuid.UUID] = None,
        input: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = {},
        attributes: Optional[dict[str, Any]] = {},
        check_event_names: list[str] = [],
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
            checkEventNames=check_event_names,
            events=[],
        )

    def update(
        self,
        end_time: Optional[datetime.datetime],
        output: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
        check_event_names: Optional[list[str]] = None,
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
        new_check_event_names = (
            check_event_names or {}
            if override
            else self.checkEventNames + (check_event_names or [])
        )
        self.metadata = new_metadata
        self.attributes = new_attributes
        self.checkEventNames = new_check_event_names

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
    version: str = LATEST_VERSION
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
    typeName: str
    timestamp: datetime.datetime
    spanId: uuid.UUID

    def __init__(
        self,
        name: str,
        span_id: uuid.UUID,
        timestamp: Optional[datetime.datetime] = None,
    ):
        super().__init__(
            id=uuid.uuid4(),
            typeName=name,
            spanId=span_id,
            timestamp=timestamp or datetime.datetime.now(datetime.timezone.utc),
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

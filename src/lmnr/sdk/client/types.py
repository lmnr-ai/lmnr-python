from typing import Any, Literal, TypedDict
import uuid
import datetime


class SpanStartData(TypedDict):
    name: str
    span_id: uuid.UUID
    parent_span_id: uuid.UUID | None
    trace_id: uuid.UUID
    start_time: datetime.datetime
    attributes: dict[str, Any]
    span_type: Literal[
        "DEFAULT", "LLM", "TOOL", "EXECUTOR", "EVALUATOR", "EVALUATION", "CACHED"
    ]

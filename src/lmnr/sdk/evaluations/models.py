import uuid
from typing import TypedDict

from lmnr.sdk.types import Numeric

DEFAULT_BATCH_SIZE = 5
MAX_EXPORT_BATCH_SIZE = 64

class EvaluationRunResult(TypedDict):
    average_scores: dict[str, Numeric]
    evaluation_id: uuid.UUID
    project_id: uuid.UUID
    url: str
    error_message: str | None

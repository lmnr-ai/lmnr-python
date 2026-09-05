import datetime
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from pydantic import BaseModel, Field

from lmnr.sdk.types import (
    EvaluationDatapointData,
    EvaluationDatapointMetadata,
    EvaluationDatapointTarget,
    Numeric,
    parse_iso_datetime,
)
from lmnr.sdk.utils import json_dumps, serialize

DEFAULT_BATCH_SIZE = 5
MAX_EXPORT_BATCH_SIZE = 64
DEFAULT_DATAPOINT_MAX_DATA_LENGTH = 16_000_000  # 16MB


class EvaluationRunResult(TypedDict):
    average_scores: dict[str, Numeric]
    evaluation_id: uuid.UUID
    project_id: uuid.UUID
    url: str
    error_message: str | None


ExecutorFunctionReturnType = Any  # pyright: ignore[reportExplicitAny]
EvaluatorFunctionReturnType = Numeric | dict[str, Numeric]

ExecutorFunction = Callable[
    [EvaluationDatapointData, Any],  # pyright: ignore[reportExplicitAny]
    ExecutorFunctionReturnType | Awaitable[ExecutorFunctionReturnType],  # pyright: ignore[reportExplicitAny]
]

# EvaluatorFunction is a function that takes the output of the executor and the
# target data, and returns a score. The score can be a single number or a
# record of string keys and number values. The latter is useful for evaluating
# multiple criteria in one go instead of running multiple evaluators.
EvaluatorFunction = Callable[
    [ExecutorFunctionReturnType, Any],  # pyright: ignore[reportExplicitAny]
    EvaluatorFunctionReturnType | Awaitable[EvaluatorFunctionReturnType],
]


class HumanEvaluatorOptionsEntry(TypedDict):
    label: str
    value: float


class HumanEvaluator(TypedDict, total=False):
    options: list[HumanEvaluatorOptionsEntry]


class InitEvaluationResponse(TypedDict):
    id: uuid.UUID
    createdAt: datetime.datetime
    groupId: str
    name: str
    projectId: uuid.UUID


# values are not necessarily str, but the ones we parse are
def parse_init_evaluation_response(data: dict[str, str]) -> InitEvaluationResponse:
    """Parse an `InitEvaluationResponse` from a `POST /v1/evals` (or update)
    response. Field names already match the wire shape (no aliasing)."""
    return InitEvaluationResponse(
        id=uuid.UUID(str(data["id"])),
        createdAt=parse_iso_datetime(data["createdAt"]),
        groupId=data["groupId"],
        name=data["name"],
        projectId=uuid.UUID(str(data["projectId"])),
    )


class EvaluationDatapointDatasetLink(TypedDict):
    dataset_id: uuid.UUID
    datapoint_id: uuid.UUID
    created_at: datetime.datetime


def _dataset_link_to_dict(link: EvaluationDatapointDatasetLink) -> dict[str, str]:
    return {
        "datasetId": str(link["dataset_id"]),
        "datapointId": str(link["datapoint_id"]),
        "createdAt": link["created_at"].isoformat(),
    }


class PartialEvaluationDatapoint(BaseModel):
    id: uuid.UUID
    data: EvaluationDatapointData
    target: EvaluationDatapointTarget
    index: int
    trace_id: uuid.UUID
    executor_span_id: uuid.UUID
    metadata: EvaluationDatapointMetadata = Field(default=None)
    dataset_link: EvaluationDatapointDatasetLink | None = Field(default=None)

    # uuid is not serializable by default, so we need to convert it to a string
    def to_dict(self, max_data_length: int = DEFAULT_DATAPOINT_MAX_DATA_LENGTH):
        serialized_data = serialize(self.data)
        serialized_target = serialize(self.target)
        str_data = json_dumps(serialized_data)
        str_target = json_dumps(serialized_target)
        try:
            return {
                "id": str(self.id),
                "data": (
                    str_data[:max_data_length]
                    if len(str_data) > max_data_length
                    else serialized_data
                ),
                "target": (
                    str_target[:max_data_length]
                    if len(str_target) > max_data_length
                    else serialized_target
                ),
                "index": self.index,
                "traceId": str(self.trace_id),
                "executorSpanId": str(self.executor_span_id),
                "metadata": (
                    serialize(self.metadata) if self.metadata is not None else {}
                ),
                "datasetLink": (
                    _dataset_link_to_dict(self.dataset_link)
                    if self.dataset_link is not None
                    else None
                ),
            }
        except Exception as e:
            raise ValueError(f"Error serializing PartialEvaluationDatapoint: {e}")


class EvaluationResultDatapoint(BaseModel):
    id: uuid.UUID
    index: int
    data: EvaluationDatapointData
    target: EvaluationDatapointTarget
    executor_output: ExecutorFunctionReturnType
    scores: dict[str, Numeric | None]
    trace_id: uuid.UUID
    executor_span_id: uuid.UUID
    metadata: EvaluationDatapointMetadata = Field(default=None)
    dataset_link: EvaluationDatapointDatasetLink | None = Field(default=None)

    # uuid is not serializable by default, so we need to convert it to a string
    def to_dict(self, max_data_length: int = DEFAULT_DATAPOINT_MAX_DATA_LENGTH):
        try:
            serialized_data = serialize(self.data)
            serialized_target = serialize(self.target)
            serialized_executor_output = serialize(self.executor_output)
            str_data = json.dumps(serialized_data)
            str_target = json.dumps(serialized_target)
            str_executor_output = json.dumps(serialized_executor_output)
            return {
                # preserve only preview of the data, target and executor output
                # (full data is in trace)
                "id": str(self.id),
                "data": (
                    str_data[:max_data_length]
                    if len(str_data) > max_data_length
                    else serialized_data
                ),
                "target": (
                    str_target[:max_data_length]
                    if len(str_target) > max_data_length
                    else serialized_target
                ),
                "executorOutput": (
                    str_executor_output[:max_data_length]
                    if len(str_executor_output) > max_data_length
                    else serialized_executor_output
                ),
                "scores": self.scores,
                "traceId": str(self.trace_id),
                "executorSpanId": str(self.executor_span_id),
                "index": self.index,
                "metadata": (
                    serialize(self.metadata) if self.metadata is not None else {}
                ),
                "datasetLink": (
                    _dataset_link_to_dict(self.dataset_link)
                    if self.dataset_link is not None
                    else None
                ),
            }
        except Exception as e:
            raise ValueError(f"Error serializing EvaluationResultDatapoint: {e}")

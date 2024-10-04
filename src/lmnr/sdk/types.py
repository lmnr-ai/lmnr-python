import datetime
from enum import Enum
import pydantic
import requests
from typing import Any, Awaitable, Callable, Optional, Union
import uuid

from .utils import serialize


class ChatMessage(pydantic.BaseModel):
    role: str
    content: str


class ConditionedValue(pydantic.BaseModel):
    condition: str
    value: "NodeInput"


Numeric = Union[int, float]
NumericTypes = (int, float)  # for use with isinstance

NodeInput = Union[str, list[ChatMessage], ConditionedValue, Numeric, bool]
PipelineOutput = Union[NodeInput]


class PipelineRunRequest(pydantic.BaseModel):
    inputs: dict[str, NodeInput]
    pipeline: str
    env: dict[str, str] = pydantic.Field(default_factory=dict)
    metadata: dict[str, str] = pydantic.Field(default_factory=dict)
    stream: bool = pydantic.Field(default=False)
    parent_span_id: Optional[uuid.UUID] = pydantic.Field(default=None)
    trace_id: Optional[uuid.UUID] = pydantic.Field(default=None)

    # uuid is not serializable by default, so we need to convert it to a string
    def to_dict(self):
        return {
            "inputs": {
                k: v.model_dump() if isinstance(v, pydantic.BaseModel) else serialize(v)
                for k, v in self.inputs.items()
            },
            "pipeline": self.pipeline,
            "env": self.env,
            "metadata": self.metadata,
            "stream": self.stream,
            "parentSpanId": str(self.parent_span_id) if self.parent_span_id else None,
            "traceId": str(self.trace_id) if self.trace_id else None,
        }


class PipelineRunResponse(pydantic.BaseModel):
    outputs: dict[str, dict[str, PipelineOutput]]
    run_id: str


class PipelineRunError(Exception):
    error_code: str
    error_message: str

    def __init__(self, response: requests.Response):
        try:
            resp_json = response.json()
            self.error_code = resp_json["error_code"]
            self.error_message = resp_json["error_message"]
            super().__init__(self.error_message)
        except Exception:
            super().__init__(response.text)

    def __str__(self) -> str:
        try:
            return str(
                {"error_code": self.error_code, "error_message": self.error_message}
            )
        except Exception:
            return super().__str__()


EvaluationDatapointData = dict[str, Any]
EvaluationDatapointTarget = dict[str, Any]


# EvaluationDatapoint is a single data point in the evaluation
class Datapoint(pydantic.BaseModel):
    # input to the executor function. Must be a dict with string keys
    data: EvaluationDatapointData
    # input to the evaluator function (alongside the executor output).
    # Must be a dict with string keys
    target: EvaluationDatapointTarget


ExecutorFunctionReturnType = Any
EvaluatorFunctionReturnType = Union[Numeric, dict[str, Numeric]]

ExecutorFunction = Callable[
    [EvaluationDatapointData, Any, dict[str, Any]],
    Union[ExecutorFunctionReturnType, Awaitable[ExecutorFunctionReturnType]],
]

# EvaluatorFunction is a function that takes the output of the executor and the
# target data, and returns a score. The score can be a single number or a
# record of string keys and number values. The latter is useful for evaluating
# multiple criteria in one go instead of running multiple evaluators.
EvaluatorFunction = Callable[
    [ExecutorFunctionReturnType, Any, dict[str, Any]],
    Union[EvaluatorFunctionReturnType, Awaitable[EvaluatorFunctionReturnType]],
]


class CreateEvaluationResponse(pydantic.BaseModel):
    id: uuid.UUID
    createdAt: datetime.datetime
    groupId: str
    name: str
    projectId: uuid.UUID


class EvaluationResultDatapoint(pydantic.BaseModel):
    data: EvaluationDatapointData
    target: EvaluationDatapointTarget
    executor_output: ExecutorFunctionReturnType
    scores: dict[str, Numeric]
    trace_id: uuid.UUID

    # uuid is not serializable by default, so we need to convert it to a string
    def to_dict(self):
        return {
            "data": {
                k: v.model_dump() if isinstance(v, pydantic.BaseModel) else serialize(v)
                for k, v in self.data.items()
            },
            "target": {
                k: v.model_dump() if isinstance(v, pydantic.BaseModel) else serialize(v)
                for k, v in self.target.items()
            },
            "executorOutput": serialize(self.executor_output),
            "scores": self.scores,
            "traceId": str(self.trace_id),
        }


class SpanType(Enum):
    DEFAULT = "DEFAULT"
    LLM = "LLM"
    PIPELINE = "PIPELINE"  # must not be set manually
    EXECUTOR = "EXECUTOR"
    EVALUATOR = "EVALUATOR"
    EVALUATION = "EVALUATION"


class TraceType(Enum):
    DEFAULT = "DEFAULT"
    EVENT = "EVENT"  # must not be set manually
    EVALUATION = "EVALUATION"

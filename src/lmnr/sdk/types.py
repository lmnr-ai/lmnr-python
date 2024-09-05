import datetime
import requests
import pydantic
import uuid
from typing import Any, Awaitable, Callable, Literal, Optional, TypeAlias, Union

from .utils import to_dict


class ChatMessage(pydantic.BaseModel):
    role: str
    content: str


class ConditionedValue(pydantic.BaseModel):
    condition: str
    value: "NodeInput"


Numeric: TypeAlias = Union[int, float]
NodeInput: TypeAlias = Union[str, list[ChatMessage], ConditionedValue, Numeric, bool]
PipelineOutput: TypeAlias = Union[NodeInput]


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
                k: v.model_dump() if isinstance(v, pydantic.BaseModel) else to_dict(v)
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


EvaluationDatapointData: TypeAlias = dict[str, Any]
EvaluationDatapointTarget: TypeAlias = dict[str, Any]


# EvaluationDatapoint is a single data point in the evaluation
class EvaluationDatapoint(pydantic.BaseModel):
    # input to the executor function. Must be a dict with string keys
    data: EvaluationDatapointData
    # input to the evaluator function (alongside the executor output).
    # Must be a dict with string keys
    target: EvaluationDatapointTarget


ExecutorFunctionReturnType: TypeAlias = Any
EvaluatorFunctionReturnType: TypeAlias = Union[Numeric, dict[str, Numeric]]

ExecutorFunction: TypeAlias = Callable[
    [EvaluationDatapointData, *tuple[Any, ...], dict[str, Any]],
    Union[ExecutorFunctionReturnType, Awaitable[ExecutorFunctionReturnType]],
]

# EvaluatorFunction is a function that takes the output of the executor and the
# target data, and returns a score. The score can be a single number or a
# record of string keys and number values. The latter is useful for evaluating
# multiple criteria in one go instead of running multiple evaluators.
EvaluatorFunction: TypeAlias = Callable[
    [ExecutorFunctionReturnType, *tuple[Any, ...], dict[str, Any]],
    Union[EvaluatorFunctionReturnType, Awaitable[EvaluatorFunctionReturnType]],
]

EvaluationStatus: TypeAlias = Literal["Started", "Finished", "Error"]


class CreateEvaluationResponse(pydantic.BaseModel):
    id: uuid.UUID
    createdAt: datetime.datetime
    name: str
    status: EvaluationStatus
    projectId: uuid.UUID
    metadata: Optional[dict[str, Any]] = None


class EvaluationResultDatapoint(pydantic.BaseModel):
    data: EvaluationDatapointData
    target: EvaluationDatapointTarget
    executor_output: ExecutorFunctionReturnType
    scores: dict[str, Numeric]

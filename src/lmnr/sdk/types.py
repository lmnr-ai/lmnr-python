import logging
import datetime
from enum import Enum
import httpx
import json
from opentelemetry.trace import SpanContext, TraceFlags
import pydantic
from typing import Any, Awaitable, Callable, Literal, Optional, Union
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


class SemanticSearchRequest(pydantic.BaseModel):
    query: str
    dataset_id: uuid.UUID
    limit: Optional[int] = pydantic.Field(default=None)
    threshold: Optional[float] = pydantic.Field(default=None, ge=0.0, le=1.0)

    def to_dict(self):
        res = {
            "query": self.query,
            "datasetId": str(self.dataset_id),
        }
        if self.limit is not None:
            res["limit"] = self.limit
        if self.threshold is not None:
            res["threshold"] = self.threshold
        return res


class SemanticSearchResult(pydantic.BaseModel):
    dataset_id: uuid.UUID
    score: float
    data: dict[str, Any]
    content: str


class SemanticSearchResponse(pydantic.BaseModel):
    results: list[SemanticSearchResult]


class PipelineRunError(Exception):
    error_code: str
    error_message: str

    def __init__(self, response: httpx.Response):
        try:
            resp_json = response.json()
            try:
                resp_dict = dict(resp_json)
            except Exception:
                resp_dict = {}
            self.error_code = resp_dict.get("error_code")
            self.error_message = resp_dict.get("error_message")
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


EvaluationDatapointData = Any  # non-null, must be JSON-serializable
EvaluationDatapointTarget = Optional[Any]  # must be JSON-serializable
EvaluationDatapointMetadata = Optional[Any]  # must be JSON-serializable


# EvaluationDatapoint is a single data point in the evaluation
class Datapoint(pydantic.BaseModel):
    # input to the executor function.
    data: EvaluationDatapointData
    # input to the evaluator function (alongside the executor output).
    target: EvaluationDatapointTarget = pydantic.Field(default=None)
    metadata: EvaluationDatapointMetadata = pydantic.Field(default=None)


ExecutorFunctionReturnType = Any
EvaluatorFunctionReturnType = Union[Numeric, dict[str, Numeric]]

ExecutorFunction = Callable[
    [EvaluationDatapointData, Any],
    Union[ExecutorFunctionReturnType, Awaitable[ExecutorFunctionReturnType]],
]

# EvaluatorFunction is a function that takes the output of the executor and the
# target data, and returns a score. The score can be a single number or a
# record of string keys and number values. The latter is useful for evaluating
# multiple criteria in one go instead of running multiple evaluators.
EvaluatorFunction = Callable[
    [ExecutorFunctionReturnType, Any],
    Union[EvaluatorFunctionReturnType, Awaitable[EvaluatorFunctionReturnType]],
]


class HumanEvaluator(pydantic.BaseModel):
    queueName: str


class InitEvaluationResponse(pydantic.BaseModel):
    id: uuid.UUID
    createdAt: datetime.datetime
    groupId: str
    name: str
    projectId: uuid.UUID


class PartialEvaluationDatapoint(pydantic.BaseModel):
    id: uuid.UUID
    data: EvaluationDatapointData
    target: EvaluationDatapointTarget
    index: int
    trace_id: uuid.UUID
    executor_span_id: uuid.UUID

    # uuid is not serializable by default, so we need to convert it to a string
    def to_dict(self):
        try:
            return {
                "id": str(self.id),
                "data": str(serialize(self.data))[:100],
                "target": str(serialize(self.target))[:100],
                "index": self.index,
                "traceId": str(self.trace_id),
                "executorSpanId": str(self.executor_span_id),
            }
        except Exception as e:
            raise ValueError(f"Error serializing PartialEvaluationDatapoint: {e}")


class EvaluationResultDatapoint(pydantic.BaseModel):
    id: uuid.UUID
    index: int
    data: EvaluationDatapointData
    target: EvaluationDatapointTarget
    executor_output: ExecutorFunctionReturnType
    scores: dict[str, Numeric]
    human_evaluators: list[HumanEvaluator] = pydantic.Field(default_factory=list)
    trace_id: uuid.UUID
    executor_span_id: uuid.UUID

    # uuid is not serializable by default, so we need to convert it to a string
    def to_dict(self):
        try:
            return {
                # preserve only preview of the data, target and executor output
                # (full data is in trace)
                "id": str(self.id),
                "data": str(serialize(self.data))[:100],
                "target": str(serialize(self.target))[:100],
                "executorOutput": str(serialize(self.executor_output))[:100],
                "scores": self.scores,
                "traceId": str(self.trace_id),
                "humanEvaluators": [
                    (
                        v.model_dump()
                        if isinstance(v, pydantic.BaseModel)
                        else serialize(v)
                    )
                    for v in self.human_evaluators
                ],
                "executorSpanId": str(self.executor_span_id),
                "index": self.index,
            }
        except Exception as e:
            raise ValueError(f"Error serializing EvaluationResultDatapoint: {e}")


class SpanType(Enum):
    DEFAULT = "DEFAULT"
    LLM = "LLM"
    PIPELINE = "PIPELINE"  # must not be set manually
    EXECUTOR = "EXECUTOR"
    EVALUATOR = "EVALUATOR"
    EVALUATION = "EVALUATION"


class TraceType(Enum):
    DEFAULT = "DEFAULT"
    EVALUATION = "EVALUATION"


class GetDatapointsResponse(pydantic.BaseModel):
    items: list[Datapoint]
    totalCount: int


class TracingLevel(Enum):
    OFF = 0
    META_ONLY = 1
    ALL = 2


class LaminarSpanContext(pydantic.BaseModel):
    """
    A span context that can be used to continue a trace across services. This
    is a slightly modified version of the OpenTelemetry span context. For
    usage examples, see `Laminar.get_laminar_span_context_dict`,
    `Laminar.get_laminar_span_context_str`, `Laminar.get_span_context`, and
    `Laminar.deserialize_laminar_span_context`.

    The difference between this and the OpenTelemetry span context is that
    the `trace_id` and `span_id` are stored as UUIDs instead of integers for
    easier debugging, and the separate trace flags are not currently stored.
    """

    trace_id: uuid.UUID
    span_id: uuid.UUID
    is_remote: bool = pydantic.Field(default=False)

    def __str__(self) -> str:
        return self.model_dump_json()

    @classmethod
    def try_to_otel_span_context(
        cls,
        span_context: Union["LaminarSpanContext", dict[str, Any], str, SpanContext],
        logger: Optional[logging.Logger] = None,
    ) -> SpanContext:
        if logger is None:
            logger = logging.getLogger(__name__)

        if isinstance(span_context, LaminarSpanContext):
            return SpanContext(
                trace_id=span_context.trace_id.int,
                span_id=span_context.span_id.int,
                is_remote=span_context.is_remote,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
        elif isinstance(span_context, SpanContext) or (
            isinstance(getattr(span_context, "trace_id", None), int)
            and isinstance(getattr(span_context, "span_id", None), int)
        ):
            logger.warning(
                "span_context provided"
                " is likely a raw OpenTelemetry span context. Will try to use it. "
                "Please use `LaminarSpanContext` instead."
            )
            return span_context
        elif isinstance(span_context, dict) or isinstance(span_context, str):
            try:
                laminar_span_context = cls.deserialize(span_context)
                return SpanContext(
                    trace_id=laminar_span_context.trace_id.int,
                    span_id=laminar_span_context.span_id.int,
                    is_remote=laminar_span_context.is_remote,
                    trace_flags=TraceFlags(TraceFlags.SAMPLED),
                )
            except Exception:
                raise ValueError("Invalid span_context provided")
        else:
            raise ValueError("Invalid span_context provided")

    @classmethod
    def deserialize(cls, data: Union[dict[str, Any], str]) -> "LaminarSpanContext":
        if isinstance(data, dict):
            # Convert camelCase to snake_case for known fields
            converted_data = {
                "trace_id": data.get("trace_id") or data.get("traceId"),
                "span_id": data.get("span_id") or data.get("spanId"),
                "is_remote": data.get("is_remote") or data.get("isRemote", False),
            }
            return cls.model_validate(converted_data)
        elif isinstance(data, str):
            return cls.deserialize(json.loads(data))
        else:
            raise ValueError("Invalid span_context provided")


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"


# class AgentChatMessageContentTextBlock(pydantic.BaseModel):
#     type: Literal["text"]
#     text: str


# class AgentChatMessageImageUrlBlock(pydantic.BaseModel):
#     type: Literal["image"]
#     imageUrl: str


# class AgentChatMessageImageBase64Block(pydantic.BaseModel):
#     type: Literal["image"]
#     imageB64: str


# class AgentChatMessageImageBlock(pydantic.RootModel):
#     root: Union[AgentChatMessageImageUrlBlock, AgentChatMessageImageBase64Block]


# class AgentChatMessageContentBlock(pydantic.RootModel):
#     root: Union[AgentChatMessageContentTextBlock, AgentChatMessageImageBlock]


# class AgentChatMessageContent(pydantic.RootModel):
#     root: Union[str, list[AgentChatMessageContentBlock]]


# class AgentChatMessage(pydantic.BaseModel):
#     role: str
#     content: AgentChatMessageContent
#     name: Optional[str] = None
#     toolCallId: Optional[str] = None
#     isStateMessage: bool = False


# class AgentState(pydantic.BaseModel):
# messages: str = pydantic.Field(default="")
# messages: list[AgentChatMessage] = pydantic.Field(default_factory=list)
# browser_state: Optional[BrowserState] = None


class RunAgentRequest(pydantic.BaseModel):
    prompt: str
    state: Optional[str] = pydantic.Field(default=None)
    parent_span_context: Optional[str] = pydantic.Field(default=None)
    model_provider: Optional[ModelProvider] = pydantic.Field(default=None)
    model: Optional[str] = pydantic.Field(default=None)
    stream: bool = pydantic.Field(default=False)
    enable_thinking: bool = pydantic.Field(default=True)
    cdp_url: Optional[str] = pydantic.Field(default=None)
    return_screenshots: bool = pydantic.Field(default=False)

    def to_dict(self):
        result = {
            "prompt": self.prompt,
            "stream": self.stream,
            "enableThinking": self.enable_thinking,
            "returnScreenshots": self.return_screenshots,
        }
        if self.state:
            result["state"] = self.state
        if self.parent_span_context:
            result["parentSpanContext"] = self.parent_span_context
        if self.model_provider:
            result["modelProvider"] = self.model_provider.value
        if self.model:
            result["model"] = self.model
        if self.cdp_url:
            result["cdpUrl"] = self.cdp_url
        return result


class ActionResult(pydantic.BaseModel):
    isDone: bool = pydantic.Field(default=False)
    content: Optional[str] = pydantic.Field(default=None)
    error: Optional[str] = pydantic.Field(default=None)


class AgentOutput(pydantic.BaseModel):
    result: ActionResult = pydantic.Field(default_factory=ActionResult)


class StepChunkContent(pydantic.BaseModel):
    chunkType: Literal["step"]
    messageId: uuid.UUID
    actionResult: ActionResult
    summary: str
    screenshot: Optional[str] = pydantic.Field(default=None)


class FinalOutputChunkContent(pydantic.BaseModel):
    chunkType: Literal["finalOutput"]
    messageId: uuid.UUID
    content: AgentOutput


class RunAgentResponseChunk(pydantic.RootModel):
    root: Union[StepChunkContent, FinalOutputChunkContent]

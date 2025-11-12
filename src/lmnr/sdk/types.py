from __future__ import annotations  # For "Self" | str | ... type hint

import json
import logging
import datetime
import pydantic
import pydantic.alias_generators
import uuid

from enum import Enum
from opentelemetry.trace import SpanContext, TraceFlags
from typing import Any, Awaitable, Callable, Literal, Optional
from typing_extensions import TypedDict  # compatibility with python < 3.12

from .utils import serialize

DEFAULT_DATAPOINT_MAX_DATA_LENGTH = 16_000_000  # 16MB


Numeric = int | float
NumericTypes = (int, float)  # for use with isinstance

EvaluationDatapointData = Any  # non-null, must be JSON-serializable
EvaluationDatapointTarget = Any | None  # must be JSON-serializable
EvaluationDatapointMetadata = Any | None  # must be JSON-serializable


# EvaluationDatapoint is a single data point in the evaluation
class Datapoint(pydantic.BaseModel):
    # input to the executor function.
    data: EvaluationDatapointData
    # input to the evaluator function (alongside the executor output).
    target: EvaluationDatapointTarget = pydantic.Field(default_factory=dict)
    metadata: EvaluationDatapointMetadata = pydantic.Field(default_factory=dict)
    id: uuid.UUID | None = pydantic.Field(default=None)
    created_at: datetime.datetime | None = pydantic.Field(
        default=None, alias="createdAt"
    )


class Dataset(pydantic.BaseModel):
    id: uuid.UUID = pydantic.Field()
    name: str = pydantic.Field()
    created_at: datetime.datetime = pydantic.Field(alias="createdAt")


class PushDatapointsResponse(pydantic.BaseModel):
    dataset_id: uuid.UUID = pydantic.Field(alias="datasetId")


ExecutorFunctionReturnType = Any
EvaluatorFunctionReturnType = Numeric | dict[str, Numeric]

ExecutorFunction = Callable[
    [EvaluationDatapointData, Any],
    ExecutorFunctionReturnType | Awaitable[ExecutorFunctionReturnType],
]

# EvaluatorFunction is a function that takes the output of the executor and the
# target data, and returns a score. The score can be a single number or a
# record of string keys and number values. The latter is useful for evaluating
# multiple criteria in one go instead of running multiple evaluators.
EvaluatorFunction = Callable[
    [ExecutorFunctionReturnType, Any],
    EvaluatorFunctionReturnType | Awaitable[EvaluatorFunctionReturnType],
]


class HumanEvaluatorOptionsEntry(TypedDict):
    label: str
    value: float


class HumanEvaluator(pydantic.BaseModel):
    options: list[HumanEvaluatorOptionsEntry] = pydantic.Field(default_factory=list)


class InitEvaluationResponse(pydantic.BaseModel):
    id: uuid.UUID
    createdAt: datetime.datetime
    groupId: str
    name: str
    projectId: uuid.UUID


class EvaluationDatapointDatasetLink(pydantic.BaseModel):
    dataset_id: uuid.UUID
    datapoint_id: uuid.UUID
    created_at: datetime.datetime

    def to_dict(self):
        return {
            "datasetId": str(self.dataset_id),
            "datapointId": str(self.datapoint_id),
            "createdAt": self.created_at.isoformat(),
        }


class PartialEvaluationDatapoint(pydantic.BaseModel):
    id: uuid.UUID
    data: EvaluationDatapointData
    target: EvaluationDatapointTarget
    index: int
    trace_id: uuid.UUID
    executor_span_id: uuid.UUID
    metadata: EvaluationDatapointMetadata = pydantic.Field(default=None)
    dataset_link: EvaluationDatapointDatasetLink | None = pydantic.Field(default=None)

    # uuid is not serializable by default, so we need to convert it to a string
    def to_dict(self, max_data_length: int = DEFAULT_DATAPOINT_MAX_DATA_LENGTH):
        serialized_data = serialize(self.data)
        serialized_target = serialize(self.target)
        # TODO: use json_dumps instead of json.dumps once we
        # move it to utils so we can avoid circular imports
        str_data = json.dumps(serialized_data)
        str_target = json.dumps(serialized_target)
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
                    self.dataset_link.to_dict()
                    if self.dataset_link is not None
                    else None
                ),
            }
        except Exception as e:
            raise ValueError(f"Error serializing PartialEvaluationDatapoint: {e}")


class EvaluationResultDatapoint(pydantic.BaseModel):
    id: uuid.UUID
    index: int
    data: EvaluationDatapointData
    target: EvaluationDatapointTarget
    executor_output: ExecutorFunctionReturnType
    scores: dict[str, Optional[Numeric]]
    trace_id: uuid.UUID
    executor_span_id: uuid.UUID
    metadata: EvaluationDatapointMetadata = pydantic.Field(default=None)
    dataset_link: EvaluationDatapointDatasetLink | None = pydantic.Field(default=None)

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
                    self.dataset_link.to_dict()
                    if self.dataset_link is not None
                    else None
                ),
            }
        except Exception as e:
            raise ValueError(f"Error serializing EvaluationResultDatapoint: {e}")


class SpanType(Enum):
    DEFAULT = "DEFAULT"
    LLM = "LLM"
    PIPELINE = "PIPELINE"  # must not be set manually
    EXECUTOR = "EXECUTOR"
    EVALUATOR = "EVALUATOR"
    HUMAN_EVALUATOR = "HUMAN_EVALUATOR"
    EVALUATION = "EVALUATION"


class TraceType(Enum):
    DEFAULT = "DEFAULT"
    EVALUATION = "EVALUATION"


class GetDatapointsResponse(pydantic.BaseModel):
    items: list[Datapoint]
    total_count: int = pydantic.Field(alias="totalCount")


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
    span_path: list[str] = pydantic.Field(default=[])
    span_ids_path: list[str] = pydantic.Field(default=[])  # stringified UUIDs

    def __str__(self) -> str:
        return self.model_dump_json()

    @classmethod
    def try_to_otel_span_context(
        cls,
        span_context: "LaminarSpanContext" | dict[str, Any] | str | SpanContext,
        logger: logging.Logger | None = None,
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
        elif isinstance(span_context, (dict, str)):
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
    def deserialize(cls, data: dict[str, Any] | str) -> "LaminarSpanContext":
        if isinstance(data, dict):
            # Convert camelCase to snake_case for known fields
            converted_data = {
                "trace_id": data.get("trace_id") or data.get("traceId"),
                "span_id": data.get("span_id") or data.get("spanId"),
                "is_remote": data.get("is_remote") or data.get("isRemote", False),
                "span_path": data.get("span_path") or data.get("spanPath", []),
                "span_ids_path": data.get("span_ids_path")
                or data.get("spanIdsPath", []),
            }
            return cls.model_validate(converted_data)
        elif isinstance(data, str):
            return cls.deserialize(json.loads(data))
        else:
            raise ValueError("Invalid span_context provided")


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    OPENAI = "openai"
    GEMINI = "gemini"


class RunAgentRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
    )
    prompt: str
    storage_state: str | None = pydantic.Field(default=None)
    agent_state: str | None = pydantic.Field(default=None)
    parent_span_context: str | None = pydantic.Field(default=None)
    model_provider: ModelProvider | None = pydantic.Field(default=None)
    model: str | None = pydantic.Field(default=None)
    stream: bool = pydantic.Field(default=False)
    enable_thinking: bool = pydantic.Field(default=True)
    cdp_url: str | None = pydantic.Field(default=None)
    return_screenshots: bool = pydantic.Field(default=False)
    return_storage_state: bool = pydantic.Field(default=False)
    return_agent_state: bool = pydantic.Field(default=False)
    timeout: int | None = pydantic.Field(default=None)
    max_steps: int | None = pydantic.Field(default=None)
    thinking_token_budget: int | None = pydantic.Field(default=None)
    start_url: str | None = pydantic.Field(default=None)
    disable_give_control: bool = pydantic.Field(default=False)
    user_agent: str | None = pydantic.Field(default=None)


class ActionResult(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel
    )
    is_done: bool = pydantic.Field(default=False)
    content: str | None = pydantic.Field(default=None)
    error: str | None = pydantic.Field(default=None)


class AgentOutput(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel
    )
    result: ActionResult = pydantic.Field(default_factory=ActionResult)
    # Browser state with data related to auth, such as cookies.
    # A stringified JSON object.
    # Only returned if return_storage_state is True.
    # CAUTION: This object may become large. It also may contain sensitive data.
    storage_state: str | None = pydantic.Field(default=None)
    # Agent state with data related to the agent's state, such as the chat history.
    # A stringified JSON object.
    # Only returned if return_agent_state is True.
    # CAUTION: This object is large.
    agent_state: str | None = pydantic.Field(default=None)


class StepChunkContent(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel
    )
    chunk_type: Literal["step"] = pydantic.Field(default="step")
    message_id: uuid.UUID = pydantic.Field()
    action_result: ActionResult = pydantic.Field()
    summary: str = pydantic.Field()
    screenshot: str | None = pydantic.Field(default=None)


class TimeoutChunkContent(pydantic.BaseModel):
    """Chunk content to indicate that timeout has been hit. The only difference from a regular step
    is the chunk type. This is the last chunk in the stream.
    """

    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel
    )
    chunk_type: Literal["timeout"] = pydantic.Field(default="timeout")
    message_id: uuid.UUID = pydantic.Field()
    summary: str = pydantic.Field()
    screenshot: str | None = pydantic.Field(default=None)


class FinalOutputChunkContent(pydantic.BaseModel):
    """Chunk content to indicate that the agent has finished executing. This
    is the last chunk in the stream.
    """

    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel
    )

    chunk_type: Literal["finalOutput"] = pydantic.Field(default="finalOutput")
    message_id: uuid.UUID = pydantic.Field()
    content: AgentOutput = pydantic.Field()


class ErrorChunkContent(pydantic.BaseModel):
    """Chunk content to indicate that an error has occurred. Typically, this
    is the last chunk in the stream.
    """

    model_config = pydantic.ConfigDict(
        alias_generator=pydantic.alias_generators.to_camel
    )
    chunk_type: Literal["error"] = pydantic.Field(default="error")
    message_id: uuid.UUID = pydantic.Field()
    error: str = pydantic.Field()


class RunAgentResponseChunk(pydantic.RootModel):
    root: (
        StepChunkContent
        | FinalOutputChunkContent
        | ErrorChunkContent
        | TimeoutChunkContent
    )


class MaskInputOptions(TypedDict):
    textarea: bool | None
    text: bool | None
    number: bool | None
    select: bool | None
    email: bool | None
    tel: bool | None


class SessionRecordingOptions(TypedDict):
    mask_input_options: MaskInputOptions | None

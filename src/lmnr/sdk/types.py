from __future__ import annotations  # For "Self" | str | ... type hint

import datetime
import json
import logging
import uuid
from enum import Enum
from typing import Any, Literal, cast

from opentelemetry.trace import SpanContext, TraceFlags
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, override  # compatibility with python < 3.12


def parse_iso_datetime(value: str) -> datetime.datetime:
    """Parse an ISO-8601 timestamp from the API, tolerating a trailing 'Z'.

    `datetime.fromisoformat` only accepts a bare 'Z' suffix on Python >= 3.11;
    this package supports 3.10, and the app-server emits UTC timestamps with
    'Z'. Pydantic's own parser (speedate) handled this for us before these
    responses were plain dicts; this is the manual equivalent.
    """
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.datetime.fromisoformat(value)


Numeric = int | float
NumericTypes = (int, float)  # for use with isinstance

EvaluationDatapointData = Any  # pyright: ignore[reportExplicitAny] non-null, must be JSON-serializable
EvaluationDatapointTarget = Any | None  # pyright: ignore[reportExplicitAny] must be JSON-serializable
EvaluationDatapointMetadata = Any | None  # pyright: ignore[reportExplicitAny] must be JSON-serializable
LaminarSpanType = Literal[
    "DEFAULT",
    "LLM",
    "TOOL",
]


# EvaluationDatapoint is a single data point in the evaluation
class Datapoint(BaseModel):
    # input to the executor function.
    data: EvaluationDatapointData  # pyright: ignore[reportExplicitAny]
    # input to the evaluator function (alongside the executor output).
    target: EvaluationDatapointTarget = Field(default_factory=dict)  # pyright: ignore[reportAny]
    metadata: EvaluationDatapointMetadata = Field(default_factory=dict)  # pyright: ignore[reportAny]
    id: uuid.UUID | None = Field(default=None)
    created_at: datetime.datetime | None = Field(default=None, alias="createdAt")


class Dataset(TypedDict):
    id: uuid.UUID
    name: str
    created_at: datetime.datetime


def parse_dataset(data: dict[str, str]) -> Dataset:
    """Parse a `Dataset` from a `GET /v1/datasets` response entry."""
    return Dataset(
        id=uuid.UUID(str(data["id"])),
        name=data["name"],
        created_at=parse_iso_datetime(data["createdAt"]),
    )


class PushDatapointsResponse(TypedDict):
    dataset_id: uuid.UUID


def parse_push_datapoints_response(data: dict[str, str]) -> PushDatapointsResponse:
    """Parse a `PushDatapointsResponse` from a `POST /v1/datasets/datapoints`
    response."""
    return PushDatapointsResponse(dataset_id=uuid.UUID(str(data["datasetId"])))


class SpanType(Enum):
    DEFAULT = "DEFAULT"
    LLM = "LLM"
    PIPELINE = "PIPELINE"  # must not be set manually
    EXECUTOR = "EXECUTOR"
    EVALUATOR = "EVALUATOR"
    HUMAN_EVALUATOR = "HUMAN_EVALUATOR"
    EVALUATION = "EVALUATION"
    TOOL = "TOOL"


class TraceType(Enum):
    DEFAULT = "DEFAULT"
    EVALUATION = "EVALUATION"


class GetDatapointsResponse(TypedDict):
    items: list[Datapoint]
    total_count: int


def parse_get_datapoints_response(data: dict[str, int | list[dict[str, Any]]]) -> GetDatapointsResponse:  # pyright: ignore[reportExplicitAny]
    """Parse a `GetDatapointsResponse` from a `GET /v1/datasets/datapoints`
    response. Each item still goes through `Datapoint.model_validate` for its
    own uuid/datetime coercion."""
    return GetDatapointsResponse(
        items=[Datapoint.model_validate(item) for item in cast(list[dict[str, Any]], data["items"])],  # pyright: ignore[reportExplicitAny]
        total_count=cast(int, data["totalCount"]),
    )


class TraceBlockContent(TypedDict, total=False):
    """`content` of a `trace` session block."""

    traceId: str
    # Legacy note folded onto the trace block at ingest, if any.
    note: str | None


class EvaluationBlockContent(TypedDict, total=False):
    """`content` of an `evaluation` session block."""

    evaluationId: str
    # Legacy note folded onto the evaluation block at ingest, if any.
    note: str | None


class TextBlockContent(TypedDict):
    """`content` of a `text` session block — a standalone agent note."""

    text: str


# Block type the SDK knows how to render. `type` is a plain string on the wire
# so new block types can be added without a client bump; this is the set this
# SDK knows how to render.
SessionBlockType = Literal["trace", "evaluation", "text"]

# Union of the known block content shapes.
SessionBlockContent = TraceBlockContent | EvaluationBlockContent | TextBlockContent


class SessionBlock(TypedDict):
    """One block in a debugger session, as returned by
    `GET /v1/rollouts/{session_id}/blocks`.

    A debug session renders as an ordered list of blocks (the app-server
    `debugger_session_blocks` table). Each block has a plain-text `type` and a
    jsonb `content` whose shape depends on the type:

    - `trace`      — a trace produced under the session (`rollout.session_id`);
      written at ingest.
    - `evaluation` — an evaluation created under the session; written at eval
      creation.
    - `text`       — a free-text note the agent attaches post-factum via
      `lmnr-cli debug session add-note` (keyed by session id, not tied to any
      trace / eval).

    `content` is typed loosely (`dict[str, Any]`) because `type` is open-ended
    on the wire; narrow it with the `*BlockContent` shapes above once `type` is
    known. Kept parity with the TS SDK's `@lmnr-ai/types` `session-block.ts`.
    """

    # Block id (deterministic UUIDv5 for trace/eval blocks; random for text).
    id: str
    # ISO-8601 creation timestamp — the sort key for rendering oldest-first.
    createdAt: str
    # Block type; one of `SessionBlockType` for known blocks.
    type: str
    # Type-specific payload; narrow via the `*BlockContent` shapes.
    content: dict[str, Any]  # pyright: ignore[reportExplicitAny]


class DebugContext(TypedDict, total=False):
    """Debugger context propagated as ONE nested block of a LaminarSpanContext.

    Carries the debug-replay v2 coordinates a downstream run needs to consult
    the same server-side cache window as the run that produced this context.
    Laminar is the only producer; a hand-forged or `enabled=False` block is
    treated as absent by the consumer (behaviour is explicitly undefined).

    - `enabled`: armed flag — only `True` blocks are ever constructed by us.
    - `session_id`: the run's session id, kept VERBATIM. It is the exact string
      the origin registered with the backend, and `LMNR_DEBUG_SESSION_ID` may be
      an arbitrary (non-UUID) value — normalizing it here would drop or mutate
      the id so the downstream never joins the run.
    - `replay_trace_id`: the source trace to replay, kept VERBATIM (the origin
      sends it un-normalized as `replayTraceId` to the cache endpoint).
    - `cache_until`: the cache-window span-id needle, kept VERBATIM (hyphenated
      or not, full UUID or short suffix) — the server resolves it.

    `total=False` so a producer/test can omit keys it doesn't care about; every
    consumer reads via `.get(key, default)`, never attribute access.
    """

    enabled: bool
    session_id: str | None
    replay_trace_id: str | None
    cache_until: str | None


def deserialize_debug_context(data: dict[str, Any]) -> DebugContext:  # pyright: ignore[reportExplicitAny]
    """Parse a debug block from a dict, accepting camelCase and snake_case.

    All ids are kept VERBATIM: the producer emits the run's exact session /
    replay-trace / cache-until strings (un-normalized), so the consumer must
    round-trip them unchanged or a downstream run fails to join the run.
    """
    return DebugContext(
        # Strict `is True`, NOT bool(...): the producer always emits a real
        # boolean, so anything else (e.g. the string "false", which is
        # truthy) is a malformed/forged block and must NOT arm a downstream
        # runtime.
        enabled=data.get("enabled") is True,
        session_id=(data.get("session_id") or data.get("sessionId")) or None,
        replay_trace_id=(
            data.get("replay_trace_id") or data.get("replayTraceId")
        )
        or None,
        cache_until=(data.get("cache_until") or data.get("cacheUntil")) or None,
    )


#: Shape of a plain (non-`LaminarSpanContext`, non-`SpanContext`) dict a
#: caller can hand to `LaminarSpanContext.deserialize` /
#: `try_to_otel_span_context` -- e.g. a JSON-decoded header or env var.
SpanContextDict = dict[
    str, str | bool | int | float | list[str] | dict[str, str | bool | int | float]
]


class LaminarSpanContext(BaseModel):
    """
    A span context that can be used to continue a trace across services. This
    is a slightly modified version of the OpenTelemetry span context. For
    usage examples, see `Laminar.serialize_span_context`,
    `Laminar.get_span_context`, and `Laminar.deserialize_laminar_span_context`.

    The difference between this and the OpenTelemetry span context is that
    the `trace_id` and `span_id` are stored as UUIDs instead of integers for
    easier debugging, and the separate trace flags are not currently stored.
    """

    trace_id: uuid.UUID
    span_id: uuid.UUID
    is_remote: bool = Field(default=False)
    span_path: list[str] = Field(default=[])
    span_ids_path: list[str] = Field(default=[])  # stringified UUIDs
    user_id: str | None = Field(default=None)
    session_id: str | None = Field(default=None)
    trace_type: TraceType | None = Field(default=None)
    metadata: dict[str, dict[str, str | int | float | bool]] | None = Field(default=None)
    debug: DebugContext | None = Field(default=None)

    @override
    def __str__(self) -> str:
        return self.model_dump_json()

    @classmethod
    def try_to_otel_span_context(
        cls,
        span_context: LaminarSpanContext | SpanContextDict | str | SpanContext,
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
        elif isinstance(span_context, SpanContext):
            logger.warning(
                "span_context provided" +
                " is likely a raw OpenTelemetry span context. Will try to use it. " +
                "Please use `LaminarSpanContext` instead."
            )
            return span_context
        elif isinstance(getattr(span_context, "trace_id", None), int) and isinstance(
            getattr(span_context, "span_id", None), int
        ):
            # Not an actual `SpanContext` instance (that's handled above), but
            # duck-types as one (e.g. a `SpanContext` from a different otel
            # version). The `getattr` checks are the only signal here, so this
            # cast is trusting the same runtime check pyright can't itself see.
            logger.warning(
                "span_context provided" +
                " is likely a raw OpenTelemetry span context. Will try to use it. " +
                "Please use `LaminarSpanContext` instead."
            )
            return cast(SpanContext, cast(object, span_context))
        elif isinstance(span_context, (dict, str)):  # pyright: ignore[reportUnnecessaryIsInstance]
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
            raise TypeError("Invalid span_context provided")  # pyright: ignore[reportUnreachable]

    @classmethod
    def deserialize(cls, data: SpanContextDict | str) -> LaminarSpanContext:
        if isinstance(data, dict):
            # Convert camelCase to snake_case for known fields
            debug_raw = data.get("debug")
            converted_data = {
                "trace_id": data.get("trace_id") or data.get("traceId"),
                "span_id": data.get("span_id") or data.get("spanId"),
                "is_remote": data.get("is_remote") or data.get("isRemote", False),
                "span_path": data.get("span_path") or data.get("spanPath", []),
                "span_ids_path": data.get("span_ids_path")
                or data.get("spanIdsPath", []),
                "user_id": data.get("user_id") or data.get("userId"),
                "session_id": data.get("session_id") or data.get("sessionId"),
                "trace_type": data.get("trace_type") or data.get("traceType"),
                "metadata": data.get("metadata") or data.get("metadata", {}),
                "debug": (
                    deserialize_debug_context(debug_raw)
                    if isinstance(debug_raw, dict)
                    else None
                ),
            }
            return cls.model_validate(converted_data)
        elif isinstance(data, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            return cls.deserialize(json.loads(data))  # pyright: ignore[reportAny]
        else:
            raise TypeError("Invalid span_context provided")  # pyright: ignore[reportUnreachable]


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    OPENAI = "openai"
    GEMINI = "gemini"


class MaskInputOptions(TypedDict):
    textarea: bool | None
    text: bool | None
    number: bool | None
    select: bool | None
    email: bool | None
    tel: bool | None


class SessionRecordingOptions(TypedDict):
    mask_input_options: MaskInputOptions | None

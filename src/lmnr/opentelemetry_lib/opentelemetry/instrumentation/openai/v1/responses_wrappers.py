from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, NotRequired, Self, cast

import pydantic
from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import Span as SDKSpan
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Span, SpanKind, StatusCode
from typing_extensions import TypeVar, override
from wrapt import ObjectProxy

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.context_keys import (
    DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    model_as_dict,
    safe_start_span,
    set_span_attribute,
    should_send_prompts,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    stamp_instrumentation_scope,
)
from lmnr.opentelemetry_lib.tracing.context import (
    get_current_context,
    get_event_attributes_from_context,
)
from lmnr.sdk.debug.outcome import CacheOutcome
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import json_dumps
from openai import AsyncStream, Stream
from openai._legacy_response import LegacyAPIResponse
from openai._types import NotGiven, Omit

if TYPE_CHECKING:
    from ..rollout import AsyncCachedResponsesEventStream, CachedResponsesEventStream

# real types (first branch); the runtime fallback below keeps old SDKs working.
if TYPE_CHECKING:
    from openai.types.responses import (
        FunctionToolParam,
        Response,
        ResponseInputItemParam,
        ResponseInputParam,
        ResponseOutputItem,
        ResponseStreamEvent,
        ResponseUsage,
        ToolParam,
    )
    from openai.types.responses.response_output_message_param import (
        ResponseOutputMessageParam,
    )

    RESPONSES_AVAILABLE = True
else:
    try:
        from openai.types.responses import (
            FunctionToolParam,
            Response,
            ResponseInputItemParam,
            ResponseInputParam,
            ResponseOutputItem,
            ResponseStreamEvent,
            ResponseUsage,
            ToolParam,
        )
        from openai.types.responses.response_output_message_param import (
            ResponseOutputMessageParam,
        )

        RESPONSES_AVAILABLE = True
    except ImportError:
        # Fallback types for older OpenAI SDK versions
        FunctionToolParam = dict[str, Any]
        Response = Any
        ResponseInputItemParam = dict[str, Any]
        ResponseInputParam = str | list[dict[str, Any]]
        ResponseOutputItem = dict[str, Any]
        ResponseStreamEvent = Any
        ResponseUsage = dict[str, Any]
        ToolParam = dict[str, Any]
        ResponseOutputMessageParam = dict[str, Any]
        RESPONSES_AVAILABLE = False


SPAN_NAME = "openai.response"
# Statuses after which a response can no longer change (or be streamed again).
_TERMINAL_STATUSES = frozenset({"completed", "incomplete", "failed", "cancelled"})
# Header the SDK's `with_raw_response` / `with_streaming_response` set on the
# request; the call then returns an `APIResponse` wrapper, not the parsed body.
_RAW_RESPONSE_HEADER = "X-Stainless-Raw-Response"
logger = get_default_logger(__name__)
T = TypeVar("T")
S = TypeVar("S")


def _replay_enabled() -> bool:
    """True on a debug run with replay configured. Imported lazily to keep the
    debug machinery out of the instrumentation import path on normal runs."""
    try:
        from lmnr.sdk.debug.replay import replay_enabled

        return replay_enabled()
    except Exception:
        return False


def prepare_input_param(input_param: ResponseInputItemParam) -> ResponseInputItemParam:
    """
    Looks like OpenAI API infers the type "message" if the shape is correct,
    but type is not specified.
    It is marked as required on the message types. We add this to our
    traced data to make it work.
    """
    try:
        d = model_as_dict(input_param)
        if "type" not in d:
            d["type"] = "message"
        # ResponseInputItemParam is a Union of TypedDicts, so it cannot be
        # instantiated; `d` already has the right runtime shape.
        return cast(Any, d)
    except Exception:
        return input_param


def process_input(inp: ResponseInputParam) -> ResponseInputParam:
    if not isinstance(inp, list):  # pyright: ignore[reportUnnecessaryIsInstance]
        return inp
    return [prepare_input_param(item) for item in inp]


def is_validator_iterator(content: Any) -> re.Match[str] | None:
    """
    Some OpenAI objects contain fields typed as Iterable, which pydantic
    internally converts to a ValidatorIterator, and they cannot be trivially
    serialized without consuming the iterator to, for example, a list.

    See: https://github.com/pydantic/pydantic/issues/9541#issuecomment-2189045051
    """
    return re.search(r"pydantic.*ValidatorIterator'>$", str(type(content)))  # pyright: ignore[reportUnknownArgumentType]


# OpenAI API accepts output messages without an ID in its inputs, but
# the ID is marked as required in the output type.
if RESPONSES_AVAILABLE:

    class ResponseOutputMessageParamWithoutId(ResponseOutputMessageParam):
        id: NotRequired[str]  # pyright: ignore[reportGeneralTypeIssues]

else:
    # Fallback for older SDK versions
    ResponseOutputMessageParamWithoutId = dict  # pyright: ignore[reportAssignmentType]


class TracedData(pydantic.BaseModel):
    start_time: float  # time.time_ns()
    response_id: str
    # actually Union[str, list[Union[ResponseInputItemParam, ResponseOutputMessageParamWithoutId]]],
    # but this only works properly in Python 3.10+ / newer pydantic
    input: Any
    # system message
    instructions: str | None = pydantic.Field(default=None)
    # Any: pydantic would otherwise validate against the ToolParam union, which
    # rejects valid user tools (e.g. FunctionToolParam requires `strict`).
    tools: list[Any | ToolParam] | None = pydantic.Field(default=None)
    output_blocks: dict[str, ResponseOutputItem] | None = pydantic.Field(default=None)
    usage: ResponseUsage | None = pydantic.Field(default=None)
    output_text: str | None = pydantic.Field(default=None)
    request_model: str | None = pydantic.Field(default=None)
    response_model: str | None = pydantic.Field(default=None)

    # Reasoning attributes
    request_reasoning_summary: str | None = pydantic.Field(default=None)
    request_reasoning_effort: str | None = pydantic.Field(default=None)

    request_service_tier: str | None = pydantic.Field(default=None)
    response_service_tier: str | None = pydantic.Field(default=None)


responses: dict[str, TracedData] = {}


def parse_response(response: LegacyAPIResponse[Any] | Response) -> Response:
    if isinstance(response, LegacyAPIResponse):
        return response.parse()
    return response


def get_tools_from_kwargs(kwargs: dict[str, Any]) -> list[ToolParam]:
    tools_input = kwargs.get("tools", [])
    tools: list[Any] = []

    for tool in tools_input:
        if tool.get("type") == "function":
            if RESPONSES_AVAILABLE:
                tools.append(FunctionToolParam(**tool))
            else:
                tools.append(tool)

    return tools


def build_genai_input_messages(input_param: Any) -> list[dict[str, Any]] | None:
    """Build the `gen_ai.input.messages` array for a Responses-API request.

    Mirrors the LiteLLM responses path (`process_responses_inputs`): each input
    item is dumped as-is, so the array app-server stores as `spans.input` (it
    prefers `gen_ai.input.messages` over the legacy `gen_ai.prompt.N.*`
    reconstruction) is exactly what the debugger replay cache hashes. A bare
    string input — the common Responses-API case — is wrapped as a single user
    message so it is still cacheable (LiteLLM drops it; here we keep it). The
    system prompt lives in `instructions`, not the input array, and is excluded
    from the hash anyway, so it is intentionally not prepended.

    Returns None when there is no usable input (the replay path then runs live
    without hashing a partial input).
    """
    if isinstance(input_param, str):
        return [{"role": "user", "content": input_param}]
    if isinstance(input_param, list):
        return [model_as_dict(item) for item in cast(list[dict[str, Any]], input_param)]
    return None


def build_genai_output_messages(
    output_blocks: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build the `gen_ai.output.messages` array from a response's output blocks.

    Each block keeps its native `{type, ...}` shape (message / function_call /
    reasoning / *_call), matching the LiteLLM responses path and what
    `OpenAIRolloutWrapper.cached_response_to_responses` reads back as the cached
    `Response.output`. This is the attribute the replay cache stores as the
    recorded response, so without it a HIT carries no output.
    """
    if not output_blocks:
        return []
    return [model_as_dict(block) for block in output_blocks.values()]


def process_content_block(
    block: dict[str, Any],
) -> dict[str, Any]:
    # TODO: keep the original type once backend supports it
    if block.get("type") in ["text", "input_text", "output_text"]:
        return {"type": "text", "text": block.get("text")}
    elif block.get("type") in ["image", "input_image", "output_image"]:
        return {
            "type": "image",
            "image_url": block.get("image_url"),
            "detail": block.get("detail"),
            "file_id": block.get("file_id"),
        }
    elif block.get("type") in ["file", "input_file", "output_file"]:
        return {
            "type": "file",
            "file_id": block.get("file_id"),
            "filename": block.get("filename"),
            "file_data": block.get("file_data"),
        }
    return block


@dont_throw
def set_data_attributes(traced_response: TracedData, span: Span):
    set_span_attribute(span, "gen_ai.system", "openai")
    set_span_attribute(span, GEN_AI_REQUEST_MODEL, traced_response.request_model)
    set_span_attribute(span, GEN_AI_RESPONSE_ID, traced_response.response_id)
    set_span_attribute(span, GEN_AI_RESPONSE_MODEL, traced_response.response_model)
    if usage := traced_response.usage:
        set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)
        set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)
        set_span_attribute(span, "llm.usage.total_tokens", usage.total_tokens)
        if usage.input_tokens_details:
            set_span_attribute(
                span,
                "gen_ai.usage.cache_read_input_tokens",
                usage.input_tokens_details.cached_tokens,
            )

        reasoning_tokens = None
        if usage.output_tokens_details:
            reasoning_tokens = usage.output_tokens_details.reasoning_tokens

        set_span_attribute(
            span,
            "gen_ai.usage.reasoning_tokens",
            reasoning_tokens or 0,
        )

    set_span_attribute(
        span,
        "gen_ai.request.reasoning_summary",
        traced_response.request_reasoning_summary or (),
    )

    set_span_attribute(
        span,
        "gen_ai.request.reasoning_effort",
        traced_response.request_reasoning_effort or (),
    )

    set_span_attribute(
        span,
        "openai.request.service_tier",
        traced_response.request_service_tier,
    )
    set_span_attribute(
        span,
        "openai.response.service_tier",
        traced_response.response_service_tier,
    )

    if should_send_prompts():
        # Modern OTel GenAI attributes. app-server prefers these over the legacy
        # `gen_ai.prompt.N.*` / `gen_ai.completion.N.*` below when reconstructing
        # `spans.input` / `spans.output`, and the debugger replay cache hashes
        # `gen_ai.input.messages` + serves `gen_ai.output.messages` — so they must
        # be present for a Responses-API span to participate in replay.
        input_messages = build_genai_input_messages(traced_response.input)
        if input_messages is not None:
            set_span_attribute(span, "gen_ai.input.messages", json_dumps(input_messages))
        output_messages = build_genai_output_messages(traced_response.output_blocks)
        if output_messages:
            set_span_attribute(span, "gen_ai.output.messages", json_dumps(output_messages))

        prompt_index = 0
        if traced_response.tools:
            set_span_attribute(
                span,
                "gen_ai.tool.definitions",
                json_dumps([model_as_dict(tool) for tool in traced_response.tools]),
            )
        if traced_response.instructions:
            set_span_attribute(
                span,
                f"gen_ai.prompt.{prompt_index}.content",
                traced_response.instructions,
            )
            set_span_attribute(span, f"gen_ai.prompt.{prompt_index}.role", "system")
            prompt_index += 1

        if isinstance(traced_response.input, str):
            set_span_attribute(span, f"gen_ai.prompt.{prompt_index}.content", traced_response.input)
            set_span_attribute(span, f"gen_ai.prompt.{prompt_index}.role", "user")
            prompt_index += 1
        else:
            for block in traced_response.input:
                block_dict = model_as_dict(block)
                if block_dict.get("type", "message") == "message":
                    content: Any | None = block_dict.get("content")
                    if is_validator_iterator(content):
                        # we're after the actual call here, so we can consume the iterator
                        content = [
                            process_content_block(block)
                            for block in cast(list[dict[str, Any]], content or [])
                        ]
                    try:
                        stringified_content = (
                            content if isinstance(content, str) else json.dumps(content)
                        )
                    except Exception:
                        stringified_content = str(content) if content is not None else ""
                    set_span_attribute(
                        span,
                        f"gen_ai.prompt.{prompt_index}.content",
                        stringified_content,
                    )
                    set_span_attribute(
                        span,
                        f"gen_ai.prompt.{prompt_index}.role",
                        block_dict.get("role"),
                    )
                    prompt_index += 1
                elif block_dict.get("type") == "computer_call_output":
                    set_span_attribute(
                        span,
                        f"gen_ai.prompt.{prompt_index}.role",
                        "computer_call_output",
                    )
                    output_image_url = None
                    try:
                        output_image_url = block_dict.get("output", {}).get("image_url")
                    except Exception:
                        logger.debug("failed to get output image url", exc_info=True)
                    if output_image_url:
                        set_span_attribute(
                            span,
                            f"gen_ai.prompt.{prompt_index}.content",
                            json.dumps(
                                [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": output_image_url},
                                    }
                                ]
                            ),
                        )
                    prompt_index += 1
                elif block_dict.get("type") == "computer_call":
                    set_span_attribute(span, f"gen_ai.prompt.{prompt_index}.role", "assistant")
                    call_content = {}
                    if block_dict.get("id"):
                        call_content["id"] = block_dict.get("id")
                    if block_dict.get("action"):
                        call_content["action"] = block_dict.get("action")
                    set_span_attribute(
                        span,
                        f"gen_ai.prompt.{prompt_index}.tool_calls.0.arguments",
                        json.dumps(call_content),
                    )
                    set_span_attribute(
                        span,
                        f"gen_ai.prompt.{prompt_index}.tool_calls.0.id",
                        block_dict.get("call_id"),
                    )
                    set_span_attribute(
                        span,
                        f"gen_ai.prompt.{prompt_index}.tool_calls.0.name",
                        "computer_call",
                    )
                    prompt_index += 1
                elif block_dict.get("type") == "reasoning":
                    reasoning_summary = block_dict.get("summary")
                    if reasoning_summary and isinstance(reasoning_summary, list):
                        reasoning_summary = cast(list[dict[str, str]], reasoning_summary)
                        processed_chunks = [
                            {"type": "text", "text": chunk.get("text") or ""}
                            for chunk in reasoning_summary
                            if isinstance(chunk, dict)  # pyright: ignore[reportUnnecessaryIsInstance]
                            and chunk.get("type") == "summary_text"
                        ]
                        set_span_attribute(
                            span,
                            f"gen_ai.prompt.{prompt_index}.reasoning",
                            json_dumps(processed_chunks),
                        )
                        set_span_attribute(
                            span,
                            f"gen_ai.prompt.{prompt_index}.role",
                            "assistant",
                        )
                    # reasoning is followed by other content parts in the same messge,
                    # so we don't increment the prompt index
                # TODO: handle other block types

        set_span_attribute(span, "gen_ai.completion.0.role", "assistant")
        if traced_response.output_text:
            set_span_attribute(span, "gen_ai.completion.0.content", traced_response.output_text)
        tool_call_index = 0
        for block in (traced_response.output_blocks or {}).values():
            block_dict = model_as_dict(block)
            if block_dict.get("type") == "message":
                # either a refusal or handled in output_text above
                continue
            if block_dict.get("type") == "function_call":
                set_span_attribute(
                    span,
                    f"gen_ai.completion.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("id"),
                )
                set_span_attribute(
                    span,
                    f"gen_ai.completion.0.tool_calls.{tool_call_index}.name",
                    block_dict.get("name"),
                )
                set_span_attribute(
                    span,
                    f"gen_ai.completion.0.tool_calls.{tool_call_index}.arguments",
                    block_dict.get("arguments"),
                )
                tool_call_index += 1
            elif block_dict.get("type") == "file_search_call":
                set_span_attribute(
                    span,
                    f"gen_ai.completion.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("id"),
                )
                set_span_attribute(
                    span,
                    f"gen_ai.completion.0.tool_calls.{tool_call_index}.name",
                    "file_search_call",
                )
                tool_call_index += 1
            elif block_dict.get("type") == "web_search_call":
                set_span_attribute(
                    span,
                    f"gen_ai.completion.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("id"),
                )
                set_span_attribute(
                    span,
                    f"gen_ai.completion.0.tool_calls.{tool_call_index}.name",
                    "web_search_call",
                )
                tool_call_index += 1
            elif block_dict.get("type") == "computer_call":
                set_span_attribute(
                    span,
                    f"gen_ai.completion.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("call_id"),
                )
                set_span_attribute(
                    span,
                    f"gen_ai.completion.0.tool_calls.{tool_call_index}.name",
                    "computer_call",
                )
                set_span_attribute(
                    span,
                    f"gen_ai.completion.0.tool_calls.{tool_call_index}.arguments",
                    json.dumps(block_dict.get("action")),
                )
                tool_call_index += 1
            elif block_dict.get("type") == "reasoning":
                reasoning_summary = block_dict.get("summary")
                if reasoning_summary and isinstance(reasoning_summary, list):
                    reasoning_summary = cast(list[dict[str, str]], reasoning_summary)
                    processed_chunks = [
                        {"type": "text", "text": chunk.get("text")}
                        for chunk in reasoning_summary
                        if isinstance(chunk, dict)  # pyright: ignore[reportUnnecessaryIsInstance]
                        and chunk.get("type") == "summary_text"
                    ]
                    set_span_attribute(
                        span,
                        "gen_ai.completion.0.reasoning",
                        json_dumps(processed_chunks),
                    )
            # TODO: handle other block types, in particular other calls


def responses_get_or_create_wrapper(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T | Response | CachedResponsesEventStream:
    # Not @dont_throw: that would swallow the provider's own exceptions (and
    # return None to the caller). Instead every tracing helper called here is
    # @dont_throw, and a None from one falls back to the untraced result, so
    # `wrapped()` runs exactly once and its result or exception reaches the caller.
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)
    if context_api.get_value(
        DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY, get_current_context()
    ):
        return wrapped(*args, **kwargs)
    start_time = time.time_ns()

    # Debugger replay: probe the server-side cache before the live call.
    if _replay_enabled():
        return _replay_response_sync(to_wrap, start_time, wrapped, args, kwargs)

    traced_kwargs = _given_kwargs(kwargs)
    if traced_kwargs is None:
        return wrapped(*args, **kwargs)
    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        _process_exception(to_wrap, start_time, traced_kwargs, e)
        raise
    if _is_stream(response):
        traced = _trace_stream(to_wrap, start_time, response, traced_kwargs)
        return response if traced is None else traced
    _process_response(to_wrap, start_time, response, traced_kwargs)
    return response


async def async_responses_get_or_create_wrapper(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[T]],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T | Response | AsyncCachedResponsesEventStream:
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)
    if context_api.get_value(
        DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY, get_current_context()
    ):
        return await wrapped(*args, **kwargs)
    start_time = time.time_ns()

    if _replay_enabled():
        return await _replay_response_async(to_wrap, start_time, wrapped, args, kwargs)

    traced_kwargs = _given_kwargs(kwargs)
    if traced_kwargs is None:
        return await wrapped(*args, **kwargs)
    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:
        _process_exception(to_wrap, start_time, traced_kwargs, e)
        raise
    if _is_stream(response):
        traced = _trace_stream(to_wrap, start_time, response, traced_kwargs)
        return response if traced is None else traced
    _process_response(to_wrap, start_time, response, traced_kwargs)
    return response


def _is_stream(response: object) -> bool:
    # A bool rather than an inline isinstance, so `response` keeps its `T` type.
    return isinstance(response, (Stream, AsyncStream))


@dont_throw
def _given_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The request kwargs minus the SDK's `NOT_GIVEN` / `omit` placeholders.

    `responses.stream()` forwards every param it doesn't set as `omit`, which
    `kwargs.get(key, default)` would return in place of the default. Tracing
    reads this copy; the wrapped call still gets the original kwargs.

    Both sentinel classes are in `openai._types` since openai 1.0.0, older than the
    `openai._legacy_response` this module already needs, so they add no version floor.
    """
    return {key: value for key, value in kwargs.items() if not isinstance(value, NotGiven | Omit)}


@dont_throw
def _start_span(to_wrap: WrappedFunctionSpec, start_time: int) -> Span | None:
    span = safe_start_span(
        name=to_wrap.get("span_name") or SPAN_NAME,
        kind=SpanKind.CLIENT,
        start_time=start_time,
        context=get_current_context(),
        span_type="LLM",
    )
    if span is not None:
        stamp_instrumentation_scope(span, to_wrap)
    return span


@dont_throw
def _wrap_stream(
    span: Span,
    stream: T,
    start_time: int,
    kwargs: dict[str, Any],
    record_raw_response: bool = False,
) -> T:
    """Wrap a live `Stream` / `AsyncStream` of Responses-API events in a
    `TracedResponseStream` that ends `span`. The stream type is preserved for
    the caller: the proxy forwards everything it doesn't intercept."""
    return cast(
        T,
        TracedResponseStream(
            span, stream, start_time, kwargs, record_raw_response=record_raw_response
        ),
    )


@dont_throw
def _trace_stream(
    to_wrap: WrappedFunctionSpec,
    start_time: int,
    stream: T,
    kwargs: dict[str, Any],
) -> T | None:
    """Open the span for a live stream and wrap it. None if tracing failed."""
    span = _start_span(to_wrap, start_time)
    if span is None:
        return None
    traced = _wrap_stream(span, stream, start_time, kwargs)
    if traced is None:
        span.end()
    return traced


@dont_throw
def _open_replay_span(
    to_wrap: WrappedFunctionSpec, start_time: int, kwargs: dict[str, Any]
) -> Span | None:
    """Open the Responses-API span up front (before the live call) and stamp
    `gen_ai.input.messages` so the replay cache can hash the input.

    The bytes stamped here MUST equal what `set_data_attributes` later stamps,
    or the source-trace hash and the replay-run hash diverge. Both go through
    `build_genai_input_messages(process_input(kwargs["input"]))`.
    """
    span = _start_span(to_wrap, start_time)
    if span is None:
        return None
    if should_send_prompts():
        try:
            processed_input = process_input(cast(ResponseInputParam, kwargs.get("input")))
            input_messages = build_genai_input_messages(processed_input)
            if input_messages is not None:
                set_span_attribute(span, "gen_ai.input.messages", json_dumps(input_messages))
        except Exception:
            logger.debug("Failed to stamp Responses replay input", exc_info=True)
    return span


@dont_throw
def _cache_outcome(span: Span) -> CacheOutcome | None:
    from lmnr.sdk.debug.replay import cache_outcome_for

    return cache_outcome_for(cast(SDKSpan, span))


@dont_throw
async def _acache_outcome(span: Span) -> CacheOutcome | None:
    from lmnr.sdk.debug.replay import acache_outcome_for

    return await acache_outcome_for(cast(SDKSpan, span))


def _record_raw_response(span: Span, response: Any):
    """Stamp `lmnr.sdk.raw.response` (the raw provider response) so the source
    trace is cacheable as a `type="raw"` envelope, mirroring the chat path."""
    try:
        if hasattr(response, "model_dump_json"):
            set_span_attribute(span, "lmnr.sdk.raw.response", response.model_dump_json())
    except Exception:
        logger.debug("Failed to record raw Responses response")


@dont_throw
def _finish_live_replay(
    start_time: int,
    span: Span,
    response: Any,
    kwargs: dict[str, Any],
):
    """Process a live response onto the pre-opened replay span and end it."""
    try:
        parsed_response = parse_response(cast(Response, response))
        traced_data = _build_traced_data(start_time, parsed_response, kwargs)
        if traced_data is not None:
            set_data_attributes(traced_data, span)
            _record_raw_response(span, response)
    finally:
        span.end()


def _is_raw_response_request(kwargs: dict[str, Any]) -> bool:
    """True for `with_raw_response` / `with_streaming_response` calls, where the
    caller expects an `APIResponse` wrapper we can't fabricate from the cache."""
    extra_headers = kwargs.get("extra_headers")
    if not isinstance(extra_headers, dict):
        return False
    return _RAW_RESPONSE_HEADER in cast(dict[str, Any], extra_headers)


@dont_throw
def _serve_cached_response(
    start_time: int,
    span: Span,
    outcome: CacheOutcome,
    kwargs: dict[str, Any],
    make_stream: Callable[[list[ResponseStreamEvent]], S],
) -> Response | S | None:
    """On a cache HIT, reconstruct + serve an OpenAI `Response`, or for
    `stream=True` a `make_stream(events)` replaying it. Returns None to fall
    through to the live path (the span is then still open)."""
    if _is_raw_response_request(kwargs):
        return None

    from ..rollout import get_openai_rollout_wrapper

    rollout_wrapper = get_openai_rollout_wrapper()
    if rollout_wrapper is None:
        return None
    cached = rollout_wrapper.cached_response_to_responses(outcome.cached or {})
    if cached is None:
        return None
    served: Response | S = cached
    if kwargs.get("stream"):
        events = rollout_wrapper.cached_response_to_responses_events(cached)
        if events is None:
            return None
        served = make_stream(events)

    from lmnr.sdk.debug.replay import mark_span_cached

    traced_data = _build_traced_data(start_time, cached, kwargs, stash=False)
    if traced_data is not None:
        set_data_attributes(traced_data, span)
    mark_span_cached(cast(SDKSpan, span))
    span.end()
    return served


@dont_throw
def _end_span_with_exception(span: Span, e: BaseException):
    span.set_attribute(ERROR_TYPE, e.__class__.__name__)
    span.record_exception(e, attributes=get_event_attributes_from_context())
    span.set_status(StatusCode.ERROR, str(e))
    span.end()


def _replay_response_sync(
    to_wrap: WrappedFunctionSpec,
    start_time: int,
    wrapped: Callable[..., T],
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T | Response | CachedResponsesEventStream:
    from ..rollout import CachedResponsesEventStream

    traced_kwargs = _given_kwargs(kwargs)
    if traced_kwargs is None:
        return wrapped(*args, **kwargs)
    span = _open_replay_span(to_wrap, start_time, traced_kwargs)
    if span is None:
        return wrapped(*args, **kwargs)
    outcome = _cache_outcome(span)
    if outcome is not None and outcome.kind == "hit":
        served = _serve_cached_response(
            start_time, span, outcome, traced_kwargs, CachedResponsesEventStream
        )
        if served is not None:
            return served

    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        _end_span_with_exception(span, e)
        raise
    if _is_stream(response):
        traced = _wrap_stream(span, response, start_time, traced_kwargs, record_raw_response=True)
        if traced is not None:
            return traced
        span.end()
        return response
    _finish_live_replay(start_time, span, response, traced_kwargs)
    return response


async def _replay_response_async(
    to_wrap: WrappedFunctionSpec,
    start_time: int,
    wrapped: Callable[..., Awaitable[T]],
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T | Response | AsyncCachedResponsesEventStream:
    from ..rollout import AsyncCachedResponsesEventStream

    traced_kwargs = _given_kwargs(kwargs)
    if traced_kwargs is None:
        return await wrapped(*args, **kwargs)
    span = _open_replay_span(to_wrap, start_time, traced_kwargs)
    if span is None:
        return await wrapped(*args, **kwargs)
    outcome = await _acache_outcome(span)
    if outcome is not None and outcome.kind == "hit":
        served = _serve_cached_response(
            start_time, span, outcome, traced_kwargs, AsyncCachedResponsesEventStream
        )
        if served is not None:
            return served

    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:
        _end_span_with_exception(span, e)
        raise
    if _is_stream(response):
        traced = _wrap_stream(span, response, start_time, traced_kwargs, record_raw_response=True)
        if traced is not None:
            return traced
        span.end()
        return response
    _finish_live_replay(start_time, span, response, traced_kwargs)
    return response


def _get_request_reasoning(kwargs: dict[str, Any]) -> tuple[str | None, str | None]:
    """`(summary, effort)` from the request's `reasoning` param; tolerates pydantic
    models."""
    reasoning: Any = kwargs.get("reasoning")
    if isinstance(reasoning, pydantic.BaseModel):
        reasoning = reasoning.model_dump()
    if not isinstance(reasoning, dict):
        return None, None
    reasoning = cast(dict[str, Any], reasoning)
    summary = reasoning.get("summary")
    effort = reasoning.get("effort")
    return (
        summary if isinstance(summary, str) else None,
        effort if isinstance(effort, str) else None,
    )


def _traced_data_from_request(start_time: int, kwargs: dict[str, Any]) -> TracedData | None:
    """Build a `TracedData` from the request alone (merged with anything
    stashed under `kwargs["response_id"]`), for calls that produced no response
    to read from. Returns None if construction fails."""
    response_id = kwargs.get("response_id")
    existing_data: dict[str, Any] = {}
    if response_id and response_id in responses:
        existing_data = responses[response_id].model_dump()
    try:
        request_reasoning_summary, request_reasoning_effort = _get_request_reasoning(kwargs)
        return TracedData(
            start_time=existing_data.get("start_time", start_time),
            response_id=response_id or "",
            input=process_input(kwargs.get("input", existing_data.get("input", []))),
            instructions=kwargs.get("instructions", existing_data.get("instructions")),
            tools=get_tools_from_kwargs(kwargs) or existing_data.get("tools", []),
            output_blocks=existing_data.get("output_blocks", {}),
            usage=existing_data.get("usage"),
            output_text=kwargs.get("output_text", existing_data.get("output_text", "")),
            request_model=kwargs.get("model", existing_data.get("request_model", "")),
            response_model=existing_data.get("response_model", ""),
            request_reasoning_summary=request_reasoning_summary
            or existing_data.get("request_reasoning_summary"),
            request_reasoning_effort=request_reasoning_effort
            or existing_data.get("request_reasoning_effort"),
            request_service_tier=kwargs.get(
                "service_tier", existing_data.get("request_service_tier")
            ),
        )
    except Exception:
        logger.debug("Failed to build Responses traced data from request", exc_info=True)
        return None


@dont_throw
def _process_exception(
    to_wrap: WrappedFunctionSpec, start_time: int, kwargs: dict[str, Any], e: Exception
):
    traced_data = _traced_data_from_request(start_time, kwargs)
    span = _start_span(to_wrap, start_time if traced_data is None else int(traced_data.start_time))
    if span is None:
        return
    if traced_data:
        set_data_attributes(traced_data, span)
    _end_span_with_exception(span, e)


def _build_traced_data(
    start_time: int,
    parsed_response: Any,
    kwargs: dict[str, Any],
    stash: bool = True,
) -> TracedData | None:
    """Accumulate a `TracedData` for a parsed response, merging any prior data
    stashed under its id (request fields from the stash win, response fields
    from `parsed_response` win). With `stash`, the result is stored back under
    the id so a later `retrieve` / `cancel` can merge the original request.
    Returns None if construction fails."""
    response_id = getattr(parsed_response, "id", None)
    if not response_id:
        return None
    existing_data = responses.get(response_id)
    if existing_data is None:
        existing_data = {}
    else:
        existing_data = existing_data.model_dump()

    try:
        request_tools = get_tools_from_kwargs(kwargs)
        merged_tools = (existing_data.get("tools") or []) + request_tools
        request_reasoning_summary, request_reasoning_effort = _get_request_reasoning(kwargs)
        response_service_tier = None
        try:
            response_service_tier = parsed_response.service_tier
        except Exception:
            logger.debug("failed to get response service tier", exc_info=True)
        traced_data = TracedData(
            start_time=existing_data.get("start_time", start_time),
            response_id=response_id,
            input=process_input(cast(Any, existing_data.get("input", kwargs.get("input")))),
            instructions=existing_data.get("instructions", kwargs.get("instructions")),
            tools=merged_tools if merged_tools else None,
            output_blocks=(existing_data.get("output_blocks") or {})
            | {block.id: block for block in parsed_response.output},
            usage=parsed_response.usage or existing_data.get("usage"),
            output_text=_get_output_text(parsed_response) or existing_data.get("output_text"),
            request_model=existing_data.get("request_model", kwargs.get("model")),
            response_model=parsed_response.model or existing_data.get("response_model"),
            request_reasoning_summary=existing_data.get("request_reasoning_summary")
            or request_reasoning_summary,
            request_reasoning_effort=existing_data.get("request_reasoning_effort")
            or request_reasoning_effort,
            request_service_tier=existing_data.get(
                "request_service_tier", kwargs.get("service_tier")
            ),
            response_service_tier=response_service_tier
            or existing_data.get("response_service_tier"),
        )
        if stash:
            responses[response_id] = traced_data
        return traced_data
    except Exception:
        logger.debug("Failed to build Responses traced data", exc_info=True)
        return None


@dont_throw
def _process_response(
    to_wrap: WrappedFunctionSpec, start_time: int, response: Any, kwargs: dict[str, Any]
):
    parsed_response = parse_response(response)

    response_id = getattr(parsed_response, "id", None)
    if not response_id:
        return

    traced_data = _build_traced_data(start_time, parsed_response, kwargs)
    if traced_data is None:
        return

    if parsed_response.status == "completed":
        span = _start_span(to_wrap, int(traced_data.start_time))
        if span is not None:
            set_data_attributes(traced_data, span)
            span.end()


if TYPE_CHECKING:
    _ResponseStreamBase = ObjectProxy[Any]
else:
    # wrapt 1.x proxies are not subscriptable at runtime
    _ResponseStreamBase = ObjectProxy


class _SpanEndingHttpResponse(_ResponseStreamBase):
    """Proxy over an `httpx.Response` that runs `on_close` before closing it."""

    _on_close: Callable[[], None] | None = None

    @override
    def __init__(self, response: Any, on_close: Callable[[], None]):
        super().__init__(response)
        self._on_close = on_close

    def _notify(self):
        if self._on_close is not None:
            self._on_close()

    def close(self) -> None:
        self._notify()
        cast(Any, self.__wrapped__).close()

    async def aclose(self) -> None:
        self._notify()
        await cast(Any, self.__wrapped__).aclose()


class TracedResponseStream(_ResponseStreamBase):
    """Proxy over a Responses-API `Stream` / `AsyncStream` of `ResponseStreamEvent`s.

    The span is opened before the call, so it covers the whole stream and, on a
    debug run, already carries the hashed input. It is ended exactly once: when
    the stream is exhausted, raises, is closed or leaves its context manager, or
    is garbage collected half-consumed. Attributes come from the latest
    `response` snapshot an event carried (`response.completed` on the happy
    path), merged with the output items and text accumulated from the events,
    so an early exit still records what was streamed so far.

    Both `client.responses.create(stream=True)` and `client.responses.stream()`
    go through here: the latter wraps the stream `create` / `retrieve` return.
    """

    # Class-level defaults keep these attributes on the proxy: wrapt forwards
    # assignments to names the proxy class doesn't define to the wrapped stream.
    # Mutable ones are replaced per instance in __init__.
    _span: Span | None = None
    _start_time: int = 0
    _kwargs: dict[str, Any] = {}  # noqa: RUF012
    _record_raw_response: bool = False
    _output_text: str = ""
    _output_blocks: dict[str, ResponseOutputItem] = {}  # noqa: RUF012
    _response: Response | None = None
    _error_message: str | None = None
    _cleanup_completed: bool = False
    _cleanup_lock: threading.Lock = threading.Lock()

    @override
    def __init__(
        self,
        span: Span,
        stream: Any,
        start_time: int,
        kwargs: dict[str, Any],
        record_raw_response: bool = False,
    ):
        super().__init__(stream)
        self._span = span
        self._start_time = start_time
        self._kwargs = kwargs
        self._record_raw_response = record_raw_response
        self._output_text = ""
        self._output_blocks = {}
        self._response = None
        self._error_message = None
        self._cleanup_completed = False
        self._cleanup_lock = threading.Lock()

    @property
    def _stream(self) -> Any:
        return cast(Any, self.__wrapped__)

    def __del__(self):
        if not self._cleanup_completed:
            self._finish()

    def __enter__(self) -> Self:  # pyright: ignore[reportMissingSuperCall]
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(  # pyright: ignore[reportMissingSuperCall]
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        self._finish(exc_val)
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(exc_type, exc_val, exc_tb)
        return None

    async def __aenter__(self) -> Self:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        self._finish(exc_val)
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(exc_type, exc_val, exc_tb)
        return None

    def close(self) -> Any:
        """Ends the span, then closes the wrapped stream. `AsyncStream.close` is
        a coroutine function, so for async streams this returns the coroutine
        for the caller to await, exactly like the unwrapped call."""
        self._finish()
        return self._stream.close()

    @property
    def response(self) -> Any:
        """The stream's `httpx.Response`, wrapped to end the span when closed.

        `responses.stream()`'s `ResponseStream` never calls `close()` or exits
        this proxy: its `close()` / `__exit__` close `raw_stream.response`
        directly. Without this, breaking out of the helper early would leave
        the span open until garbage collection."""
        return _SpanEndingHttpResponse(self._stream.response, self._finish)

    def __iter__(self) -> Self:
        return self

    def __aiter__(self) -> Self:
        return self

    def __next__(self) -> ResponseStreamEvent:
        try:
            event = self._stream.__next__()
        except StopIteration:
            self._finish()
            raise
        except Exception as e:
            self._finish(e)
            raise
        self._process_event(event)
        return event

    async def __anext__(self) -> ResponseStreamEvent:
        try:
            event = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except Exception as e:
            self._finish(e)
            raise
        self._process_event(event)
        return event

    @dont_throw
    def _process_event(self, event: ResponseStreamEvent):
        if event.type == "response.output_text.delta":
            self._output_text += event.delta
        elif event.type == "response.output_item.done":
            item_id = getattr(event.item, "id", None) or f"output_{event.output_index}"
            self._output_blocks[item_id] = event.item
        elif event.type == "error":
            self._error_message = event.message

        # created / queued / in_progress / completed / incomplete / failed
        # events carry a snapshot of the whole response; keep the latest.
        snapshot = getattr(event, "response", None)
        if isinstance(snapshot, Response):
            self._response = snapshot

    def _build_traced_data(self) -> TracedData | None:
        response = self._response
        if response is None:
            traced_data = _traced_data_from_request(self._start_time, self._kwargs)
        else:
            # Only an unfinished background response can be picked up later
            # (by `retrieve`), so only it keeps its stashed request data.
            finished = response.status in _TERMINAL_STATUSES or not self._kwargs.get("background")
            traced_data = _build_traced_data(
                self._start_time, response, self._kwargs, stash=not finished
            )
            if finished:
                _ = responses.pop(response.id, None)
        if traced_data is None:
            return None
        # The final snapshot is authoritative; the accumulated items and text
        # fill in what an early exit never received a snapshot for.
        traced_data.output_blocks = self._output_blocks | (traced_data.output_blocks or {})
        if not traced_data.output_text:
            traced_data.output_text = self._output_text or None
        return traced_data

    @dont_throw
    def _finish(self, exception: BaseException | None = None):
        with self._cleanup_lock:
            if self._cleanup_completed:
                return
            self._cleanup_completed = True
            span = self._span
            if span is None or not span.is_recording():
                return
            try:
                traced_data = self._build_traced_data()
                if traced_data is not None:
                    set_data_attributes(traced_data, span)
                response = self._response
                if (
                    self._record_raw_response
                    and response is not None
                    and response.status in _TERMINAL_STATUSES
                ):
                    _record_raw_response(span, response)
            except Exception:
                logger.debug("Failed to record Responses stream data", exc_info=True)

            if exception is not None:
                _end_span_with_exception(span, exception)
                return
            error_message = self._error_message
            response = self._response
            if error_message is None and response is not None and response.status == "failed":
                error_message = response.error.message if response.error else "Response failed"
            if error_message is not None:
                span.set_status(StatusCode.ERROR, error_message)
            else:
                span.set_status(StatusCode.OK)
            span.end()


def responses_cancel_wrapper(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T:
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    if context_api.get_value(
        DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY, get_current_context()
    ):
        return wrapped(*args, **kwargs)

    response = wrapped(*args, **kwargs)
    _process_cancel(to_wrap, response)
    return response


async def async_responses_cancel_wrapper(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[T]],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T:
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    if context_api.get_value(
        DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY, get_current_context()
    ):
        return await wrapped(*args, **kwargs)

    response = await wrapped(*args, **kwargs)
    _process_cancel(to_wrap, response)
    return response


@dont_throw
def _process_cancel(to_wrap: WrappedFunctionSpec, response: Any):
    """Emit the span for a stashed (background) response that got cancelled."""
    if isinstance(response, (Stream, AsyncStream)):
        return
    parsed_response = parse_response(cast(Response, response))
    response_id = getattr(parsed_response, "id", None)
    if not response_id:
        return
    existing_data = responses.pop(response_id, None)
    if existing_data is None:
        return
    span = _start_span(to_wrap, int(existing_data.start_time))
    if span is None:
        return
    span.record_exception(
        Exception("Response cancelled"),
        attributes=get_event_attributes_from_context(),
    )
    set_data_attributes(existing_data, span)
    span.end()


def _get_output_text(parsed_response: Response) -> str | None:
    output_text = None
    if hasattr(parsed_response, "output_text"):
        output_text = parsed_response.output_text
    else:
        try:
            if output := parsed_response.output:
                content = output[0]
                output_text = getattr(content, "text", None)
        except Exception:
            logger.debug("failed to get output text from Response", exc_info=True)
    return output_text

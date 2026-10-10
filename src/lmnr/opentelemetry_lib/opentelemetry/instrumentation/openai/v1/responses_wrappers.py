import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

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
from typing_extensions import NotRequired, TypeVar

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.helpers import (
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

# real types (first branch); the runtime fallback below keeps old SDKs working.
if TYPE_CHECKING:
    from openai.types.responses import (
        FunctionToolParam,
        Response,
        ResponseInputItemParam,
        ResponseInputParam,
        ResponseOutputItem,
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
        ResponseUsage = dict[str, Any]
        ToolParam = dict[str, Any]
        ResponseOutputMessageParam = dict[str, Any]
        RESPONSES_AVAILABLE = False


SPAN_NAME = "openai.response"
logger = get_default_logger(__name__)
T = TypeVar("T")

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
    output_blocks: dict[str, ResponseOutputItem] | None = pydantic.Field(
        default=None
    )
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
            set_span_attribute(
                span, "gen_ai.input.messages", json_dumps(input_messages)
            )
        output_messages = build_genai_output_messages(traced_response.output_blocks)
        if output_messages:
            set_span_attribute(
                span, "gen_ai.output.messages", json_dumps(output_messages)
            )

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
            set_span_attribute(
                span, f"gen_ai.prompt.{prompt_index}.content", traced_response.input
            )
            set_span_attribute(span, f"gen_ai.prompt.{prompt_index}.role", "user")
            prompt_index += 1
        else:
            for block in traced_response.input:
                block_dict = model_as_dict(block)
                if block_dict.get("type", "message") == "message":
                    content: Any | None = block_dict.get("content")
                    if is_validator_iterator(content):
                        # we're after the actual call here, so we can consume the iterator
                        content = [process_content_block(block) for block in cast(list[dict[str, Any]], content or [])]
                    try:
                        stringified_content = (
                            content if isinstance(content, str) else json.dumps(content)
                        )
                    except Exception:
                        stringified_content = (
                            str(content) if content is not None else ""
                        )
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
                    set_span_attribute(
                        span, f"gen_ai.prompt.{prompt_index}.role", "assistant"
                    )
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
            set_span_attribute(
                span, "gen_ai.completion.0.content", traced_response.output_text
            )
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


@dont_throw
def responses_get_or_create_wrapper(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T | Response | Stream[Any]:
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)
    if context_api.get_value(
        DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY, get_current_context()
    ):
        return wrapped(*args, **kwargs)
    start_time = time.time_ns()

    # Debugger replay (non-streaming only): probe the server-side cache before
    # the live call. Streaming responses fall through to the live path below.
    if _replay_enabled() and not kwargs.get("stream", False):
        return _replay_response_sync(to_wrap, start_time, cast(Callable[..., Any], wrapped), args, kwargs)

    try:
        response = wrapped(*args, **kwargs)
        if isinstance(response, Stream):
            return response
    except Exception as e:
        _process_exception(to_wrap, start_time, kwargs, e)
        raise
    return _process_response(to_wrap, start_time, response, kwargs)


@dont_throw
async def async_responses_get_or_create_wrapper(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[T]],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T | Response | AsyncStream[Any] | Stream[Any]:
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)
    if context_api.get_value(
        DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY, get_current_context()
    ):
        return await wrapped(*args, **kwargs)
    start_time = time.time_ns()

    if _replay_enabled() and not kwargs.get("stream", False):
        return await _replay_response_async(to_wrap, start_time, wrapped, args, kwargs)

    try:
        response = await wrapped(*args, **kwargs)
        if isinstance(response, (Stream, AsyncStream)):
            return response
    except Exception as e:
        _process_exception(to_wrap, start_time, kwargs, e)
        raise
    return _process_response(to_wrap, start_time, response, kwargs)


def _open_replay_span(
    to_wrap: WrappedFunctionSpec, start_time: int, kwargs: dict[str, Any]
) -> Span | None:
    """Open the Responses-API span up front (before the live call) and stamp
    `gen_ai.input.messages` so the replay cache can hash the input.

    The bytes stamped here MUST equal what `set_data_attributes` later stamps,
    or the source-trace hash and the replay-run hash diverge. Both go through
    `build_genai_input_messages(process_input(kwargs["input"]))`.
    """
    span = safe_start_span(
        name=to_wrap.get("span_name") or SPAN_NAME,
        kind=SpanKind.CLIENT,
        start_time=start_time,
        context=get_current_context(),
        span_type="LLM",
    )
    if span is None:
        return None
    stamp_instrumentation_scope(span, to_wrap)
    if should_send_prompts():
        processed_input = process_input(cast(ResponseInputParam, kwargs.get("input")))
        input_messages = build_genai_input_messages(processed_input)
        if input_messages is not None:
            set_span_attribute(
                span, "gen_ai.input.messages", json_dumps(input_messages)
            )
    return span


def _record_raw_response(span: Span, response: Any):
    """Stamp `lmnr.sdk.raw.response` (the raw provider response) so the source
    trace is cacheable as a `type="raw"` envelope, mirroring the chat path."""
    try:
        if hasattr(response, "model_dump_json"):
            set_span_attribute(
                span, "lmnr.sdk.raw.response", response.model_dump_json()
            )
    except Exception:
        logger.debug("Failed to record raw Responses response")


def _finish_live_replay(
    start_time: int,
    span: Span,
    response: T,
    kwargs: dict[str, Any],
) -> T:
    """Process a live response onto the pre-opened replay span and end it."""
    parsed_response = parse_response(cast(Response, response))
    traced_data = _build_traced_data(start_time, parsed_response, kwargs)
    if traced_data is not None:
        set_data_attributes(traced_data, span)
        _record_raw_response(span, response)
    span.end()
    return response


def _serve_cached_response(
    start_time: int,
    span: Span,
    outcome: CacheOutcome,
    kwargs: dict[str, Any]
) -> Response | None:
    """On a cache HIT, reconstruct + serve an OpenAI `Response`. Returns the
    served response, or None to fall through to the live path."""
    from ..rollout import get_openai_rollout_wrapper

    rollout_wrapper = get_openai_rollout_wrapper()
    if rollout_wrapper is None:
        return None
    cached = rollout_wrapper.cached_response_to_responses(outcome.cached or {})
    if cached is None:
        return None

    from lmnr.sdk.debug.replay import mark_span_cached

    traced_data = _build_traced_data(start_time, cached, kwargs)
    if traced_data is not None:
        set_data_attributes(traced_data, span)
    mark_span_cached(cast(SDKSpan, span))
    span.end()
    return cached


def _replay_response_sync(
    to_wrap: WrappedFunctionSpec,
    start_time: int,
    wrapped: Callable[..., Response],
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Response | Stream[Any]:
    from lmnr.sdk.debug.replay import cache_outcome_for

    span = _open_replay_span(to_wrap, start_time, kwargs)
    if span is None:
        return wrapped(*args, **kwargs)
    outcome = cache_outcome_for(cast(SDKSpan, span))
    if outcome is not None and outcome.kind == "hit":
        served = _serve_cached_response(start_time, span, outcome, kwargs)
        if served is not None:
            return served

    try:
        response = wrapped(*args, **kwargs)
        if isinstance(response, Stream):
            span.end()
            return response
    except Exception as e:
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=get_event_attributes_from_context())
        span.set_status(StatusCode.ERROR, str(e))
        span.end()
        raise
    return _finish_live_replay(start_time, span, response, kwargs)


async def _replay_response_async(
    to_wrap: WrappedFunctionSpec,
    start_time: int,
    wrapped: Callable[..., Awaitable[T]],
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T | Response | Stream[Any] | AsyncStream[Any]:
    from lmnr.sdk.debug.replay import acache_outcome_for

    span = _open_replay_span(to_wrap, start_time, kwargs)
    if span is None:
        return await wrapped(*args, **kwargs)
    outcome = await acache_outcome_for(cast(SDKSpan, span))
    if outcome is not None and outcome.kind == "hit":
        served = _serve_cached_response(start_time, span, outcome, kwargs)
        if served is not None:
            return served

    try:
        response = await wrapped(*args, **kwargs)
        if isinstance(response, (Stream, AsyncStream)):
            span.end()
            return response
    except Exception as e:
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=get_event_attributes_from_context())
        span.set_status(StatusCode.ERROR, str(e))
        span.end()
        raise
    return _finish_live_replay(start_time, span, response, kwargs)


@dont_throw
def _process_exception(to_wrap: WrappedFunctionSpec, start_time: int, kwargs: dict[str, Any], e: Exception):
    response_id = kwargs.get("response_id")
    existing_data = {}
    if response_id and response_id in responses:
        existing_data = responses[response_id].model_dump()
    try:
        request_reasoning_summary = None
        request_reasoning_effort = None
        request_reasoning = kwargs.get("reasoning", {})
        try:
            request_reasoning_summary = request_reasoning.get("summary")
        except Exception:
            logger.debug("failed to get request reasoning summary", exc_info=True)
        try:
            request_reasoning_effort = request_reasoning.get("effort")
        except Exception:
            logger.debug("failed to get request reasoning effort", exc_info=True)
        traced_data = TracedData(
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
            # response_service_tier=existing_data.get("response_service_tier"),
        )
    except Exception:
        traced_data = None

    span = safe_start_span(
        name=to_wrap.get("span_name") or SPAN_NAME,
        kind=SpanKind.CLIENT,
        start_time=(start_time if traced_data is None else int(traced_data.start_time)),
        context=get_current_context(),
        span_type="LLM",
    )
    if span is None:
        return
    stamp_instrumentation_scope(span, to_wrap)
    span.set_attribute(ERROR_TYPE, e.__class__.__name__)
    span.record_exception(e, attributes=get_event_attributes_from_context())
    span.set_status(StatusCode.ERROR, str(e))
    if traced_data:
        set_data_attributes(traced_data, span)
    span.end()


def _build_traced_data(
    start_time: int,
    parsed_response: Any,
    kwargs: dict[str, Any]
) -> TracedData | None:
    """Accumulate a `TracedData` for a parsed response, merging any prior data
    stashed under its id. Returns None if construction fails."""
    response_id = getattr(parsed_response, "id", None)
    if not response_id:
        return None
    existing_data = responses.get(response_id)
    if existing_data is None:
        existing_data = {}
    else:
        existing_data = existing_data.model_dump()

    request_tools = get_tools_from_kwargs(kwargs)
    merged_tools = existing_data.get("tools", []) + request_tools

    try:
        request_reasoning_summary = None
        request_reasoning_effort = None
        request_reasoning = kwargs.get("reasoning", {})
        response_service_tier = None
        try:
            request_reasoning_summary = request_reasoning.get("summary")
        except Exception:
            logger.debug("failed to get response reasoning summary", exc_info=True)
        try:
            request_reasoning_effort = request_reasoning.get("effort")
        except Exception:
            logger.debug("failed to get response reasoning effort", exc_info=True)
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
            output_blocks={block.id: block for block in parsed_response.output}
            | existing_data.get("output_blocks", {}),
            usage=existing_data.get("usage", parsed_response.usage),
            output_text=existing_data.get(
                "output_text", _get_output_text(parsed_response)
            ),
            request_model=existing_data.get("request_model", kwargs.get("model")),
            response_model=existing_data.get("response_model", parsed_response.model),
            request_reasoning_summary=existing_data.get("request_reasoning_summary")
            or request_reasoning_summary,
            request_reasoning_effort=existing_data.get("request_reasoning_effort")
            or request_reasoning_effort,
            request_service_tier=existing_data.get(
                "request_service_tier", kwargs.get("service_tier")
            ),
            response_service_tier=existing_data.get(
                "response_service_tier",
                response_service_tier,
            ),
        )
        responses[response_id] = traced_data
        return traced_data
    except Exception:
        return None


@dont_throw
def _process_response(
    to_wrap: WrappedFunctionSpec,
    start_time: int,
    response: Any,
    kwargs: dict[str, Any]
) -> Any:
    parsed_response = parse_response(response)

    response_id = getattr(parsed_response, "id", None)
    if not response_id:
        return response

    traced_data = _build_traced_data(start_time, parsed_response, kwargs)
    if traced_data is None:
        return response

    if parsed_response.status == "completed":
        span = safe_start_span(
            name=to_wrap.get("span_name") or SPAN_NAME,
            kind=SpanKind.CLIENT,
            start_time=int(traced_data.start_time),
            context=get_current_context(),
            span_type="LLM",
        )
        if span is not None:
            stamp_instrumentation_scope(span, to_wrap)
            set_data_attributes(traced_data, span)
            span.end()

    return response


@dont_throw
def responses_cancel_wrapper(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T | Stream[Any]:
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    if context_api.get_value(
        DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY, get_current_context()
    ):
        return wrapped(*args, **kwargs)

    response = wrapped(*args, **kwargs)
    if isinstance(response, Stream):
        return response
    parsed_response = parse_response(cast(Response, response))
    response_id = getattr(parsed_response, "id", None)
    if not response_id:
        return response
    existing_data = responses.pop(response_id, None)
    if existing_data is not None:
        span = safe_start_span(
            name=to_wrap.get("span_name") or SPAN_NAME,
            kind=SpanKind.CLIENT,
            start_time=int(existing_data.start_time),
            context=get_current_context(),
            span_type="LLM",
        )
        if span is not None:
            stamp_instrumentation_scope(span, to_wrap)
            span.record_exception(
                Exception("Response cancelled"),
                attributes=get_event_attributes_from_context(),
            )
            set_data_attributes(existing_data, span)
            span.end()
    return response


@dont_throw
async def async_responses_cancel_wrapper(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[T]],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T | Stream[Any] | AsyncStream[Any]:
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    if context_api.get_value(
        DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY, get_current_context()
    ):
        return await wrapped(*args, **kwargs)

    response = await wrapped(*args, **kwargs)
    if isinstance(response, (Stream, AsyncStream)):
        return response
    parsed_response = parse_response(cast(Response, response))
    response_id = getattr(parsed_response, "id", None)
    if not response_id:
        return response
    existing_data = responses.pop(response_id, None)
    if existing_data is not None:
        span = safe_start_span(
            name=to_wrap.get("span_name") or SPAN_NAME,
            kind=SpanKind.CLIENT,
            start_time=int(existing_data.start_time),
            context=get_current_context(),
            span_type="LLM",
        )
        if span is not None:
            stamp_instrumentation_scope(span, to_wrap)
            span.record_exception(
                Exception("Response cancelled"),
                attributes=get_event_attributes_from_context(),
            )
            set_data_attributes(existing_data, span)
            span.end()
    return response


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

# TODO: build streaming responses

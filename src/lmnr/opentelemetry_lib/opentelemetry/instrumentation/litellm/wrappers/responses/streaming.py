from collections.abc import AsyncIterator, Iterator
from typing import Any, TypedDict, TypeVar, cast

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.trace import Span, Status, StatusCode

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    set_span_attribute,
    to_dict,
)
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import json_dumps

logger = get_default_logger(__name__)
T = TypeVar("T")

class _Usage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_tokens_details: dict[str, Any]  # pyright: ignore[reportExplicitAny]


class _FinalResponse(TypedDict, total=False):
    id: str
    model: str
    usage: _Usage
    reasoning: dict[str, Any]  # pyright: ignore[reportExplicitAny]
    output: list[Any]  # pyright: ignore[reportExplicitAny]


@dont_throw
def _set_final_response_attributes(
    span: Span, final_response: _FinalResponse, record_raw_response: bool = False
):
    """Set span attributes from the final completed response."""
    try:
        set_span_attribute(span, "gen_ai.response.id", final_response.get("id"))
        set_span_attribute(span, "gen_ai.response.model", final_response.get("model"))

        # Handle usage information
        if usage := final_response.get("usage"):
            usage_dict = cast(_Usage, cast(object, to_dict(usage)))
            input_tokens = usage_dict.get(
                "input_tokens", usage_dict.get("prompt_tokens", 0)
            ) or 0
            output_tokens = usage_dict.get(
                "output_tokens", usage_dict.get("completion_tokens", 0)
            ) or 0
            total_tokens = usage_dict.get("total_tokens", input_tokens + output_tokens)
            set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
            set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
            set_span_attribute(span, "llm.usage.total_tokens", total_tokens)
            if input_details := usage_dict.get("input_tokens_details"):
                details = to_dict(input_details)
                cache_read_tokens = cast(int, details.get("cached_tokens") or 0)
                set_span_attribute(
                    span, "gen_ai.usage.cache_read_input_tokens", cache_read_tokens
                )

        # Handle output messages/items
        final_items: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]
        if reasoning := final_response.get("reasoning"):
            reasoning_dict = to_dict(reasoning)
            if reasoning_dict.get("summary") or reasoning_dict.get("effort"):
                final_items.append(reasoning_dict)
        for item in final_response.get("output") or []:  # pyright: ignore[reportAny]
            final_items.append(to_dict(item))

        span.set_attribute("gen_ai.output.messages", json_dumps(final_items))

        # Record raw response in rollout mode
        if record_raw_response:
            set_span_attribute(
                span, "lmnr.sdk.raw.response", json_dumps(dict(final_response))
            )
    finally:
        span.end()


def process_responses_streaming_response(
    span: Span,
    stream: Iterator[T],
    record_raw_response: bool = False,
) -> Iterator[T]:
    """
    Process streaming responses for sync responses.
    The stream contains various event types, but we primarily care about
    the final 'response.completed' event which has all the data we need.
    """
    final_response = None
    try:
        for chunk in stream:
            chunk_dict = to_dict(chunk)
            # Check if this is the final completed event
            if chunk_dict.get("type") == "response.completed":
                final_response = chunk_dict.get("response")
            yield chunk
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        if final_response:
            _set_final_response_attributes(
                span, cast(_FinalResponse, cast(object, to_dict(final_response))), record_raw_response
            )
        else:
            span.end()


async def process_responses_async_streaming_response(
    span: Span,
    stream: AsyncIterator[T],
    record_raw_response: bool = False,
) -> AsyncIterator[T]:
    """
    Process streaming responses for async responses.
    The stream contains various event types, but we primarily care about
    the final 'response.completed' event which has all the data we need.
    """
    final_response = None
    try:
        async for chunk in stream:
            chunk_dict = to_dict(chunk)
            # Check if this is the final completed event
            if chunk_dict.get("type") == "response.completed":
                final_response = chunk_dict.get("response")
            yield chunk
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        if final_response:
            _set_final_response_attributes(
                span, cast(_FinalResponse, cast(object, to_dict(final_response))), record_raw_response
            )
        else:
            span.end()

import json
from collections.abc import Iterable, Mapping
from typing import Any, cast

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.trace import Span

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.groq.event_models import Usage
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.groq.utils import (
    should_send_prompts,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    model_as_dict,
    set_span_attribute,
)

CONTENT_FILTER_KEY = "content_filter_results"


@dont_throw
def set_input_attributes(span: Span, kwargs: dict[str, Any]):  # pyright: ignore[reportExplicitAny]
    if not span.is_recording():
        return

    if should_send_prompts():
        if kwargs.get("prompt") is not None:
            set_span_attribute(span, "gen_ai.prompt.0.role", "user")
            set_span_attribute(span, "gen_ai.prompt.0.content", kwargs.get("prompt"))

        elif kwargs.get("messages") is not None and isinstance(kwargs.get("messages"), Iterable):
            for i, message in enumerate(cast(Iterable[dict[str, Any]], kwargs.get("messages"))):  # pyright: ignore[reportExplicitAny]
                set_span_attribute(
                    span,
                    f"gen_ai.prompt.{i}.content",
                    _dump_content(message.get("content")),
                )
                set_span_attribute(span, f"gen_ai.prompt.{i}.role", message.get("role"))


@dont_throw
def set_model_input_attributes(span: Span, kwargs: dict[str, Any]):  # pyright: ignore[reportExplicitAny]
    if not span.is_recording():
        return

    set_span_attribute(span, GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    set_span_attribute(
        span, GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample")
    )
    set_span_attribute(span, GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature"))
    set_span_attribute(span, GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(
        span, GEN_AI_REQUEST_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )
    set_span_attribute(
        span, GEN_AI_REQUEST_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    set_span_attribute(span, "llm.is_streaming", kwargs.get("stream") or False)


def set_streaming_response_attributes(
    span: Span,
    accumulated_content: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    finish_reason: str | None = None,
    usage: Any=None  # TODO: set usage on attributes
):
    """Set span attributes for accumulated streaming response."""
    if not span.is_recording() or not should_send_prompts():
        return

    prefix = "gen_ai.completion.0"
    set_span_attribute(span, f"{prefix}.role", "assistant")
    set_span_attribute(span, f"{prefix}.content", accumulated_content)
    if finish_reason:
        set_span_attribute(span, f"{prefix}.finish_reason", finish_reason)


def set_model_streaming_response_attributes(
    span: Span,
    usage: Usage | None,
):
    if not span.is_recording():
        return

    if usage:
        set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens)
        set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, usage.completion_tokens)
        set_span_attribute(span, "llm.usage.total_tokens", usage.total_tokens)


@dont_throw
def set_model_response_attributes(
    span: Span,
    response: Any,  # pyright: ignore[reportAny, reportExplicitAny]
):
    if not span.is_recording():
        return
    response = model_as_dict(response)
    set_span_attribute(span, GEN_AI_RESPONSE_MODEL, response.get("model"))  # pyright: ignore[reportAny]
    set_span_attribute(span, GEN_AI_RESPONSE_ID, response.get("id"))  # pyright: ignore[reportAny]

    usage = cast(dict[str, int | float], response.get("usage") or {})  # pyright: ignore[reportAny]
    if usage:
        set_span_attribute(span, "llm.usage.total_tokens", usage.get("total_tokens"))
        set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, usage.get("completion_tokens"))
        set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, usage.get("prompt_tokens"))


def set_response_attributes(span: Span, response: Any):  # pyright: ignore[reportAny, reportExplicitAny]
    if not span.is_recording():
        return
    choices = model_as_dict(response).get("choices")
    if should_send_prompts() and choices:
        _set_completions(span, choices)  # pyright: ignore[reportAny]


def _set_completions(span: Span, choices: list[dict[str, Any]] | None):  # pyright: ignore[reportExplicitAny]
    if choices is None or not should_send_prompts():
        return

    for choice in choices:
        index = choice.get("index")
        prefix = f"gen_ai.completion.{index}"
        set_span_attribute(span, f"{prefix}.finish_reason", choice.get("finish_reason"))

        if choice.get("content_filter_results"):
            set_span_attribute(
                span,
                f"{prefix}.{CONTENT_FILTER_KEY}",
                json.dumps(choice.get("content_filter_results")),
            )

        if choice.get("finish_reason") == "content_filter":
            set_span_attribute(span, f"{prefix}.role", "assistant")
            set_span_attribute(span, f"{prefix}.content", "FILTERED")

            return

        message = choice.get("message")
        if not message or not isinstance(message, Mapping):
            return

        set_span_attribute(span, f"{prefix}.role", message.get("role"))  # pyright: ignore[reportUnknownMemberType]
        set_span_attribute(span, f"{prefix}.content", message.get("content"))  # pyright: ignore[reportUnknownMemberType]

        function_call = message.get("function_call")  # pyright: ignore[reportUnknownMemberType]
        if function_call and isinstance(function_call, dict):
            set_span_attribute(
                span, f"{prefix}.tool_calls.0.name", function_call.get("name")  # pyright: ignore[reportUnknownMemberType]
            )
            set_span_attribute(
                span,
                f"{prefix}.tool_calls.0.arguments",
                function_call.get("arguments"),  # pyright: ignore[reportUnknownMemberType]
            )

        tool_calls = message.get("tool_calls")  # pyright: ignore[reportUnknownMemberType]
        if tool_calls and isinstance(tool_calls, Iterable):
            for i, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, Mapping):
                    continue
                function = tool_call.get("function")  # pyright: ignore[reportUnknownMemberType]
                if not isinstance(function, Mapping):
                    continue
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.id",
                    tool_call.get("id"),  # pyright: ignore[reportUnknownMemberType]
                )
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.name",
                    function.get("name"),  # pyright: ignore[reportUnknownMemberType]
                )
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.arguments",
                    function.get("arguments"),  # pyright: ignore[reportUnknownMemberType]
                )


def _dump_content(content: Any) -> str:  # pyright: ignore[reportAny, reportExplicitAny]
    if isinstance(content, str):
        return content
    json_serializable = []
    if not isinstance(content, Iterable):
        return ""
    for item in content:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "text":  # pyright: ignore[reportUnknownMemberType]
            json_serializable.append({"type": "text", "text": item.get("text")})  # pyright: ignore[reportUnknownMemberType]
        elif image_url := item.get("image_url"):  # pyright: ignore[reportUnknownMemberType]
            if not isinstance(image_url, Mapping):
                continue
            json_serializable.append(  # pyright: ignore[reportUnknownMemberType]
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url.get("url"),  # pyright: ignore[reportUnknownMemberType]
                        "detail": image_url.get("detail"),  # pyright: ignore[reportUnknownMemberType]
                    },
                }
            )
    return json.dumps(json_serializable)

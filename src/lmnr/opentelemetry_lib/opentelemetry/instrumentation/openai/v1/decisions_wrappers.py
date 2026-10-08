"""Instrumentation for the OpenAI Decisions API (`client.decisions.create`).

Decisions is a standalone, non-streaming endpoint (`POST /decisions`, added in
`openai==3.26.0`) that scores ordered predicate / choice / score questions
against shared user input. The questions define the answer contract, so they
go into `gen_ai.request.structured_output_schema` (same as the Jev / typesafe
instrumentation); `gen_ai.input.messages` holds only the user input, and the
typed answers are a single assistant message whose content is the JSON answer
list.
"""

from typing import Any

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Span, Tracer
from opentelemetry.trace.status import Status, StatusCode

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    safe_start_span,
)
from lmnr.opentelemetry_lib.tracing.context import get_event_attributes_from_context
from lmnr.sdk.utils import json_dumps
from openai._legacy_response import LegacyAPIResponse

from ..shared import model_as_dict, set_span_attribute
from ..utils import dont_throw, should_send_prompts, with_tracer_wrapper

SPAN_NAME = "openai.decision"


def build_genai_input_messages(kwargs: dict) -> list[dict[str, Any]] | None:
    """The shared user input as chat messages.

    Only materialized sequences are recorded (here and for `questions`): a
    generator is consumed by the SDK, so reading it here would either break
    the request (before the call) or see nothing (after it).
    """
    input_param = kwargs.get("input")
    if isinstance(input_param, str):
        return [{"role": "user", "content": input_param}]
    if isinstance(input_param, (list, tuple)):
        return [model_as_dict(message) for message in input_param] or None
    return None


def _parse_decision(response: Any) -> Any:
    """Return the parsed `Decision`, or None for shapes we can't read without
    side effects (e.g. `with_streaming_response`, whose body the caller owns)."""
    if isinstance(response, LegacyAPIResponse):
        return response.parse()
    if hasattr(response, "answers"):
        return response
    return None


@dont_throw
def _set_request_attributes(span: Span, kwargs: dict) -> None:
    set_span_attribute(span, GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    # Same key the chat path uses for its `user` param.
    set_span_attribute(span, "llm.user", kwargs.get("safety_identifier"))
    if should_send_prompts():
        input_messages = build_genai_input_messages(kwargs)
        if input_messages is not None:
            set_span_attribute(
                span, "gen_ai.input.messages", json_dumps(input_messages)
            )
        questions = kwargs.get("questions")
        if isinstance(questions, (list, tuple)):
            set_span_attribute(
                span,
                "gen_ai.request.structured_output_schema",
                json_dumps([model_as_dict(q) for q in questions]),
            )


@dont_throw
def _set_response_attributes(span: Span, response: Any) -> None:
    decision = _parse_decision(response)
    if decision is None:
        return

    set_span_attribute(span, GEN_AI_RESPONSE_MODEL, decision.model)
    if usage := decision.usage:
        set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)
        set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)
        set_span_attribute(span, "llm.usage.total_tokens", usage.total_tokens)
        if details := usage.input_tokens_details:
            set_span_attribute(
                span, "gen_ai.usage.cache_read_input_tokens", details.cached_tokens
            )
            set_span_attribute(
                span,
                "gen_ai.usage.cache_creation_input_tokens",
                details.cache_write_tokens,
            )
        if details := usage.output_tokens_details:
            set_span_attribute(
                span, "gen_ai.usage.reasoning_tokens", details.reasoning_tokens
            )

    if should_send_prompts():
        answers = [model_as_dict(answer) for answer in decision.answers]
        set_span_attribute(
            span,
            "gen_ai.output.messages",
            json_dumps([{"role": "assistant", "content": json_dumps(answers)}]),
        )


def _start_span(kwargs: dict) -> Span | None:
    span = safe_start_span(
        name=SPAN_NAME, attributes={"gen_ai.system": "openai"}, span_type="LLM"
    )
    if span is not None:
        _set_request_attributes(span, kwargs)
    return span


def _end_span_with_error(span: Span, e: Exception) -> None:
    span.set_attribute(ERROR_TYPE, e.__class__.__name__)
    span.record_exception(e, attributes=get_event_attributes_from_context())
    span.set_status(Status(StatusCode.ERROR, str(e)))
    span.end()


@with_tracer_wrapper
def decisions_create_wrapper(tracer: Tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = _start_span(kwargs)
    if span is None:
        return wrapped(*args, **kwargs)

    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        _end_span_with_error(span, e)
        raise

    _set_response_attributes(span, response)
    span.end()
    return response


@with_tracer_wrapper
async def async_decisions_create_wrapper(
    tracer: Tracer, wrapped, instance, args, kwargs
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = _start_span(kwargs)
    if span is None:
        return await wrapped(*args, **kwargs)

    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:
        _end_span_with_error(span, e)
        raise

    _set_response_attributes(span, response)
    span.end()
    return response

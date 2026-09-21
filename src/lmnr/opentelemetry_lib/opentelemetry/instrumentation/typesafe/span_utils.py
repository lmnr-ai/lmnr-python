import os

from opentelemetry.trace import Span

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    set_span_attribute,
    to_dict,
)
from lmnr.sdk.utils import json_dumps


def should_send_prompts() -> bool:
    return (os.getenv("LMNR_TRACE_CONTENT") or "true").lower() == "true"


def _default_model(instance) -> str | None:
    """Best-effort read of the client's default model (private attribute)."""
    try:
        return instance._config.default_model
    except Exception:
        return None


def _questions_to_dicts(questions) -> dict:
    """Questions are pydantic models (`Noul`/`Choice`/`Score`) or raw dicts."""
    return {
        name: question if isinstance(question, dict) else to_dict(question)
        for name, question in dict(questions).items()
    }


@dont_throw
def set_request_attributes(span: Span, call_kwargs: dict, instance):
    set_span_attribute(
        span,
        "gen_ai.request.model",
        call_kwargs.get("model") or _default_model(instance),
    )
    if not should_send_prompts():
        return
    state = call_kwargs.get("state")
    if state is not None:
        content = state if isinstance(state, str) else json_dumps(state)
        set_span_attribute(
            span,
            "gen_ai.input.messages",
            json_dumps([{"role": "user", "content": content}]),
        )
    questions = call_kwargs.get("questions")
    if questions:
        set_span_attribute(
            span,
            "gen_ai.request.structured_output_schema",
            json_dumps(_questions_to_dicts(questions)),
        )


@dont_throw
def set_response_attributes(span: Span, response: dict | None):
    if not response:
        return
    set_span_attribute(span, "gen_ai.response.model", response.get("model"))
    usage = response.get("usage") or {}
    set_span_attribute(span, "gen_ai.usage.input_tokens", usage.get("input_tokens"))
    set_span_attribute(span, "gen_ai.usage.output_tokens", usage.get("output_tokens"))
    # A custom `response_model` result may not carry `answers`.
    answers = response.get("answers")
    if answers and should_send_prompts():
        set_span_attribute(
            span,
            "gen_ai.output.messages",
            json_dumps([{"role": "assistant", "content": json_dumps(answers)}]),
        )

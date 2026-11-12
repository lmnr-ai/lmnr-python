"""Reasoning / thinking request config as span attributes.

The OpenAI-style effort string goes to the GenAI semconv's
`gen_ai.request.reasoning.level`. The semconv has nothing for the other knobs,
and every provider names and shapes them differently, so each one is recorded
under its own name rather than squeezed into a single normalized attribute.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any, cast

from opentelemetry.trace import Span

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    set_span_attribute,
)

# OpenAI / Groq / LiteLLM `reasoning_effort`, OpenAI Responses `reasoning.effort`
GEN_AI_REQUEST_REASONING_LEVEL = "gen_ai.request.reasoning.level"
# OpenAI Responses echoes the effort actually applied (a default when unset);
# named after the request-side `gen_ai.request.reasoning.level`.
GEN_AI_RESPONSE_REASONING_LEVEL = "gen_ai.response.reasoning.level"
# Anthropic `thinking.type` (enabled / adaptive / disabled / ...)
GEN_AI_REQUEST_THINKING_TYPE = "gen_ai.request.thinking_type"
# Anthropic `thinking.budget_tokens`, Gemini `thinking_config.thinking_budget`
GEN_AI_REQUEST_THINKING_BUDGET = "gen_ai.request.thinking_budget"
# Gemini `thinking_config.thinking_level`
GEN_AI_REQUEST_THINKING_LEVEL = "gen_ai.request.thinking_level"
# Anthropic `output_config.effort`
GEN_AI_REQUEST_EFFORT = "gen_ai.request.effort"
# Semconv reasoning usage, a subset of `gen_ai.usage.output_tokens`
GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"


def _get(obj: Any, key: str) -> Any:
    """`obj[key]` for mappings, `obj.key` otherwise; None on anything unexpected."""
    try:
        if obj is None:
            return None
        if isinstance(obj, Mapping):
            return cast(Mapping[str, Any], obj).get(key)
        return getattr(obj, key, None)
    except Exception:
        return None


def _as_str(value: Any) -> str | None:
    # str enums (google-genai `ThinkingLevel`) by value, int enums (proto) by name.
    # SDK sentinels (`NOT_GIVEN`, `omit`) and anything else are dropped.
    if isinstance(value, Enum):
        value = value.value if isinstance(value.value, str) else value.name
    return value if isinstance(value, str) and value else None


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def get_reasoning_effort(params: Any) -> str | None:
    """Flat `reasoning_effort` (chat completions) or nested `reasoning.effort` (Responses)."""
    return _first(
        _as_str(_get(params, "reasoning_effort")),
        _as_str(_get(_get(params, "reasoning"), "effort")),
    )


@dont_throw
def set_reasoning_attributes(span: Span, params: Any) -> None:
    """Record whichever known reasoning / thinking shapes appear in `params`.

    `params` is the request kwargs or a provider config object (mapping,
    pydantic model or plain object). Malformed or unexpected values are skipped,
    never raised.

    - OpenAI chat, Groq, LiteLLM: `reasoning_effort`
    - OpenAI Responses: `reasoning.effort`
    - Anthropic: `thinking.type`, `thinking.budget_tokens`, `output_config.effort`
    - Gemini (google-genai, Vertex AI): `thinking_config.thinking_budget` /
      `.thinking_level`; flat `thinking_budget` / `thinking_level` as LangChain's
      `ChatGoogleGenerativeAI` reports them
    """
    thinking = _get(params, "thinking")
    thinking_config = _get(params, "thinking_config")
    attributes: dict[str, str | int | None] = {
        GEN_AI_REQUEST_REASONING_LEVEL: get_reasoning_effort(params),
        GEN_AI_REQUEST_THINKING_TYPE: _as_str(_get(thinking, "type")),
        GEN_AI_REQUEST_THINKING_BUDGET: _first(
            _as_int(_get(thinking, "budget_tokens")),
            _as_int(_get(thinking_config, "thinking_budget")),
            _as_int(_get(params, "thinking_budget")),
        ),
        GEN_AI_REQUEST_THINKING_LEVEL: _first(
            _as_str(_get(thinking_config, "thinking_level")),
            _as_str(_get(params, "thinking_level")),
        ),
        GEN_AI_REQUEST_EFFORT: _as_str(_get(_get(params, "output_config"), "effort")),
    }
    for name, value in attributes.items():
        set_span_attribute(span, name, value)


@dont_throw
def set_reasoning_usage(span: Span, usage: Any) -> None:
    """Record the reasoning token count from an OpenAI-shaped `usage`.

    `usage` is a mapping, pydantic model or plain object with chat completions'
    `completion_tokens_details.reasoning_tokens` or the Responses API's
    `output_tokens_details.reasoning_tokens`. The count is already included in
    the output tokens. An explicit `0` is recorded; a missing or malformed count
    leaves the attribute unset.
    """
    set_span_attribute(
        span,
        GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
        _first(
            _as_int(_get(_get(usage, "completion_tokens_details"), "reasoning_tokens")),
            _as_int(_get(_get(usage, "output_tokens_details"), "reasoning_tokens")),
        ),
    )

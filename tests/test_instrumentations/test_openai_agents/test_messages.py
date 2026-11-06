"""Unit tests for the OpenAI Agents gen_ai attribute helpers.

Hermetic: `provider_from_model` is a pure function over the model name, and
`_apply_usage` only calls `set_attribute` on the span it is given.
"""

from typing import Any, cast

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.messages import (
    _apply_usage,
    provider_from_model,
)


class _AttributeSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


def test_provider_is_derived_from_a_litellm_model_string():
    """`LitellmModel` names are `<provider>/<model>`; the provider is not openai."""
    assert provider_from_model("anthropic/claude-sonnet-5") == "anthropic"
    assert provider_from_model("gemini/gemini-3-pro") == "gemini"
    # Nested routing prefixes keep the outermost provider - that is who serves
    # the request and whose pricing applies.
    assert provider_from_model("openrouter/anthropic/claude-sonnet-5") == "openrouter"


def test_provider_falls_back_to_openai_for_bare_model_names():
    """Models reached through the Responses API carry no provider prefix."""
    assert provider_from_model("gpt-5") == "openai"
    assert provider_from_model("/gpt-5") == "openai"
    assert provider_from_model(None) == "openai"


def test_usage_details_use_semconv_keys():
    span = _AttributeSpan()
    _apply_usage(
        cast(Any, span),
        {
            "input_tokens": 20,
            "output_tokens": 10,
            "input_tokens_details": {"cached_tokens": 5},
            "output_tokens_details": {"reasoning_tokens": 4},
        },
    )
    assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == 5
    assert span.attributes["gen_ai.usage.reasoning.output_tokens"] == 4

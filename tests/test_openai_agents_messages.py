"""Unit tests for the OpenAI Agents gen_ai attribute helpers.

Hermetic: `provider_from_model` is a pure function over the model name.
"""

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.messages import (
    provider_from_model,
)


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

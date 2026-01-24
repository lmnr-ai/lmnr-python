"""Tests for update_options_env_for_proxy function."""

import pytest

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent.wrappers import (
    update_options_env_for_proxy,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent.utils import (
    FOUNDRY_BASE_URL_ENV,
    FOUNDRY_RESOURCE_ENV,
)


class MockOptions:
    """Mock ClaudeAgentOptions for testing."""

    def __init__(self, env=None):
        self.env = env or {}


@pytest.fixture
def clean_env(monkeypatch):
    """Clean up environment variables."""
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_ORIGINAL_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USE_FOUNDRY", raising=False)


def test_basic_proxy_setup(clean_env):
    """Test basic proxy setup without foundry."""
    options = MockOptions()
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://api.anthropic.com"

    update_options_env_for_proxy(options, proxy_url, target_url)

    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    # ANTHROPIC_ORIGINAL_BASE_URL is now set to tell proxy where to forward
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url


def test_preserves_existing_env_keys(clean_env):
    """Test that existing keys in options.env are preserved."""
    options = MockOptions({"SOME_CUSTOM_KEY": "custom_value"})
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://api.anthropic.com"

    update_options_env_for_proxy(options, proxy_url, target_url)

    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    assert options.env["SOME_CUSTOM_KEY"] == "custom_value"


def test_foundry_in_options_env(clean_env):
    """Test foundry configuration in options.env (not in os.environ)."""
    options = MockOptions(
        {
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "ANTHROPIC_FOUNDRY_BASE_URL": "https://foundry.example.com",
        }
    )
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://foundry.example.com"

    update_options_env_for_proxy(options, proxy_url, target_url)

    # Should set ANTHROPIC_BASE_URL to proxy
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    # Should set FOUNDRY_BASE_URL to proxy
    assert options.env[FOUNDRY_BASE_URL_ENV] == proxy_url
    # Should keep CLAUDE_CODE_USE_FOUNDRY flag
    assert options.env["CLAUDE_CODE_USE_FOUNDRY"] == "1"
    # Should remove FOUNDRY_RESOURCE if present
    assert FOUNDRY_RESOURCE_ENV not in options.env
    # Should set ANTHROPIC_ORIGINAL_BASE_URL to target
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url


def test_foundry_in_system_env(monkeypatch, clean_env):
    """Test foundry configuration in os.environ (should be copied to options.env)."""
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://foundry.example.com")

    options = MockOptions()
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://foundry.example.com"

    update_options_env_for_proxy(options, proxy_url, target_url)

    # Should copy foundry config from os.environ into options.env
    assert options.env["CLAUDE_CODE_USE_FOUNDRY"] == "1"
    assert options.env[FOUNDRY_BASE_URL_ENV] == proxy_url
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url


def test_foundry_with_resource_in_options_env(clean_env):
    """Test foundry configuration with resource in options.env."""
    options = MockOptions(
        {"CLAUDE_CODE_USE_FOUNDRY": "1", "ANTHROPIC_FOUNDRY_RESOURCE": "my-resource"}
    )
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://my-resource.services.ai.azure.com/anthropic"

    update_options_env_for_proxy(options, proxy_url, target_url)

    # Should set ANTHROPIC_BASE_URL to proxy
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    # Should set FOUNDRY_BASE_URL to proxy
    assert options.env[FOUNDRY_BASE_URL_ENV] == proxy_url
    # Should remove FOUNDRY_RESOURCE (mutually exclusive with ANTHROPIC_BASE_URL)
    assert FOUNDRY_RESOURCE_ENV not in options.env
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url


def test_foundry_misconfigured_in_options_env(clean_env):
    """Test foundry enabled but misconfigured in options.env."""
    options = MockOptions(
        {
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            # No FOUNDRY_BASE_URL or FOUNDRY_RESOURCE
        }
    )
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://api.anthropic.com"  # Fallback

    update_options_env_for_proxy(options, proxy_url, target_url)

    # Should still set proxy URL
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    # Should set FOUNDRY_BASE_URL to proxy (even though misconfigured)
    assert options.env[FOUNDRY_BASE_URL_ENV] == proxy_url
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url


def test_anthropic_base_url_from_system_env_is_ignored(monkeypatch, clean_env):
    """Test that target_url parameter is used regardless of system env."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://custom.anthropic.com")

    options = MockOptions()
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://api.anthropic.com"

    update_options_env_for_proxy(options, proxy_url, target_url)

    # Should set new proxy URL
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    # Should set target URL from parameter
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url


def test_options_env_base_url_gets_overwritten(monkeypatch, clean_env):
    """Test that ANTHROPIC_BASE_URL in options.env gets overwritten with proxy URL."""
    options = MockOptions({"ANTHROPIC_BASE_URL": "https://options.anthropic.com"})
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://api.anthropic.com"

    update_options_env_for_proxy(options, proxy_url, target_url)

    # Should overwrite with proxy URL
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url


def test_updates_dict_in_place(clean_env):
    """Test that the function updates the dict in place, doesn't replace it."""
    original_dict = {"SOME_EXISTING_KEY": "existing_value"}
    options = MockOptions(original_dict)

    # Store reference to original dict
    dict_id = id(options.env)

    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://api.anthropic.com"
    update_options_env_for_proxy(options, proxy_url, target_url)

    # Should be same dict object
    assert id(options.env) == dict_id
    # Should preserve existing keys
    assert options.env["SOME_EXISTING_KEY"] == "existing_value"
    # Should add new keys
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url


def test_sets_anthropic_original_base_url_in_options_env(
    monkeypatch, clean_env
):
    """Test that ANTHROPIC_ORIGINAL_BASE_URL is set in options.env to target URL."""
    options = MockOptions()
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://api.anthropic.com"

    update_options_env_for_proxy(options, proxy_url, target_url)

    # Should set ANTHROPIC_BASE_URL
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    # Should set ANTHROPIC_ORIGINAL_BASE_URL to target
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url


def test_removes_foundry_resource_from_options_env(clean_env):
    """Test that FOUNDRY_RESOURCE is removed from options.env if present."""
    options = MockOptions({"ANTHROPIC_FOUNDRY_RESOURCE": "leftover-resource"})
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://api.anthropic.com"

    update_options_env_for_proxy(options, proxy_url, target_url)

    # Should remove FOUNDRY_RESOURCE from options.env
    assert FOUNDRY_RESOURCE_ENV not in options.env
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url


def test_foundry_resource_in_os_environ_is_handled_elsewhere(monkeypatch, clean_env):
    """Test that FOUNDRY_RESOURCE from os.environ is handled in wrap_transport_connect."""
    # Set in os.environ but not in options.env
    monkeypatch.setenv(FOUNDRY_RESOURCE_ENV, "my-resource")

    options = MockOptions()
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://api.anthropic.com"

    update_options_env_for_proxy(options, proxy_url, target_url)

    # Should NOT add FOUNDRY_RESOURCE to options.env
    # (it's handled by temporarily removing from os.environ in wrap_transport_connect)
    assert FOUNDRY_RESOURCE_ENV not in options.env
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url


def test_foundry_base_url_and_resource_are_mutually_exclusive(clean_env):
    """Test that FOUNDRY_RESOURCE is removed when setting FOUNDRY_BASE_URL."""
    options = MockOptions(
        {
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "ANTHROPIC_FOUNDRY_BASE_URL": "https://foundry.example.com",
            "ANTHROPIC_FOUNDRY_RESOURCE": "my-resource",
        }
    )
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://foundry.example.com"

    update_options_env_for_proxy(options, proxy_url, target_url)

    # Should set proxy URLs
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    assert options.env[FOUNDRY_BASE_URL_ENV] == proxy_url
    # Should remove FOUNDRY_RESOURCE (mutually exclusive)
    assert FOUNDRY_RESOURCE_ENV not in options.env

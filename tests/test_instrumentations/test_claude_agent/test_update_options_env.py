"""Tests for update_options_env_for_proxy function."""

import pytest

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent.wrappers import (
    update_options_env_for_proxy,
    snapshot_options_env_for_proxy,
    restore_options_env_from_snapshot,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent.utils import (
    FOUNDRY_BASE_URL_ENV,
    FOUNDRY_RESOURCE_ENV,
    resolve_target_url_from_env,
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


def test_sets_anthropic_original_base_url_in_options_env(monkeypatch, clean_env):
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


def test_http_proxy_removed_from_options_env(clean_env):
    """Test that HTTP_PROXY is removed from options.env."""
    options = MockOptions(
        {
            "HTTP_PROXY": "http://corporate-proxy.example.com:8080",
            "OTHER_VAR": "keep_me",
        }
    )
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://api.anthropic.com"

    update_options_env_for_proxy(options, proxy_url, target_url)

    # HTTP_PROXY should be removed
    assert "HTTP_PROXY" not in options.env
    # Other vars should be preserved
    assert options.env["OTHER_VAR"] == "keep_me"
    # Proxy config should be set
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url


def test_https_proxy_removed_from_options_env(clean_env):
    """Test that HTTPS_PROXY is removed from options.env."""
    options = MockOptions(
        {
            "HTTPS_PROXY": "https://corporate-proxy.example.com:8443",
            "OTHER_VAR": "keep_me",
        }
    )
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://api.anthropic.com"

    update_options_env_for_proxy(options, proxy_url, target_url)

    # HTTPS_PROXY should be removed
    assert "HTTPS_PROXY" not in options.env
    # Other vars should be preserved
    assert options.env["OTHER_VAR"] == "keep_me"
    # Proxy config should be set
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url


def test_both_proxy_vars_removed_from_options_env(clean_env):
    """Test that both HTTP_PROXY and HTTPS_PROXY are removed from options.env."""
    options = MockOptions(
        {
            "HTTP_PROXY": "http://proxy1.example.com",
            "HTTPS_PROXY": "https://proxy2.example.com",
            "ANTHROPIC_BASE_URL": "https://custom.anthropic.com",
        }
    )
    proxy_url = "http://127.0.0.1:45667"
    # In real scenario, target_url would be resolved from HTTPS_PROXY
    target_url = "https://proxy2.example.com"

    update_options_env_for_proxy(options, proxy_url, target_url)

    # Both proxy vars should be removed
    assert "HTTP_PROXY" not in options.env
    assert "HTTPS_PROXY" not in options.env
    # Proxy config should be set
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url


def test_https_proxy_resolution_from_options_env(monkeypatch, clean_env):
    """Test that HTTPS_PROXY from options.env is used for target URL resolution."""
    # Set conflicting values in os.environ
    monkeypatch.setenv("HTTP_PROXY", "http://system-proxy.example.com")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://system.anthropic.com")

    options = MockOptions(
        {
            "HTTPS_PROXY": "https://transport-proxy.example.com:8443",
        }
    )

    # Resolve target URL using options.env
    target_url = resolve_target_url_from_env(options.env)
    assert target_url == "https://transport-proxy.example.com:8443"

    # Update options.env
    proxy_url = "http://127.0.0.1:45667"
    update_options_env_for_proxy(options, proxy_url, target_url)

    # HTTPS_PROXY should be removed, target stored
    assert "HTTPS_PROXY" not in options.env
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url


def test_http_proxy_resolution_from_options_env(monkeypatch, clean_env):
    """Test that HTTP_PROXY from options.env is used for target URL resolution."""
    # Set conflicting values in os.environ
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://system.anthropic.com")

    options = MockOptions(
        {
            "HTTP_PROXY": "http://transport-proxy.example.com:8080",
        }
    )

    # Resolve target URL using options.env
    target_url = resolve_target_url_from_env(options.env)
    assert target_url == "http://transport-proxy.example.com:8080"

    # Update options.env
    proxy_url = "http://127.0.0.1:45667"
    update_options_env_for_proxy(options, proxy_url, target_url)

    # HTTP_PROXY should be removed, target stored
    assert "HTTP_PROXY" not in options.env
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url


def test_snapshot_and_restore_options_env(clean_env):
    """Test that options.env can be snapshotted and restored on error."""
    # Initial state with proxy vars
    options = MockOptions(
        {
            "HTTP_PROXY": "http://original-proxy.example.com:8080",
            "HTTPS_PROXY": "https://original-proxy.example.com:8443",
            "ANTHROPIC_BASE_URL": "https://custom.anthropic.com",
            "OTHER_VAR": "keep_me",
        }
    )

    # Snapshot before modification
    snapshot = snapshot_options_env_for_proxy(options)

    # Modify options.env
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://original-proxy.example.com:8443"
    update_options_env_for_proxy(options, proxy_url, target_url)

    # Verify modifications
    assert "HTTP_PROXY" not in options.env
    assert "HTTPS_PROXY" not in options.env
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == target_url
    assert options.env["OTHER_VAR"] == "keep_me"  # Unrelated vars preserved

    # Simulate error scenario: restore from snapshot
    restore_options_env_from_snapshot(options, snapshot)

    # Verify restoration
    assert options.env["HTTP_PROXY"] == "http://original-proxy.example.com:8080"
    assert options.env["HTTPS_PROXY"] == "https://original-proxy.example.com:8443"
    assert options.env["ANTHROPIC_BASE_URL"] == "https://custom.anthropic.com"
    assert "ANTHROPIC_ORIGINAL_BASE_URL" not in options.env
    assert options.env["OTHER_VAR"] == "keep_me"  # Still preserved


def test_snapshot_and_restore_with_foundry(monkeypatch, clean_env):
    """Test snapshot/restore with Foundry configuration."""
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")

    options = MockOptions(
        {
            "ANTHROPIC_FOUNDRY_RESOURCE": "my-resource",
            "HTTP_PROXY": "http://corporate-proxy.example.com",
        }
    )

    # Snapshot
    snapshot = snapshot_options_env_for_proxy(options)

    # Modify
    proxy_url = "http://127.0.0.1:45667"
    target_url = "https://my-resource.services.ai.azure.com/anthropic"
    update_options_env_for_proxy(options, proxy_url, target_url)

    # Verify Foundry resource removed, HTTP_PROXY removed
    assert FOUNDRY_RESOURCE_ENV not in options.env
    assert "HTTP_PROXY" not in options.env
    assert options.env[FOUNDRY_BASE_URL_ENV] == proxy_url

    # Restore
    restore_options_env_from_snapshot(options, snapshot)

    # Verify original state restored
    assert options.env["ANTHROPIC_FOUNDRY_RESOURCE"] == "my-resource"
    assert options.env["HTTP_PROXY"] == "http://corporate-proxy.example.com"
    assert FOUNDRY_BASE_URL_ENV not in options.env  # Was not present originally


def test_retry_after_failed_connect_uses_correct_target(clean_env):
    """
    Test that retrying after failed connection uses the correct target URL.

    This verifies the bug fix where HTTP_PROXY/HTTPS_PROXY removal from options.env
    would cause retries to use the wrong target.
    """
    # Initial state: user has HTTPS_PROXY set
    options = MockOptions(
        {
            "HTTPS_PROXY": "https://corporate-proxy.example.com:8443",
        }
    )

    # First attempt: snapshot, modify, and simulate failure
    snapshot = snapshot_options_env_for_proxy(options)
    target_url = resolve_target_url_from_env(options.env)
    assert target_url == "https://corporate-proxy.example.com:8443"

    proxy_url = "http://127.0.0.1:45667"
    update_options_env_for_proxy(options, proxy_url, target_url)

    # After update, HTTPS_PROXY is removed
    assert "HTTPS_PROXY" not in options.env
    assert options.env["ANTHROPIC_BASE_URL"] == proxy_url

    # Simulate connection failure - restore snapshot
    restore_options_env_from_snapshot(options, snapshot)

    # Retry: resolve target URL again (should work correctly now)
    retry_target_url = resolve_target_url_from_env(options.env)
    assert retry_target_url == "https://corporate-proxy.example.com:8443"

    # Without the fix, retry_target_url would be proxy_url (wrong!)
    assert retry_target_url != proxy_url

"""Tests for environment variable handling with per-transport proxies."""

import os
from unittest.mock import patch

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent import (
    proxy as claude_proxy,
    utils as claude_utils,
)
from lmnr_claude_code_proxy import ProxyServer


def test_foundry_base_url_overrides_target(monkeypatch):
    """Test that Foundry base URL is used as target."""
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv(
        "ANTHROPIC_FOUNDRY_BASE_URL", "https://foundry.example/anthropic/"
    )
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)

    proxy = ProxyServer(port=45500)
    proxy._allocated_port = 45500

    # Resolve target URL before starting proxy
    target_url = claude_utils.resolve_target_url_from_env({})

    with (
        patch.object(proxy, "run_server") as mock_run,
        patch.object(claude_proxy, "wait_for_port", return_value=True),
    ):
        result = claude_proxy.start_proxy(proxy, target_url)

        assert result == "http://127.0.0.1:45500"
        mock_run.assert_called_once_with("https://foundry.example/anthropic")

    # Clean up
    with patch.object(proxy, "stop_server"):
        claude_proxy.stop_proxy(proxy)


def test_foundry_resource_builds_target_url(monkeypatch):
    """Test that Foundry resource name builds correct URL."""
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "my-resource")

    proxy = ProxyServer(port=45501)
    proxy._allocated_port = 45501

    # Resolve target URL before starting proxy
    target_url = claude_utils.resolve_target_url_from_env({})

    with (
        patch.object(proxy, "run_server") as mock_run,
        patch.object(claude_proxy, "wait_for_port", return_value=True),
    ):
        result = claude_proxy.start_proxy(proxy, target_url)

        assert result == "http://127.0.0.1:45501"
        mock_run.assert_called_once_with(
            "https://my-resource.services.ai.azure.com/anthropic"
        )

    # Clean up
    with patch.object(proxy, "stop_server"):
        claude_proxy.stop_proxy(proxy)


def test_foundry_missing_config_fails(monkeypatch):
    """Test that missing Foundry config fails gracefully."""
    import pytest

    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)

    proxy = ProxyServer(port=45502)
    proxy._allocated_port = 45502

    # Resolve target URL - should return None for invalid config
    target_url = claude_utils.resolve_target_url_from_env({})
    assert target_url is None

    # start_proxy should fail when target_url is None
    with pytest.raises(RuntimeError):
        claude_proxy.start_proxy(proxy, target_url)

    # Port should be released on failure
    assert 45502 not in claude_proxy._ALLOCATED_PORTS


def test_env_restoration_after_stop(monkeypatch):
    """Test that environment is properly managed per-transport."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://original.anthropic.com")

    # Set up proxy environment
    snapshot = claude_utils.setup_proxy_env("http://127.0.0.1:45503")

    assert os.environ["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45503"
    assert os.environ["ANTHROPIC_ORIGINAL_BASE_URL"] == "https://original.anthropic.com"

    # Restore environment
    set_keys = {k for k, v in snapshot.items() if v is not None}
    claude_utils.restore_env(snapshot, set_keys)

    assert os.environ["ANTHROPIC_BASE_URL"] == "https://original.anthropic.com"
    assert "ANTHROPIC_ORIGINAL_BASE_URL" not in os.environ


def test_https_proxy_takes_highest_priority(monkeypatch):
    """Test that HTTPS_PROXY has highest priority in target URL resolution."""
    monkeypatch.setenv("HTTPS_PROXY", "https://corporate-proxy.example.com:8443")
    monkeypatch.setenv("HTTP_PROXY", "http://other-proxy.example.com:8080")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://custom.anthropic.com")
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://foundry.example.com")

    target_url = claude_utils.resolve_target_url_from_env({})
    assert target_url == "https://corporate-proxy.example.com:8443"


def test_http_proxy_priority_over_third_party(monkeypatch):
    """Test that HTTP_PROXY takes priority over third-party URLs."""
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://corporate-proxy.example.com:8080")
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://foundry.example.com")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://custom.anthropic.com")

    target_url = claude_utils.resolve_target_url_from_env({})
    assert target_url == "http://corporate-proxy.example.com:8080"


def test_proxy_vars_removed_from_options_env(monkeypatch):
    """Test that HTTP_PROXY and HTTPS_PROXY are removed from options.env."""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent.wrappers import (
        update_options_env_for_proxy,
    )

    class MockOptions:
        def __init__(self):
            self.env = {
                "HTTP_PROXY": "http://proxy1.example.com",
                "HTTPS_PROXY": "https://proxy2.example.com",
                "OTHER_VAR": "keep_me",
            }

    options = MockOptions()
    update_options_env_for_proxy(
        options, "http://127.0.0.1:45600", "https://api.anthropic.com"
    )

    # Proxy vars should be removed
    assert "HTTP_PROXY" not in options.env
    assert "HTTPS_PROXY" not in options.env
    # Other vars should be preserved
    assert options.env["OTHER_VAR"] == "keep_me"
    # Proxy config should be set
    assert options.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45600"
    assert options.env["ANTHROPIC_ORIGINAL_BASE_URL"] == "https://api.anthropic.com"


def test_proxy_vars_removed_from_global_env(monkeypatch):
    """Test that HTTP_PROXY and HTTPS_PROXY are removed from global env."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy1.example.com")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy2.example.com")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://custom.anthropic.com")

    snapshot = claude_utils.setup_proxy_env("http://127.0.0.1:45601")

    # Proxy vars should be removed from global env
    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ
    # Proxy config should be set
    assert os.environ["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45601"
    assert os.environ["ANTHROPIC_ORIGINAL_BASE_URL"] == "https://proxy2.example.com"

    # Restore environment
    set_keys = {k for k, v in snapshot.items() if v is not None}
    claude_utils.restore_env(snapshot, set_keys)

    # Original values should be restored
    assert os.environ["HTTP_PROXY"] == "http://proxy1.example.com"
    assert os.environ["HTTPS_PROXY"] == "https://proxy2.example.com"
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://custom.anthropic.com"
    assert "ANTHROPIC_ORIGINAL_BASE_URL" not in os.environ


def test_https_proxy_from_options_env(monkeypatch):
    """Test that HTTPS_PROXY from options.env takes priority."""
    # Set some conflicting values in os.environ
    monkeypatch.setenv("HTTP_PROXY", "http://system-proxy.example.com")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://system.anthropic.com")

    # options.env should take precedence
    env_dict = {
        "HTTPS_PROXY": "https://transport-proxy.example.com:8443",
        "ANTHROPIC_BASE_URL": "https://transport.anthropic.com",
    }

    target_url = claude_utils.resolve_target_url_from_env(env_dict)
    assert target_url == "https://transport-proxy.example.com:8443"


def test_http_proxy_strips_trailing_slash(monkeypatch):
    """Test that trailing slashes are removed from HTTP_PROXY."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.com:8080/")

    target_url = claude_utils.resolve_target_url_from_env({})
    assert target_url == "http://proxy.example.com:8080"


def test_resolution_order_complete(monkeypatch):
    """Test complete priority order: HTTPS_PROXY > HTTP_PROXY > Foundry > ANTHROPIC_BASE_URL > default."""
    
    # Test 1: Only default
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USE_FOUNDRY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    
    target_url = claude_utils.resolve_target_url_from_env({})
    assert target_url == "https://api.anthropic.com"
    
    # Test 2: ANTHROPIC_BASE_URL overrides default
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://custom.anthropic.com")
    target_url = claude_utils.resolve_target_url_from_env({})
    assert target_url == "https://custom.anthropic.com"
    
    # Test 3: Foundry overrides ANTHROPIC_BASE_URL
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://foundry.example.com")
    target_url = claude_utils.resolve_target_url_from_env({})
    assert target_url == "https://foundry.example.com"
    
    # Test 4: HTTP_PROXY overrides Foundry
    monkeypatch.setenv("HTTP_PROXY", "http://http-proxy.example.com")
    target_url = claude_utils.resolve_target_url_from_env({})
    assert target_url == "http://http-proxy.example.com"
    
    # Test 5: HTTPS_PROXY overrides everything
    monkeypatch.setenv("HTTPS_PROXY", "https://https-proxy.example.com")
    target_url = claude_utils.resolve_target_url_from_env({})
    assert target_url == "https://https-proxy.example.com"

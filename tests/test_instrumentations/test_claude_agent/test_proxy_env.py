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

    with (
        patch.object(proxy, "run_server") as mock_run,
        patch.object(claude_proxy, "wait_for_port", return_value=True),
    ):
        result = claude_proxy.start_proxy(proxy)

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

    with (
        patch.object(proxy, "run_server") as mock_run,
        patch.object(claude_proxy, "wait_for_port", return_value=True),
    ):
        result = claude_proxy.start_proxy(proxy)

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

    with pytest.raises(RuntimeError, match="Invalid provider configuration"):
        claude_proxy.start_proxy(proxy)

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

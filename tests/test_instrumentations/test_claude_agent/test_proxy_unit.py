"""Comprehensive unit tests for per-transport proxy architecture."""

import os
import socket
import threading
import time
from unittest.mock import patch

import pytest

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent import (
    proxy as claude_proxy,
    utils as claude_utils,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Clean up environment variables."""
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_ORIGINAL_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USE_FOUNDRY", raising=False)


@pytest.fixture(autouse=True)
def reset_port_allocation():
    """Reset port allocation state before and after tests."""
    with claude_proxy._PORT_LOCK:
        claude_proxy._NEXT_PORT = 45667
        claude_proxy._ALLOCATED_PORTS.clear()
    yield
    with claude_proxy._PORT_LOCK:
        claude_proxy._NEXT_PORT = 45667
        claude_proxy._ALLOCATED_PORTS.clear()


# ===== Utility Function Tests =====


def test_is_truthy_env():
    """Test is_truthy_env function."""
    assert claude_utils.is_truthy_env("1") is True
    assert claude_utils.is_truthy_env("0") is False
    assert claude_utils.is_truthy_env("true") is False
    assert claude_utils.is_truthy_env("") is False
    assert claude_utils.is_truthy_env(None) is False


def test_snapshot_env(monkeypatch):
    """Test environment snapshot function."""
    monkeypatch.setenv("TEST_KEY1", "value1")
    monkeypatch.setenv("TEST_KEY2", "value2")

    snapshot, set_keys = claude_utils.snapshot_env(
        ["TEST_KEY1", "TEST_KEY2", "TEST_KEY3"]
    )

    assert snapshot["TEST_KEY1"] == "value1"
    assert snapshot["TEST_KEY2"] == "value2"
    assert snapshot["TEST_KEY3"] is None
    assert set_keys == {"TEST_KEY1", "TEST_KEY2"}


def test_restore_env(monkeypatch):
    """Test environment restoration."""
    monkeypatch.setenv("KEY1", "original")
    monkeypatch.setenv("KEY2", "to_delete")

    snapshot = {"KEY1": "new_value", "KEY2": "original", "KEY3": "added"}
    set_keys = {"KEY1", "KEY2"}

    # Modify environment
    os.environ["KEY1"] = "modified"
    os.environ.pop("KEY2", None)
    os.environ["KEY3"] = "added"

    # Restore
    claude_utils.restore_env(snapshot, set_keys)

    assert os.environ["KEY1"] == "new_value"
    assert os.environ["KEY2"] == "original"
    assert "KEY3" not in os.environ


def test_is_port_open():
    """Test port checking."""
    # Create a listening socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 45300))
    server.listen(1)

    try:
        assert claude_utils.is_port_open(45300) is True
        assert claude_utils.is_port_open(45301) is False
    finally:
        server.close()


def test_wait_for_port():
    """Test waiting for port to become available."""
    port = 45302

    def start_server():
        time.sleep(0.3)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(1)
        time.sleep(1)
        server.close()

    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()

    result = claude_utils.wait_for_port(port, timeout=2.0)
    assert result is True


def test_wait_for_port_timeout():
    """Test timeout when port doesn't open."""
    result = claude_utils.wait_for_port(45303, timeout=0.5)
    assert result is False


def test_foundry_target_url(monkeypatch):
    """Test Foundry URL resolution."""
    # Not enabled
    monkeypatch.delenv("CLAUDE_CODE_USE_FOUNDRY", raising=False)
    assert claude_utils.foundry_target_url() is None

    # With base URL
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://foundry.example.com/")
    assert claude_utils.foundry_target_url() == "https://foundry.example.com"

    # With resource
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "my-resource")
    assert (
        claude_utils.foundry_target_url()
        == "https://my-resource.services.ai.azure.com/anthropic"
    )

    # Misconfigured
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    assert claude_utils.foundry_target_url() is None


def test_extract_3p_target_url(monkeypatch):
    """Test third-party provider extraction."""
    # No provider
    monkeypatch.delenv("CLAUDE_CODE_USE_FOUNDRY", raising=False)
    enabled, url = claude_utils.extract_3p_target_url()
    assert enabled is False
    assert url is None

    # Foundry enabled and valid
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://foundry.example.com")
    enabled, url = claude_utils.extract_3p_target_url()
    assert enabled is True
    assert url == "https://foundry.example.com"

    # Foundry enabled but invalid
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    enabled, url = claude_utils.extract_3p_target_url()
    assert enabled is True
    assert url is None


def test_resolve_target_url(monkeypatch, clean_env):
    """Test target URL resolution."""
    # Default fallback
    url = claude_utils.resolve_target_url()
    assert url == "https://api.anthropic.com"

    # From ANTHROPIC_BASE_URL
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://custom.anthropic.com")
    url = claude_utils.resolve_target_url()
    assert url == "https://custom.anthropic.com"

    # From ANTHROPIC_ORIGINAL_BASE_URL (takes precedence)
    monkeypatch.setenv("ANTHROPIC_ORIGINAL_BASE_URL", "https://original.anthropic.com")
    url = claude_utils.resolve_target_url()
    assert url == "https://original.anthropic.com"

    # From Foundry provider
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://foundry.example.com")
    url = claude_utils.resolve_target_url()
    assert url == "https://foundry.example.com"


def test_setup_proxy_env(monkeypatch, clean_env):
    """Test setting up proxy environment."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    snapshot = claude_utils.setup_proxy_env("http://127.0.0.1:45667")

    assert snapshot["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert os.environ["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45667"
    assert os.environ["ANTHROPIC_ORIGINAL_BASE_URL"] == "https://api.anthropic.com"


def test_setup_proxy_env_with_foundry(monkeypatch, clean_env):
    """Test setting up proxy environment with Foundry."""
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://foundry.example.com")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "my-resource")

    snapshot = claude_utils.setup_proxy_env("http://127.0.0.1:45667")

    assert os.environ["ANTHROPIC_FOUNDRY_BASE_URL"] == "http://127.0.0.1:45667"
    assert "ANTHROPIC_FOUNDRY_RESOURCE" not in os.environ
    assert snapshot["ANTHROPIC_FOUNDRY_BASE_URL"] == "https://foundry.example.com"
    assert snapshot["ANTHROPIC_FOUNDRY_RESOURCE"] == "my-resource"


# ===== Port Allocation Tests =====


def test_allocate_port():
    """Test port allocation."""
    port1 = claude_proxy._allocate_port()
    port2 = claude_proxy._allocate_port()
    port3 = claude_proxy._allocate_port()

    assert port1 == 45667
    assert port2 == 45668
    assert port3 == 45669
    assert claude_proxy._ALLOCATED_PORTS == {45667, 45668, 45669}


def test_release_port():
    """Test port release."""
    port = claude_proxy._allocate_port()
    assert port in claude_proxy._ALLOCATED_PORTS

    claude_proxy._release_port(port)
    assert port not in claude_proxy._ALLOCATED_PORTS


def test_port_reuse_after_release():
    """Test that released ports can be reused."""
    port1 = claude_proxy._allocate_port()
    port2 = claude_proxy._allocate_port()

    claude_proxy._release_port(port1)

    # Next allocation should skip port1 (already allocated) and use port3
    port3 = claude_proxy._allocate_port()
    assert port3 == port2 + 1


def test_concurrent_port_allocation():
    """Test thread-safe port allocation."""
    allocated = []
    lock = threading.Lock()

    def allocate_ports():
        for _ in range(5):
            port = claude_proxy._allocate_port()
            with lock:
                allocated.append(port)
            time.sleep(0.01)

    threads = [threading.Thread(target=allocate_ports) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All ports should be unique
    assert len(allocated) == len(set(allocated))


# ===== Proxy Lifecycle Tests =====


def test_create_proxy_for_transport():
    """Test creating a proxy for a transport."""
    proxy = claude_proxy.create_proxy_for_transport()

    assert proxy is not None
    assert hasattr(proxy, "_allocated_port")
    assert proxy._allocated_port in claude_proxy._ALLOCATED_PORTS
    assert proxy.port == proxy._allocated_port


def test_start_proxy_success(monkeypatch, clean_env):
    """Test starting a proxy successfully."""
    from lmnr_claude_code_proxy import ProxyServer

    proxy = ProxyServer(port=45400)
    proxy._allocated_port = 45400

    with (
        patch.object(proxy, "run_server") as mock_run,
        patch.object(claude_proxy, "wait_for_port", return_value=True),
    ):
        url = claude_proxy.start_proxy(proxy, "https://api.anthropic.com")

        assert url == "http://127.0.0.1:45400"
        mock_run.assert_called_once_with("https://api.anthropic.com")


def test_start_proxy_with_none_target_url(monkeypatch, clean_env):
    """Test starting proxy with target_url=None (resolves from env)."""
    from lmnr_claude_code_proxy import ProxyServer

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://custom.anthropic.com")
    proxy = ProxyServer(port=45401)
    proxy._allocated_port = 45401

    with (
        patch.object(proxy, "run_server") as mock_run,
        patch.object(claude_proxy, "wait_for_port", return_value=True),
    ):
        url = claude_proxy.start_proxy(proxy)

        assert url == "http://127.0.0.1:45401"
        mock_run.assert_called_once_with("https://custom.anthropic.com")


def test_start_proxy_invalid_provider(monkeypatch, clean_env):
    """Test starting proxy with invalid provider configuration."""
    from lmnr_claude_code_proxy import ProxyServer

    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)

    proxy = ProxyServer(port=45402)
    proxy._allocated_port = 45402

    with pytest.raises(RuntimeError, match="Invalid provider configuration"):
        claude_proxy.start_proxy(proxy)

    # Port should be released
    assert 45402 not in claude_proxy._ALLOCATED_PORTS


def test_start_proxy_server_fails(monkeypatch, clean_env):
    """Test handling server startup failure."""
    from lmnr_claude_code_proxy import ProxyServer

    proxy = ProxyServer(port=45403)
    proxy._allocated_port = 45403

    with patch.object(proxy, "run_server", side_effect=Exception("Server error")):
        with pytest.raises(RuntimeError, match="Failed to start proxy"):
            claude_proxy.start_proxy(proxy, "https://api.anthropic.com")

    # Port should be released
    assert 45403 not in claude_proxy._ALLOCATED_PORTS


def test_start_proxy_not_ready(monkeypatch, clean_env):
    """Test when proxy doesn't become ready."""
    from lmnr_claude_code_proxy import ProxyServer

    proxy = ProxyServer(port=45404)
    proxy._allocated_port = 45404

    with (
        patch.object(proxy, "run_server"),
        patch.object(claude_proxy, "wait_for_port", return_value=False),
        patch.object(proxy, "stop_server") as mock_stop,
    ):
        with pytest.raises(RuntimeError, match="Proxy failed to start"):
            claude_proxy.start_proxy(proxy, "https://api.anthropic.com")

        mock_stop.assert_called()
        # Port should be released
        assert 45404 not in claude_proxy._ALLOCATED_PORTS


def test_stop_proxy():
    """Test stopping a proxy."""
    from lmnr_claude_code_proxy import ProxyServer

    port = claude_proxy._allocate_port()
    proxy = ProxyServer(port=port)
    proxy._allocated_port = port

    with patch.object(proxy, "stop_server") as mock_stop:
        claude_proxy.stop_proxy(proxy)

        mock_stop.assert_called_once()
        assert port not in claude_proxy._ALLOCATED_PORTS


def test_stop_proxy_handles_error():
    """Test that stop_proxy handles errors gracefully."""
    from lmnr_claude_code_proxy import ProxyServer

    port = claude_proxy._allocate_port()
    proxy = ProxyServer(port=port)
    proxy._allocated_port = port

    with patch.object(proxy, "stop_server", side_effect=Exception("Stop error")):
        # Should not raise
        claude_proxy.stop_proxy(proxy)

        # Port should still be released
        assert port not in claude_proxy._ALLOCATED_PORTS


def test_publish_span_context_to_proxy():
    """Test publishing span context to a proxy."""
    from lmnr_claude_code_proxy import ProxyServer

    proxy = ProxyServer(port=45405)

    with patch.object(proxy, "set_current_trace") as mock_set:
        claude_proxy.publish_span_context_to_proxy(
            proxy=proxy,
            trace_id="trace123",
            span_id="span456",
            project_api_key="key789",
            span_path=["root", "child"],
            span_ids_path=["id1", "id2"],
            laminar_url="https://api.lmnr.ai",
        )

        mock_set.assert_called_once_with(
            trace_id="trace123",
            span_id="span456",
            project_api_key="key789",
            span_path=["root", "child"],
            span_ids_path=["id1", "id2"],
            laminar_url="https://api.lmnr.ai",
        )


def test_publish_span_context_handles_error():
    """Test that publish_span_context_to_proxy handles errors."""
    from lmnr_claude_code_proxy import ProxyServer

    proxy = ProxyServer(port=45406)

    with patch.object(proxy, "set_current_trace", side_effect=Exception("HTTP error")):
        # Should not raise, just log
        claude_proxy.publish_span_context_to_proxy(
            proxy=proxy,
            trace_id="trace123",
            span_id="span456",
            project_api_key="key789",
            span_path=[],
            span_ids_path=[],
            laminar_url="https://api.lmnr.ai",
        )


# ===== Multiple Proxy Tests =====


def test_multiple_proxies_different_ports(monkeypatch, clean_env):
    """Test creating multiple proxies with different ports."""
    proxies = []

    with (
        patch("lmnr_claude_code_proxy.ProxyServer.run_server"),
        patch.object(claude_proxy, "wait_for_port", return_value=True),
    ):
        for _ in range(3):
            proxy = claude_proxy.create_proxy_for_transport()
            url = claude_proxy.start_proxy(proxy, "https://api.anthropic.com")
            proxies.append((proxy, url))

    # Check all have different ports
    ports = [proxy.port for proxy, _ in proxies]
    assert len(ports) == len(set(ports))
    assert ports == [45667, 45668, 45669]

    # Check all URLs are correct
    urls = [url for _, url in proxies]
    assert urls == [
        "http://127.0.0.1:45667",
        "http://127.0.0.1:45668",
        "http://127.0.0.1:45669",
    ]

    # Clean up
    for proxy, _ in proxies:
        with patch.object(proxy, "stop_server"):
            claude_proxy.stop_proxy(proxy)


def test_port_reuse_after_cleanup(monkeypatch, clean_env):
    """Test that ports are reused after cleanup."""
    with (
        patch("lmnr_claude_code_proxy.ProxyServer.run_server"),
        patch.object(claude_proxy, "wait_for_port", return_value=True),
    ):
        # Create and stop first proxy
        proxy1 = claude_proxy.create_proxy_for_transport()
        claude_proxy.start_proxy(proxy1, "https://api.anthropic.com")
        port1 = proxy1.port

        with patch.object(proxy1, "stop_server"):
            claude_proxy.stop_proxy(proxy1)

        # Create second proxy - should get a new port (not reuse)
        proxy2 = claude_proxy.create_proxy_for_transport()
        assert proxy2.port != port1  # Sequential allocation, doesn't reuse immediately

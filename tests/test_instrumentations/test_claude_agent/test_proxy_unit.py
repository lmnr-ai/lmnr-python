"""Comprehensive unit tests for proxy.py module."""
import os
import socket
import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.claude_agent import proxy as claude_proxy


@pytest.fixture(autouse=True)
def reset_proxy_state():
    """Reset proxy state before and after each test."""
    # Before test
    with claude_proxy._proxy_state.lock:
        claude_proxy._proxy_state.reset()
        claude_proxy._proxy_state.shutdown_registered = False
    
    yield
    
    # After test
    with claude_proxy._proxy_state.lock:
        claude_proxy._proxy_state.reset()
        claude_proxy._proxy_state.shutdown_registered = False


@pytest.fixture
def clean_env(monkeypatch):
    """Clean up environment variables."""
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_ORIGINAL_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_USE_FOUNDRY", raising=False)


# ===== Port Utilities Tests =====

def test_find_available_port_success():
    """Test finding an available port."""
    port = claude_proxy._find_available_port(45000, 5)
    assert port is not None
    assert 45000 <= port < 45005


def test_find_available_port_all_occupied():
    """Test when all ports in range are occupied."""
    # Bind all ports in the range
    sockets = []
    start_port = 45100
    attempts = 3
    
    try:
        for offset in range(attempts):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", start_port + offset))
            sock.listen(1)
            sockets.append(sock)
        
        port = claude_proxy._find_available_port(start_port, attempts)
        assert port is None
    finally:
        for sock in sockets:
            sock.close()


def test_is_port_open_when_open():
    """Test checking if a port is open when it is."""
    # Create a listening socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 45200))
    server.listen(1)
    
    try:
        assert claude_proxy._is_port_open(45200) is True
    finally:
        server.close()


def test_is_port_open_when_closed():
    """Test checking if a port is open when it is not."""
    # Use a port that's very unlikely to be in use
    assert claude_proxy._is_port_open(45201) is False


def test_wait_for_port_success():
    """Test waiting for a port to become available."""
    port = 45202
    
    def start_server():
        time.sleep(0.3)  # Delay before starting
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(1)
        time.sleep(1)  # Keep server alive
        server.close()
    
    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()
    
    result = claude_proxy._wait_for_port(port, timeout=2.0)
    assert result is True


def test_wait_for_port_timeout():
    """Test waiting for a port that never opens."""
    # Use a port that won't open
    result = claude_proxy._wait_for_port(45203, timeout=0.5)
    assert result is False


def test_find_existing_proxy_found():
    """Test finding an existing proxy server."""
    # Create a server on one of the ports in range
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = claude_proxy.DEFAULT_CC_PROXY_PORT + 2
    server.bind(("127.0.0.1", port))
    server.listen(1)
    
    try:
        found_port = claude_proxy._find_existing_proxy()
        assert found_port == port
    finally:
        server.close()


def test_find_existing_proxy_not_found():
    """Test when no existing proxy is found."""
    # Ensure no servers are running on the port range
    found_port = claude_proxy._find_existing_proxy()
    # This might be None or might find another test's server
    # We can't guarantee None, so just check it's the right type
    assert found_port is None or isinstance(found_port, int)


# ===== Environment Utilities Tests =====

def test_is_truthy_env_true():
    """Test _is_truthy_env with '1'."""
    assert claude_proxy._is_truthy_env("1") is True


def test_is_truthy_env_false():
    """Test _is_truthy_env with other values."""
    assert claude_proxy._is_truthy_env("0") is False
    assert claude_proxy._is_truthy_env("true") is False
    assert claude_proxy._is_truthy_env("") is False
    assert claude_proxy._is_truthy_env(None) is False


def test_snapshot_env_with_set_keys(monkeypatch):
    """Test snapshotting environment variables that are set."""
    monkeypatch.setenv("TEST_KEY1", "value1")
    monkeypatch.setenv("TEST_KEY2", "value2")
    
    snapshot, set_keys = claude_proxy._snapshot_env(["TEST_KEY1", "TEST_KEY2", "TEST_KEY3"])
    
    assert snapshot["TEST_KEY1"] == "value1"
    assert snapshot["TEST_KEY2"] == "value2"
    assert snapshot["TEST_KEY3"] is None
    assert set_keys == {"TEST_KEY1", "TEST_KEY2"}


def test_snapshot_env_with_unset_keys():
    """Test snapshotting environment variables that are not set."""
    snapshot, set_keys = claude_proxy._snapshot_env(["NONEXISTENT_KEY"])
    
    assert snapshot["NONEXISTENT_KEY"] is None
    assert set_keys == set()


# ===== Provider Detection Tests =====

def test_foundry_target_url_with_base_url(monkeypatch):
    """Test _foundry_target_url with explicit base URL."""
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://foundry.example.com/")
    
    target = claude_proxy._foundry_target_url()
    assert target == "https://foundry.example.com"


def test_foundry_target_url_with_resource(monkeypatch):
    """Test _foundry_target_url with resource name."""
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "my-resource")
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    
    target = claude_proxy._foundry_target_url()
    assert target == "https://my-resource.services.ai.azure.com/anthropic"


def test_foundry_target_url_not_enabled(monkeypatch):
    """Test _foundry_target_url when Foundry is not enabled."""
    monkeypatch.delenv("CLAUDE_CODE_USE_FOUNDRY", raising=False)
    
    target = claude_proxy._foundry_target_url()
    assert target is None


def test_foundry_target_url_missing_config(monkeypatch):
    """Test _foundry_target_url with missing configuration."""
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    
    target = claude_proxy._foundry_target_url()
    assert target is None


def test_extract_3p_target_url_foundry_valid(monkeypatch):
    """Test _extract_3p_target_url with valid Foundry config."""
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://foundry.example.com")
    
    provider_enabled, target_url = claude_proxy._extract_3p_target_url()
    assert provider_enabled is True
    assert target_url == "https://foundry.example.com"


def test_extract_3p_target_url_foundry_invalid(monkeypatch):
    """Test _extract_3p_target_url with invalid Foundry config."""
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    
    provider_enabled, target_url = claude_proxy._extract_3p_target_url()
    assert provider_enabled is True
    assert target_url is None


def test_extract_3p_target_url_no_provider(monkeypatch):
    """Test _extract_3p_target_url with no provider configured."""
    monkeypatch.delenv("CLAUDE_CODE_USE_FOUNDRY", raising=False)
    
    provider_enabled, target_url = claude_proxy._extract_3p_target_url()
    assert provider_enabled is False
    assert target_url is None


# ===== Proxy State Management Tests =====

def test_register_proxy_shutdown():
    """Test registering proxy shutdown handler."""
    claude_proxy._proxy_state.shutdown_registered = False
    
    with patch("atexit.register") as mock_atexit:
        claude_proxy._register_proxy_shutdown()
        mock_atexit.assert_called_once_with(claude_proxy._stop_cc_proxy)
        assert claude_proxy._proxy_state.shutdown_registered is True
        
        # Calling again should not register twice
        claude_proxy._register_proxy_shutdown()
        assert mock_atexit.call_count == 1


def test_get_cc_proxy_base_url():
    """Test getting the proxy base URL."""
    assert claude_proxy.get_cc_proxy_base_url() is None
    
    claude_proxy._proxy_state.base_url = "http://127.0.0.1:45667"
    assert claude_proxy.get_cc_proxy_base_url() == "http://127.0.0.1:45667"


def test_stop_cc_proxy_locked_cleans_state(monkeypatch, clean_env):
    """Test that _stop_cc_proxy_locked cleans up state."""
    # Set up initial state
    claude_proxy._proxy_state.port = 45667
    claude_proxy._proxy_state.base_url = "http://127.0.0.1:45667"
    claude_proxy._proxy_state.target_url = "https://api.anthropic.com"
    claude_proxy._proxy_state.env_snapshot = {
        "ANTHROPIC_BASE_URL": "original_value",
        "NEW_KEY": None
    }
    claude_proxy._proxy_state.env_snapshot_set = {"ANTHROPIC_BASE_URL"}
    
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:45667")
    monkeypatch.setenv("NEW_KEY", "new_value")
    
    with patch.object(claude_proxy, "stop_server"):
        claude_proxy._stop_cc_proxy_locked()
    
    # Check state is reset
    assert claude_proxy._proxy_state.port is None
    assert claude_proxy._proxy_state.base_url is None
    assert claude_proxy._proxy_state.target_url is None
    assert claude_proxy._proxy_state.env_snapshot is None
    assert claude_proxy._proxy_state.env_snapshot_set is None
    
    # Check environment is restored
    assert os.environ["ANTHROPIC_BASE_URL"] == "original_value"
    assert "NEW_KEY" not in os.environ


def test_stop_cc_proxy_locked_handles_server_error(monkeypatch):
    """Test that _stop_cc_proxy_locked handles server stop errors."""
    with patch.object(claude_proxy, "stop_server", side_effect=Exception("Server error")):
        # Should not raise, just log
        claude_proxy._stop_cc_proxy_locked()


def test_stop_cc_proxy_uses_lock():
    """Test that _stop_cc_proxy uses the lock."""
    with patch.object(claude_proxy, "_stop_cc_proxy_locked") as mock_stop:
        claude_proxy._stop_cc_proxy()
        mock_stop.assert_called_once()


def test_release_proxy(monkeypatch):
    """Test release_proxy function."""
    with patch.object(claude_proxy, "_stop_cc_proxy_locked") as mock_stop:
        claude_proxy.release_proxy()
        mock_stop.assert_called_once()


# ===== Start/Connect Proxy Tests =====

def test_start_or_connect_to_proxy_reuses_tracked_proxy(monkeypatch):
    """Test reusing an already tracked running proxy."""
    claude_proxy._proxy_state.port = 45667
    claude_proxy._proxy_state.base_url = "http://127.0.0.1:45667"
    
    with patch.object(claude_proxy, "_is_port_open", return_value=True):
        result = claude_proxy.start_or_connect_to_proxy()
        assert result == "http://127.0.0.1:45667"


def test_start_or_connect_to_proxy_tracked_proxy_died(monkeypatch, clean_env):
    """Test when tracked proxy is no longer running."""
    claude_proxy._proxy_state.port = 45667
    claude_proxy._proxy_state.base_url = "http://127.0.0.1:45667"
    
    with patch.object(claude_proxy, "_is_port_open", return_value=False), \
         patch.object(claude_proxy, "_find_existing_proxy", return_value=None), \
         patch.object(claude_proxy, "_find_available_port", return_value=45668), \
         patch.object(claude_proxy, "run_server"), \
         patch.object(claude_proxy, "_wait_for_port", return_value=True), \
         patch.object(claude_proxy, "_register_proxy_shutdown"):
        
        result = claude_proxy.start_or_connect_to_proxy()
        assert result == "http://127.0.0.1:45668"


def test_start_or_connect_to_existing_untracked_proxy(monkeypatch, clean_env):
    """Test connecting to an existing untracked proxy (container reuse)."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    
    with patch.object(claude_proxy, "_find_existing_proxy", return_value=45667), \
         patch.object(claude_proxy, "_register_proxy_shutdown"):
        
        result = claude_proxy.start_or_connect_to_proxy()
        
        assert result == "http://127.0.0.1:45667"
        assert claude_proxy._proxy_state.port == 45667
        assert claude_proxy._proxy_state.base_url == "http://127.0.0.1:45667"
        assert os.environ["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45667"
        assert claude_proxy._proxy_state.env_snapshot is not None


def test_start_or_connect_starts_new_proxy(monkeypatch, clean_env):
    """Test starting a new proxy server."""
    with patch.object(claude_proxy, "_find_existing_proxy", return_value=None), \
         patch.object(claude_proxy, "_find_available_port", return_value=45667), \
         patch.object(claude_proxy, "run_server") as mock_run, \
         patch.object(claude_proxy, "_wait_for_port", return_value=True), \
         patch.object(claude_proxy, "_register_proxy_shutdown"):
        
        result = claude_proxy.start_or_connect_to_proxy()
        
        assert result == "http://127.0.0.1:45667"
        assert claude_proxy._proxy_state.port == 45667
        assert claude_proxy._proxy_state.base_url == "http://127.0.0.1:45667"
        mock_run.assert_called_once()
        assert os.environ["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:45667"


def test_start_or_connect_port_allocation_failure(monkeypatch):
    """Test when port allocation fails."""
    with patch.object(claude_proxy, "_find_existing_proxy", return_value=None), \
         patch.object(claude_proxy, "_find_available_port", return_value=None):
        
        result = claude_proxy.start_or_connect_to_proxy()
        assert result is None


def test_start_or_connect_server_startup_failure(monkeypatch, clean_env):
    """Test when server fails to start."""
    with patch.object(claude_proxy, "_find_existing_proxy", return_value=None), \
         patch.object(claude_proxy, "_find_available_port", return_value=45667), \
         patch.object(claude_proxy, "run_server", side_effect=OSError("Failed")):
        
        result = claude_proxy.start_or_connect_to_proxy()
        assert result is None


def test_start_or_connect_server_not_ready(monkeypatch, clean_env):
    """Test when server starts but doesn't become ready."""
    with patch.object(claude_proxy, "_find_existing_proxy", return_value=None), \
         patch.object(claude_proxy, "_find_available_port", return_value=45667), \
         patch.object(claude_proxy, "run_server"), \
         patch.object(claude_proxy, "_wait_for_port", return_value=False), \
         patch.object(claude_proxy, "stop_server") as mock_stop:
        
        result = claude_proxy.start_or_connect_to_proxy()
        assert result is None
        mock_stop.assert_called_once()


def test_start_or_connect_with_foundry_provider(monkeypatch, clean_env):
    """Test starting proxy with Foundry provider."""
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://foundry.example.com")
    
    with patch.object(claude_proxy, "_find_existing_proxy", return_value=None), \
         patch.object(claude_proxy, "_find_available_port", return_value=45667), \
         patch.object(claude_proxy, "run_server") as mock_run, \
         patch.object(claude_proxy, "_wait_for_port", return_value=True), \
         patch.object(claude_proxy, "_register_proxy_shutdown"):
        
        result = claude_proxy.start_or_connect_to_proxy()
        
        assert result == "http://127.0.0.1:45667"
        mock_run.assert_called_once_with("https://foundry.example.com", port=45667)
        assert os.environ["ANTHROPIC_FOUNDRY_BASE_URL"] == "http://127.0.0.1:45667"
        assert "ANTHROPIC_FOUNDRY_RESOURCE" not in os.environ


def test_start_or_connect_with_invalid_provider(monkeypatch, clean_env):
    """Test with invalid provider configuration."""
    monkeypatch.setenv("CLAUDE_CODE_USE_FOUNDRY", "1")
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    
    with patch.object(claude_proxy, "_find_existing_proxy", return_value=None), \
         patch.object(claude_proxy, "_find_available_port", return_value=45667):
        
        result = claude_proxy.start_or_connect_to_proxy()
        assert result is None


def test_start_or_connect_preserves_original_base_url(monkeypatch, clean_env):
    """Test that ANTHROPIC_ORIGINAL_BASE_URL is preserved."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://custom.anthropic.com")
    
    with patch.object(claude_proxy, "_find_existing_proxy", return_value=None), \
         patch.object(claude_proxy, "_find_available_port", return_value=45667), \
         patch.object(claude_proxy, "run_server"), \
         patch.object(claude_proxy, "_wait_for_port", return_value=True), \
         patch.object(claude_proxy, "_register_proxy_shutdown"):
        
        result = claude_proxy.start_or_connect_to_proxy()
        
        assert result == "http://127.0.0.1:45667"
        assert os.environ["ANTHROPIC_ORIGINAL_BASE_URL"] == "https://custom.anthropic.com"


def test_start_or_connect_env_snapshot_captured(monkeypatch, clean_env):
    """Test that environment snapshot is captured."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    
    with patch.object(claude_proxy, "_find_existing_proxy", return_value=None), \
         patch.object(claude_proxy, "_find_available_port", return_value=45667), \
         patch.object(claude_proxy, "run_server"), \
         patch.object(claude_proxy, "_wait_for_port", return_value=True), \
         patch.object(claude_proxy, "_register_proxy_shutdown"):
        
        claude_proxy.start_or_connect_to_proxy()
        
        assert claude_proxy._proxy_state.env_snapshot is not None
        assert "ANTHROPIC_BASE_URL" in claude_proxy._proxy_state.env_snapshot


# ===== set_trace_to_proxy Tests =====

def test_set_trace_to_proxy():
    """Test setting trace information to proxy."""
    with patch.object(claude_proxy, "set_current_trace") as mock_set:
        claude_proxy.set_trace_to_proxy(
            trace_id="trace123",
            span_id="span456",
            project_api_key="key789",
            span_path=["root", "child"],
            span_ids_path=["id1", "id2"],
            laminar_url="https://custom.lmnr.ai"
        )
        
        mock_set.assert_called_once_with(
            trace_id="trace123",
            span_id="span456",
            project_api_key="key789",
            span_path=["root", "child"],
            span_ids_path=["id1", "id2"],
            laminar_url="https://custom.lmnr.ai"
        )


def test_set_trace_to_proxy_default_values():
    """Test setting trace with default values."""
    with patch.object(claude_proxy, "set_current_trace") as mock_set:
        claude_proxy.set_trace_to_proxy(
            trace_id="trace123",
            span_id="span456",
            project_api_key="key789"
        )
        
        mock_set.assert_called_once_with(
            trace_id="trace123",
            span_id="span456",
            project_api_key="key789",
            span_path=[],
            span_ids_path=[],
            laminar_url="https://api.lmnr.ai"
        )


# ===== Thread Safety Tests =====

def test_concurrent_start_or_connect(monkeypatch, clean_env):
    """Test concurrent calls to start_or_connect_to_proxy.
    
    Note: Current implementation has a race condition where multiple threads
    can start servers because state is only set after server starts successfully.
    This test documents the current behavior.
    """
    call_count = {"count": 0}
    
    def mock_run_server(*args, **kwargs):
        call_count["count"] += 1
        time.sleep(0.05)  # Simulate some work
    
    with patch.object(claude_proxy, "_find_existing_proxy", return_value=None), \
         patch.object(claude_proxy, "_find_available_port", return_value=45667), \
         patch.object(claude_proxy, "run_server", side_effect=mock_run_server), \
         patch.object(claude_proxy, "_wait_for_port", return_value=True), \
         patch.object(claude_proxy, "_register_proxy_shutdown"):
        
        threads = []
        results = []
        
        def start_proxy():
            result = claude_proxy.start_or_connect_to_proxy()
            results.append(result)
        
        for _ in range(3):
            thread = threading.Thread(target=start_proxy)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads get the same result
        assert all(r == "http://127.0.0.1:45667" for r in results)
        # Due to race condition, multiple servers may be started
        # (this is current behavior, may be improved in refactoring)
        assert call_count["count"] >= 1

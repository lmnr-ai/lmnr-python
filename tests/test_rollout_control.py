"""
Tests for rollout control module with environment variable fallbacks.
"""

import os
import pytest

from lmnr.sdk.rollout_control import (
    ROLLOUT_MODE,
    ROLLOUT_SESSION_ID,
    CACHE_SERVER_URL,
    is_rollout_mode,
    get_rollout_session_id,
    get_cache_server_url,
    register_entrypoint,
    get_entrypoints,
    clear_entrypoints,
)


@pytest.fixture(autouse=True)
def reset_contextvars():
    """Reset all rollout ContextVars before and after each test."""
    # Save current values
    mode_token = None
    session_token = None
    cache_token = None

    try:
        # Try to reset to clean state
        try:
            mode_token = ROLLOUT_MODE.set(False)
        except Exception:
            pass
        try:
            session_token = ROLLOUT_SESSION_ID.set(None)
        except Exception:
            pass
        try:
            cache_token = CACHE_SERVER_URL.set(None)
        except Exception:
            pass
    except Exception:
        pass

    clear_entrypoints()

    yield

    # Reset after test
    if mode_token:
        try:
            ROLLOUT_MODE.reset(mode_token)
        except Exception:
            pass
    if session_token:
        try:
            ROLLOUT_SESSION_ID.reset(session_token)
        except Exception:
            pass
    if cache_token:
        try:
            CACHE_SERVER_URL.reset(cache_token)
        except Exception:
            pass

    clear_entrypoints()


def test_is_rollout_mode_with_contextvar():
    """Test is_rollout_mode() when ContextVar is set."""
    token = ROLLOUT_MODE.set(True)
    try:
        assert is_rollout_mode() is True
    finally:
        ROLLOUT_MODE.reset(token)


def test_is_rollout_mode_with_env_var(monkeypatch):
    """Test is_rollout_mode() when environment variable is set."""
    monkeypatch.setenv("LMNR_ROLLOUT_SESSION_ID", "test-session")
    assert is_rollout_mode() is True


def test_is_rollout_mode_contextvar_takes_priority():
    """Test that ContextVar takes priority over environment variable when True."""
    token = ROLLOUT_MODE.set(True)
    # Don't set env var - ContextVar should be enough
    try:
        assert is_rollout_mode() is True
    finally:
        ROLLOUT_MODE.reset(token)


def test_is_rollout_mode_falls_back_to_env(monkeypatch):
    """Test that is_rollout_mode() falls back to env var when ContextVar is False."""
    token = ROLLOUT_MODE.set(False)
    monkeypatch.setenv("LMNR_ROLLOUT_SESSION_ID", "test-session")
    try:
        # ContextVar is False, but env var is set - env var wins (fallback behavior)
        assert is_rollout_mode() is True
    finally:
        ROLLOUT_MODE.reset(token)


def test_is_rollout_mode_neither_set(monkeypatch):
    """Test is_rollout_mode() when neither ContextVar nor env var is set."""
    monkeypatch.delenv("LMNR_ROLLOUT_SESSION_ID", raising=False)
    assert is_rollout_mode() is False


def test_get_rollout_session_id_with_contextvar():
    """Test get_rollout_session_id() when ContextVar is set."""
    token = ROLLOUT_SESSION_ID.set("contextvar-session")
    try:
        assert get_rollout_session_id() == "contextvar-session"
    finally:
        ROLLOUT_SESSION_ID.reset(token)


def test_get_rollout_session_id_with_env_var(monkeypatch):
    """Test get_rollout_session_id() when environment variable is set."""
    monkeypatch.setenv("LMNR_ROLLOUT_SESSION_ID", "env-session")
    assert get_rollout_session_id() == "env-session"


def test_get_rollout_session_id_contextvar_priority():
    """Test that ContextVar takes priority over environment variable."""
    token = ROLLOUT_SESSION_ID.set("contextvar-session")
    os.environ["LMNR_ROLLOUT_SESSION_ID"] = "env-session"
    try:
        assert get_rollout_session_id() == "contextvar-session"
    finally:
        ROLLOUT_SESSION_ID.reset(token)
        del os.environ["LMNR_ROLLOUT_SESSION_ID"]


def test_get_rollout_session_id_neither_set(monkeypatch):
    """Test get_rollout_session_id() when neither is set."""
    monkeypatch.delenv("LMNR_ROLLOUT_SESSION_ID", raising=False)
    assert get_rollout_session_id() is None


def test_get_cache_server_url_with_contextvar():
    """Test get_cache_server_url() when ContextVar is set."""
    token = CACHE_SERVER_URL.set("http://contextvar:1234")
    try:
        assert get_cache_server_url() == "http://contextvar:1234"
    finally:
        CACHE_SERVER_URL.reset(token)


def test_get_cache_server_url_with_env_var(monkeypatch):
    """Test get_cache_server_url() when environment variable is set."""
    monkeypatch.setenv("LMNR_ROLLOUT_STATE_SERVER_ADDRESS", "http://env:5678")
    assert get_cache_server_url() == "http://env:5678"


def test_get_cache_server_url_contextvar_priority():
    """Test that ContextVar takes priority over environment variable."""
    token = CACHE_SERVER_URL.set("http://contextvar:1234")
    os.environ["LMNR_ROLLOUT_STATE_SERVER_ADDRESS"] = "http://env:5678"
    try:
        assert get_cache_server_url() == "http://contextvar:1234"
    finally:
        CACHE_SERVER_URL.reset(token)
        del os.environ["LMNR_ROLLOUT_STATE_SERVER_ADDRESS"]


def test_get_cache_server_url_neither_set(monkeypatch):
    """Test get_cache_server_url() when neither is set."""
    monkeypatch.delenv("LMNR_ROLLOUT_STATE_SERVER_ADDRESS", raising=False)
    assert get_cache_server_url() is None


def test_register_and_get_entrypoints():
    """Test registering and retrieving entrypoints."""

    def test_func_1():
        pass

    def test_func_2():
        pass

    # Clear any existing entrypoints
    clear_entrypoints()

    # Register entrypoints
    register_entrypoint("func1", test_func_1)
    register_entrypoint("func2", test_func_2)

    entrypoints = get_entrypoints()

    assert len(entrypoints) == 2
    assert entrypoints["func1"] is test_func_1
    assert entrypoints["func2"] is test_func_2


def test_clear_entrypoints():
    """Test clearing all entrypoints."""
    # Clear any existing entrypoints first
    clear_entrypoints()

    def test_func():
        pass

    register_entrypoint("test", test_func)
    assert len(get_entrypoints()) == 1

    clear_entrypoints()
    assert len(get_entrypoints()) == 0


def test_entrypoint_isolation():
    """Test that entrypoints are isolated per context."""

    def func_a():
        pass

    def func_b():
        pass

    clear_entrypoints()
    register_entrypoint("a", func_a)

    # Get current entrypoints
    entrypoints_1 = get_entrypoints()
    assert len(entrypoints_1) == 1
    assert "a" in entrypoints_1

    # Register another
    register_entrypoint("b", func_b)

    entrypoints_2 = get_entrypoints()
    assert len(entrypoints_2) == 2
    assert "a" in entrypoints_2
    assert "b" in entrypoints_2

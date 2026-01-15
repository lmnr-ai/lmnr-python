"""
Rollout control module for managing rollout mode state.

This module provides global state management for the rollout feature using
ContextVar to ensure thread-safety and proper async propagation.
"""

import os
from contextvars import ContextVar
from typing import Callable, Dict, Optional


# Global rollout state using ContextVar for thread-safety
ROLLOUT_MODE: ContextVar[bool] = ContextVar("__lmnr_rollout_mode", default=False)

ROLLOUT_ENTRYPOINTS: ContextVar[Dict[str, Callable]] = ContextVar(
    "__lmnr_rollout_entrypoints", default={}
)

ROLLOUT_SESSION_ID: ContextVar[Optional[str]] = ContextVar(
    "__lmnr_rollout_session_id", default=None
)

CACHE_SERVER_URL: ContextVar[Optional[str]] = ContextVar(
    "__lmnr_cache_server_url", default=None
)


def is_rollout_mode() -> bool:
    """
    Check if currently in rollout mode.

    First checks the ContextVar (set during entrypoint discovery).
    Falls back to checking the environment variable (set in subprocess execution).

    Returns:
        bool: True if in rollout mode, False otherwise
    """
    try:
        # First try to get from ContextVar (used during discovery phase)
        if ROLLOUT_MODE.get():
            return True
    except LookupError:
        pass

    # Fall back to environment variable (used during subprocess execution)
    return os.environ.get("LMNR_ROLLOUT_SESSION_ID") is not None


def register_entrypoint(name: str, func: Callable) -> None:
    """
    Register a rollout entrypoint function.

    Args:
        name: Name of the function to register
        func: The callable function to register
    """
    try:
        entrypoints = ROLLOUT_ENTRYPOINTS.get().copy()
    except LookupError:
        entrypoints = {}

    entrypoints[name] = func
    ROLLOUT_ENTRYPOINTS.set(entrypoints)


def get_entrypoints() -> Dict[str, Callable]:
    """
    Get all registered rollout entrypoint functions.

    Returns:
        Dict[str, Callable]: Dictionary mapping function names to callables
    """
    try:
        return ROLLOUT_ENTRYPOINTS.get()
    except LookupError:
        return {}


def get_rollout_session_id() -> Optional[str]:
    """
    Get the current rollout session ID.

    First checks the ContextVar, falls back to environment variable.

    Returns:
        Optional[str]: Session ID if set, None otherwise
    """
    try:
        session_id = ROLLOUT_SESSION_ID.get()
        if session_id:
            return session_id
    except LookupError:
        pass

    # Fall back to environment variable (set in subprocess)
    return os.environ.get("LMNR_ROLLOUT_SESSION_ID")


def get_cache_server_url() -> Optional[str]:
    """
    Get the cache server URL.

    First checks the ContextVar, falls back to environment variable.

    Returns:
        Optional[str]: Cache server URL if set, None otherwise
    """
    try:
        url = CACHE_SERVER_URL.get()
        if url:
            return url
    except LookupError:
        pass

    # Fall back to environment variable (set in subprocess)
    return os.environ.get("LMNR_ROLLOUT_STATE_SERVER_ADDRESS")


def clear_entrypoints() -> None:
    """
    Clear all registered entrypoints.

    This is useful when reloading modules in hot reload scenarios.
    """
    ROLLOUT_ENTRYPOINTS.set({})

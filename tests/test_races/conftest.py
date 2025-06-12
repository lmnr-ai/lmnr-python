"""
Separate conftest for race condition testing.

This file provides fixtures that don't interfere with race condition tests
by avoiding global tracer initialization, but properly restores state
after tests complete.

To use this instead of the main conftest.py, run pytest with:
    pytest --confcutdir=tests/conftest_race_testing.py tests/test_tracer_wrapper_race_condition.py
"""

import pytest
from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.opentelemetry_lib import TracerManager


@pytest.fixture(scope="function", autouse=True)
def clean_tracer_state():
    """Clean tracer state before and after each test, properly saving/restoring global state."""
    # Save current state before clearing
    saved_wrapper_instance = getattr(TracerWrapper, "instance", None)
    saved_manager_wrapper = getattr(
        TracerManager, "_TracerManager__tracer_wrapper", None
    )

    # Clean before test
    if hasattr(TracerWrapper, "instance"):
        delattr(TracerWrapper, "instance")

    if hasattr(TracerManager, "_TracerManager__tracer_wrapper"):
        delattr(TracerManager, "_TracerManager__tracer_wrapper")

    yield

    # Clean after test
    if hasattr(TracerWrapper, "instance"):
        delattr(TracerWrapper, "instance")

    if hasattr(TracerManager, "_TracerManager__tracer_wrapper"):
        delattr(TracerManager, "_TracerManager__tracer_wrapper")

    # Restore saved state to avoid breaking other tests
    if saved_wrapper_instance is not None:
        TracerWrapper.instance = saved_wrapper_instance

    if saved_manager_wrapper is not None:
        TracerManager._TracerManager__tracer_wrapper = saved_manager_wrapper

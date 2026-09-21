"""
Separate conftest for race condition testing.

Race tests drop the global tracer singleton, so this fixture saves and restores
it around each test to avoid breaking the session-scoped `span_exporter`
fixture the rest of the suite shares.
"""

import pytest

from lmnr.opentelemetry_lib import tracing as tracing_mod
from lmnr.opentelemetry_lib.tracing import reset_tracing


@pytest.fixture(scope="function", autouse=True)
def clean_tracer_state():
    saved_wrapper = tracing_mod._tracer_wrapper
    saved_options = tracing_mod._session_recording_options

    # Detach (don't reset_tracing) so the saved wrapper keeps its atexit hook.
    tracing_mod._tracer_wrapper = None
    tracing_mod._session_recording_options = None

    yield

    # Drops whatever wrapper the test built, unregistering its atexit hook.
    reset_tracing()

    tracing_mod._tracer_wrapper = saved_wrapper
    tracing_mod._session_recording_options = saved_options

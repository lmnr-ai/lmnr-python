"""Fixtures for the Langfuse bridge tests."""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock

import pytest

from lmnr.opentelemetry_lib.tracing import instruments as instruments_mod
from lmnr.opentelemetry_lib.tracing.instruments import (
    INSTRUMENTATION_INITIALIZERS,
    Instruments,
)

from .utils import (
    LANGFUSE_IMPORTABLE,
    logger,
    reset_langfuse_bridge_state,
    silence_langfuse_background_threads,
)

# Silence Langfuse's OTLP exporter errors during tests — it will try to POST
# to a real endpoint on flush and fail loudly otherwise.
_new_val = os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
_new_val = os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-test")
_new_val = os.environ.setdefault("LANGFUSE_HOST", "http://127.0.0.1:1")
# Disable the media-upload consumer thread. Its `run()` loop blocks on
# `queue.get(block=True, timeout=1)` with a HARDCODED 1s timeout (unaffected by
# flush_interval), so `client.shutdown()` would join() it for ~1s per test —
# the dominant cost in this file's teardown.
_new_val = os.environ.setdefault("LANGFUSE_MEDIA_UPLOAD_ENABLED", "False")
logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.exporter.otlp.proto.http").setLevel(logging.CRITICAL)


@pytest.fixture
def reset_langfuse_bridge():
    """Ensure the bridge singleton starts AND ends the test in a clean,
    uninstalled state."""
    reset_langfuse_bridge_state()
    yield
    reset_langfuse_bridge_state()


# ---------------------------------------------------------------------------
# init_instrumentations auto-enable tests (no real langfuse interaction)
# ---------------------------------------------------------------------------


@pytest.fixture
def track_initializers(monkeypatch: pytest.MonkeyPatch) -> set[Instruments]:
    """Patch every initializer in the map to record which were invoked."""
    called: set[Instruments] = set()
    replacements: dict[Instruments, object] = {}

    for instrument, initializer in INSTRUMENTATION_INITIALIZERS.items():
        fake = MagicMock(spec=initializer)
        fake.init_instrumentor = MagicMock(
            side_effect=lambda *_a, _inst=instrument, **_kw: called.add(_inst) or None  # pyright: ignore[reportUnknownLambdaType]
        )
        replacements[instrument] = fake

    monkeypatch.setattr(
        instruments_mod,
        "INSTRUMENTATION_INITIALIZERS",
        replacements,
    )
    return called


@pytest.fixture
def langfuse_installed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(instruments_mod, "langfuse_installed", lambda: True)


@pytest.fixture
def langfuse_not_installed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(instruments_mod, "langfuse_installed", lambda: False)


@pytest.fixture(autouse=True)
def google_adk_not_installed(monkeypatch: pytest.MonkeyPatch):
    """These tests aren't about google-adk; `google-adk` is a pinned dev
    dependency, and leaving its real installedness in place would trip the
    GOOGLE_GENAI auto-removal these tests don't expect (see
    tests/test_google_adk.py for ADK-specific coverage)."""
    monkeypatch.setattr(instruments_mod, "_google_adk_installed", lambda: False)


@pytest.fixture
def deepagents_installed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(instruments_mod, "_deepagents_installed", lambda: True)


@pytest.fixture
def deepagents_not_installed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(instruments_mod, "_deepagents_installed", lambda: False)


@pytest.fixture
def pydantic_ai_not_installed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(instruments_mod, "_pydantic_ai_installed", lambda: False)


@pytest.fixture
def pydantic_ai_installed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(instruments_mod, "_pydantic_ai_installed", lambda: True)

@pytest.fixture
def langfuse_client():
    """Fresh Langfuse client with an isolated resource manager.

    Langfuse caches instances in a module-level singleton keyed by
    public_key. We reset the cache each test so every test gets its own
    TracerProvider (otherwise earlier tests' bridges stack up and confuse the
    assertions).
    """
    if not LANGFUSE_IMPORTABLE:
        pytest.skip(
            "langfuse SDK not importable on this interpreter " +
            "(known pydantic v1 incompatibility on Python 3.14)"
        )
    # Delay import so tests that don't need langfuse can still be collected.
    from langfuse._client.resource_manager import LangfuseResourceManager

    # Clear the singleton cache so we get a fresh TracerProvider.
    LangfuseResourceManager._instances.clear()

    from langfuse import Langfuse

    client = Langfuse()
    silence_langfuse_background_threads(LangfuseResourceManager._instances.values())

    yield client
    try:
        client.shutdown()
    except Exception:
        logger.debug("Failed to shutdown langfuse client")
    LangfuseResourceManager._instances.clear()

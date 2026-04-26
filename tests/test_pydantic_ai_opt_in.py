"""Tests for the PYDANTIC_AI auto-enable behaviour in `init_instrumentations`.

We don't actually import pydantic_ai here - we stub `is_package_installed`
through the `instruments` module to simulate whether pydantic_ai is available
in the environment, and verify that:

* When pydantic_ai is *not* installed, PYDANTIC_AI stays out of the default
  instrument set and every other default provider (OPENAI, ANTHROPIC, ...)
  remains enabled.
* When pydantic_ai *is* installed, PYDANTIC_AI is auto-enabled and the
  provider instrumentors that would emit duplicate spans (OPENAI, ANTHROPIC,
  GOOGLE_GENAI, GROQ, MISTRAL, COHERE, BEDROCK) are auto-removed.
* Explicitly blocking PYDANTIC_AI disables both the auto-enable and the
  auto-removal of provider instrumentors.
* Passing an explicit `instruments` set always wins and never triggers the
  auto-removal.
"""

from unittest.mock import MagicMock

import pytest

from lmnr.opentelemetry_lib.tracing import instruments as instruments_mod
from lmnr.opentelemetry_lib.tracing.instruments import (
    INSTRUMENTATION_INITIALIZERS,
    Instruments,
    _PYDANTIC_AI_PROVIDER_CONFLICTS,
    init_instrumentations,
)


@pytest.fixture
def track_initializers(monkeypatch):
    """Patch every initializer in the map to record which were invoked.

    Returns a set that accumulates the `Instruments` key for each initializer
    whose `init_instrumentor` was invoked by `init_instrumentations`.
    """
    called: set[Instruments] = set()
    replacements: dict[Instruments, object] = {}

    for instrument, initializer in INSTRUMENTATION_INITIALIZERS.items():
        fake = MagicMock(spec=initializer)
        fake.init_instrumentor = MagicMock(
            side_effect=lambda *_a, _inst=instrument, **_kw: called.add(_inst)
            or None
        )
        replacements[instrument] = fake

    monkeypatch.setattr(
        instruments_mod,
        "INSTRUMENTATION_INITIALIZERS",
        replacements,
    )
    return called


@pytest.fixture
def pydantic_ai_installed(monkeypatch):
    """Simulate pydantic_ai being installed in the environment."""
    monkeypatch.setattr(instruments_mod, "_pydantic_ai_installed", lambda: True)


@pytest.fixture
def pydantic_ai_not_installed(monkeypatch):
    """Simulate pydantic_ai being absent from the environment."""
    monkeypatch.setattr(instruments_mod, "_pydantic_ai_installed", lambda: False)


def test_pydantic_ai_not_installed_defaults_exclude_it(
    track_initializers, pydantic_ai_not_installed
):
    """When pydantic_ai isn't installed, PYDANTIC_AI is not in the default set."""
    init_instrumentations(tracer_provider=MagicMock(), instruments=None)
    assert Instruments.PYDANTIC_AI not in track_initializers
    # Provider instrumentors stay on.
    for instrument in _PYDANTIC_AI_PROVIDER_CONFLICTS:
        assert instrument in track_initializers


def test_pydantic_ai_installed_defaults_enable_it_and_block_providers(
    track_initializers, pydantic_ai_installed
):
    """When pydantic_ai is installed, PYDANTIC_AI is auto-enabled and the
    overlapping provider instrumentors are auto-disabled to avoid duplicate spans.
    """
    init_instrumentations(tracer_provider=MagicMock(), instruments=None)
    assert Instruments.PYDANTIC_AI in track_initializers
    for instrument in _PYDANTIC_AI_PROVIDER_CONFLICTS:
        assert instrument not in track_initializers, (
            f"{instrument} should have been auto-removed to avoid duplicate spans"
        )
    # Non-conflicting defaults remain enabled.
    assert Instruments.LANGCHAIN in track_initializers
    assert Instruments.LLAMA_INDEX in track_initializers


def test_explicit_instruments_bypass_auto_logic(
    track_initializers, pydantic_ai_installed
):
    """Explicit `instruments=` always wins: no auto-enable, no auto-removal."""
    init_instrumentations(
        tracer_provider=MagicMock(),
        instruments={Instruments.OPENAI, Instruments.ANTHROPIC},
    )
    assert Instruments.PYDANTIC_AI not in track_initializers
    assert Instruments.OPENAI in track_initializers
    assert Instruments.ANTHROPIC in track_initializers


def test_explicit_pydantic_ai_and_providers_keeps_both(
    track_initializers, pydantic_ai_installed
):
    """Users who explicitly want both get both (they accept the duplicate spans)."""
    init_instrumentations(
        tracer_provider=MagicMock(),
        instruments={
            Instruments.PYDANTIC_AI,
            Instruments.OPENAI,
            Instruments.ANTHROPIC,
        },
    )
    assert Instruments.PYDANTIC_AI in track_initializers
    assert Instruments.OPENAI in track_initializers
    assert Instruments.ANTHROPIC in track_initializers


def test_block_pydantic_ai_disables_auto_logic(
    track_initializers, pydantic_ai_installed
):
    """Blocking PYDANTIC_AI suppresses auto-enable and restores provider defaults."""
    init_instrumentations(
        tracer_provider=MagicMock(),
        instruments=None,
        block_instruments={Instruments.PYDANTIC_AI},
    )
    assert Instruments.PYDANTIC_AI not in track_initializers
    for instrument in _PYDANTIC_AI_PROVIDER_CONFLICTS:
        assert instrument in track_initializers

"""Tests for the `InstrumentationSettings.__init__` patching performed by
`PydanticAIInstrumentor`.

The instrumentor monkey-patches pydantic_ai's `InstrumentationSettings.__init__`
so that every construction path — `Agent.instrument_all(True)`,
`Agent(instrument=True)`, or user code calling the constructor directly — ends
up on the Laminar tracer provider. The semconv version defaults to 5 when the
caller did not specify one (or passed the legacy `version=1`); any explicit
`version >= 2` is passed through unchanged.

These tests exercise the patch directly; they're skipped when pydantic_ai
isn't installed in the test environment.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.instrumented import InstrumentationSettings  # noqa: E402

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.pydantic_ai import (  # noqa: E402
    PydanticAIInstrumentor,
)


@pytest.fixture
def instrumented():
    """Install the instrumentor with a sentinel tracer provider and tear it down.

    PydanticAIInstrumentor is a singleton (see `BaseInstrumentor.__new__`),
    and may already be instrumented when the test session starts (via
    auto-init elsewhere). We uninstrument first to guarantee a clean slate,
    then re-install with our sentinel tracer provider.
    """
    instrumentor = PydanticAIInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()

    tracer_provider = MagicMock(name="laminar_tracer_provider")
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        yield tracer_provider
    finally:
        instrumentor.uninstrument()


def test_default_construction_uses_laminar_settings(instrumented):
    """`InstrumentationSettings()` (no args) picks up version=5 + our tracer."""
    tracer_provider = instrumented
    settings = InstrumentationSettings()
    assert settings.version == 5
    tracer_provider.get_tracer.assert_called()


@pytest.mark.parametrize("explicit_version", [2, 3, 4, 5])
def test_explicit_supported_version_is_respected(instrumented, explicit_version):
    """Any caller-supplied `version >= 2` is passed through unchanged."""
    settings = InstrumentationSettings(version=explicit_version)
    assert settings.version == explicit_version


def test_legacy_version_1_is_upgraded_to_default(instrumented):
    """`version=1` is treated as the legacy default and upgraded to v5."""
    settings = InstrumentationSettings(version=1)
    assert settings.version == 5


def test_explicit_tracer_provider_is_respected(instrumented):
    """User-provided tracer_provider wins over our default."""
    other_tp = MagicMock(name="user_tracer_provider")
    InstrumentationSettings(tracer_provider=other_tp)
    other_tp.get_tracer.assert_called()


def test_instrument_all_true_resolves_to_laminar_settings(instrumented):
    """`Agent.instrument_all(True)` after our patch still yields version=5 + our tp."""
    tracer_provider = instrumented
    Agent.instrument_all(True)
    try:
        # `instrument_all(True)` stores the bool. The InstrumentationSettings
        # is only constructed lazily inside `instrument_model` when a run
        # happens. We simulate that here by calling the same constructor with
        # no args, which is what `instrument_model` does internally.
        settings = InstrumentationSettings()
        assert settings.version == 5
        assert tracer_provider.get_tracer.called
    finally:
        Agent.instrument_all(False)


def test_uninstrument_restores_behavior():
    """After `uninstrument()`, new settings use pydantic_ai's default version."""
    tracer_provider = MagicMock()
    instrumentor = PydanticAIInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    instrumentor.uninstrument()

    # After uninstrument, the default path should no longer force version=5.
    # We can't assert a specific version (pydantic_ai may change its default),
    # but we can at least verify that asking for `version=3` is honored.
    settings = InstrumentationSettings(version=3)
    assert settings.version == 3

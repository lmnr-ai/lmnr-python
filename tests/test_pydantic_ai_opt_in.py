"""Tests for the PYDANTIC_AI opt-in behaviour in `init_instrumentations`.

We don't actually import pydantic_ai here - we just verify that
`init_instrumentations` never enables `Instruments.PYDANTIC_AI` by default
and only enables it when explicitly requested via the `instruments` argument.
"""

from unittest.mock import MagicMock

import pytest

from lmnr.opentelemetry_lib.tracing import instruments as instruments_mod
from lmnr.opentelemetry_lib.tracing.instruments import (
    INSTRUMENTATION_INITIALIZERS,
    Instruments,
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


def test_pydantic_ai_not_in_defaults(track_initializers):
    """When `instruments` is None, PYDANTIC_AI must NOT be enabled."""
    init_instrumentations(tracer_provider=MagicMock(), instruments=None)
    assert Instruments.PYDANTIC_AI not in track_initializers


def test_pydantic_ai_explicit_opt_in_enables_it(track_initializers):
    """Passing PYDANTIC_AI in `instruments` enables its initializer."""
    init_instrumentations(
        tracer_provider=MagicMock(),
        instruments={Instruments.PYDANTIC_AI},
    )
    assert Instruments.PYDANTIC_AI in track_initializers


def test_pydantic_ai_with_explicit_provider_keeps_both(track_initializers):
    """Opting in to PYDANTIC_AI alongside OPENAI/ANTHROPIC enables all."""
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


def test_no_pydantic_ai_leaves_defaults_alone(track_initializers):
    """Default init (no PYDANTIC_AI) still enables OPENAI, ANTHROPIC, etc."""
    init_instrumentations(tracer_provider=MagicMock(), instruments=None)
    assert Instruments.OPENAI in track_initializers
    assert Instruments.ANTHROPIC in track_initializers

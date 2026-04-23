"""Tests for the PYDANTIC_AI opt-in and conflicting-instrumentor auto-block logic.

We don't actually import pydantic_ai here - we just verify that
`init_instrumentations` routes `Instruments.PYDANTIC_AI` correctly and blocks
the overlapping provider instrumentors unless they are explicitly opted in.
"""

from unittest.mock import MagicMock

import pytest

from lmnr.opentelemetry_lib.tracing import instruments as instruments_mod
from lmnr.opentelemetry_lib.tracing.instruments import (
    _PYDANTIC_AI_CONFLICTING_INSTRUMENTS,
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


def test_pydantic_ai_with_explicit_provider_keeps_both(track_initializers):
    """Opting in to PYDANTIC_AI alongside OPENAI/ANTHROPIC respects explicit opt-in."""
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


def test_pydantic_ai_alone_blocks_all_conflicting(track_initializers):
    """Opting in to only PYDANTIC_AI auto-blocks all conflicting providers."""
    init_instrumentations(
        tracer_provider=MagicMock(), instruments={Instruments.PYDANTIC_AI}
    )
    assert Instruments.PYDANTIC_AI in track_initializers
    for conflicting in _PYDANTIC_AI_CONFLICTING_INSTRUMENTS:
        assert (
            conflicting not in track_initializers
        ), f"{conflicting} should have been auto-blocked"


def test_no_pydantic_ai_leaves_defaults_alone(track_initializers):
    """Default init (no PYDANTIC_AI) still enables OPENAI, ANTHROPIC, etc."""
    init_instrumentations(tracer_provider=MagicMock(), instruments=None)
    assert Instruments.OPENAI in track_initializers
    assert Instruments.ANTHROPIC in track_initializers

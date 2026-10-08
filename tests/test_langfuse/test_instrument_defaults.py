"""Tests for `init_instrumentations` opt-in behaviour of the Langfuse bridge."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


from lmnr.opentelemetry_lib.tracing import instruments as instruments_mod
from lmnr.opentelemetry_lib.tracing.instruments import (
    Instruments,
    init_instrumentations,
)

from .utils import (
    LANGFUSE_OVERLAP_PROVIDERS,
)


def test_langfuse_not_installed_defaults_exclude_it(
    track_initializers: set[Instruments],
    langfuse_not_installed: Any,
    pydantic_ai_not_installed: Any,
):
    """When langfuse is absent, LANGFUSE is not in the default set and all
    overlapping raw-provider instrumentors remain enabled."""
    init_instrumentations(tracer_provider=MagicMock(), instruments=None)
    assert Instruments.LANGFUSE not in track_initializers
    for instrument in LANGFUSE_OVERLAP_PROVIDERS:
        assert (
            instrument in track_initializers
        ), f"{instrument} should remain when langfuse isn't installed"


def test_langfuse_initializer_skips_on_unreadable_or_invalid_version(monkeypatch: MagicMock):
    """Regression: the initializer used to pass version guards when
    `get_package_version` returned `None` (the check
    `if version and parse(version) < parse("3.0.0")` short-circuits on None),
    silently installing the bridge. For an explicit
    `instruments={Instruments.LANGFUSE}` call this would flip `_installed=True`
    and block any later valid install. Same story for an unparseable version
    string. Both must return `None` from the initializer."""
    from lmnr.opentelemetry_lib.tracing import _instrument_initializers

    monkeypatch.setattr(
        _instrument_initializers,
        "is_package_installed",
        lambda name: name == "langfuse",  # pyright: ignore[reportUnknownLambdaType]
    )

    # Unreadable version.
    monkeypatch.setattr(
        _instrument_initializers,
        "get_package_version",
        lambda name: None,  # pyright: ignore[reportUnknownLambdaType]
    )
    initializer = _instrument_initializers.LangfuseInstrumentorInitializer()
    assert initializer.init_instrumentor() is None

    # Unparseable version string.
    monkeypatch.setattr(
        _instrument_initializers,
        "get_package_version",
        lambda name: "not-a-version",  # pyright: ignore[reportUnknownLambdaType]
    )
    assert initializer.init_instrumentor() is None

    # Happy path — parseable, >= 3.0 → instrumentor returned.
    monkeypatch.setattr(
        _instrument_initializers,
        "get_package_version",
        lambda name: "3.14.6",  # pyright: ignore[reportUnknownLambdaType]
    )
    assert initializer.init_instrumentor() is not None


def test_langfuse_v2_reports_not_installed(monkeypatch: MagicMock):
    """langfuse 2.x is not OTel-native, so the bridge initializer returns None.
    `langfuse_installed()` must report False in that case so
    `connect_to_langfuse()` refuses to install a useless translator."""
    monkeypatch.setattr(
        instruments_mod,
        "is_package_installed",
        lambda name: name == "langfuse",  # pyright: ignore[reportUnknownLambdaType]
    )
    monkeypatch.setattr(
        instruments_mod,
        "get_package_version",
        lambda name: "2.60.0" if name == "langfuse" else None,  # pyright: ignore[reportUnknownLambdaType]
    )
    assert instruments_mod.langfuse_installed() is False

    monkeypatch.setattr(
        instruments_mod,
        "get_package_version",
        lambda name: "3.0.0" if name == "langfuse" else None,
    )
    assert instruments_mod.langfuse_installed() is True


def test_langfuse_installed_is_never_in_default_set(
    track_initializers: set[Instruments],
    langfuse_installed: Any,
    pydantic_ai_not_installed: Any,
    deepagents_not_installed: Any,
):
    """The Langfuse bridge is opt-in: having langfuse installed must NOT add
    LANGFUSE to the default set or strip any overlapping raw-provider
    instrumentor. This guards the development-environment footgun where a
    transitive langfuse install would otherwise silently disable Laminar's own
    tracing for a user who only instrumented with Laminar."""
    init_instrumentations(
        tracer_provider=MagicMock(),
        instruments=None,
        lmnr_span_processor=MagicMock(),
    )
    assert Instruments.LANGFUSE not in track_initializers
    for instrument in LANGFUSE_OVERLAP_PROVIDERS:
        assert (
            instrument in track_initializers
        ), f"{instrument} should remain when the bridge isn't opted into"


def test_langfuse_opt_in_via_explicit_instruments(
    track_initializers: set[Instruments],
    langfuse_installed: Any,
    pydantic_ai_not_installed: Any,
    deepagents_not_installed: Any,
):
    """Passing an explicit `instruments` set is the opt-in path. The set is
    honored verbatim — LANGFUSE runs alongside whatever else the caller asked
    for, with no auto-removal of overlapping providers (the caller accepts the
    double-coverage cost). Auto-enable logic for pydantic_ai/deepagents does
    NOT run when `instruments` is explicit."""
    init_instrumentations(
        tracer_provider=MagicMock(),
        instruments={Instruments.LANGFUSE, Instruments.OPENAI},
        lmnr_span_processor=MagicMock(),
    )
    assert Instruments.LANGFUSE in track_initializers
    assert Instruments.OPENAI in track_initializers


def test_explicit_instruments_without_langfuse_skips_bridge(
    track_initializers: set[Instruments],
    langfuse_installed: Any,
    pydantic_ai_not_installed: Any,
    deepagents_not_installed: Any,
):
    """An explicit `instruments` set that omits LANGFUSE leaves the bridge
    off even though langfuse is installed — only what was asked for runs."""
    init_instrumentations(
        tracer_provider=MagicMock(),
        instruments={Instruments.OPENAI, Instruments.ANTHROPIC},
        lmnr_span_processor=MagicMock(),
    )
    assert Instruments.OPENAI in track_initializers
    assert Instruments.ANTHROPIC in track_initializers
    assert Instruments.LANGFUSE not in track_initializers

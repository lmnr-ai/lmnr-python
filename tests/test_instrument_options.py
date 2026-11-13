from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any, get_type_hints

import pytest

from lmnr import InstrumentOptions, Instruments
from lmnr.opentelemetry_lib.tracing import instrument_options
from lmnr.opentelemetry_lib.tracing.instrument_options import (
    OPTION_DEFAULTS,
    env_var_name,
    explicit_instrument_option,
    get_instrument_option,
    resolve_instrument_options,
    set_instrument_options,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for instrument, defaults in OPTION_DEFAULTS.items():
        for option in defaults:
            monkeypatch.delenv(env_var_name(instrument, option), raising=False)
    # Keep a developer's .env out of the resolution
    monkeypatch.setattr(instrument_options, "from_env", os.getenv)
    yield
    # Not `set_instrument_options(None)`: that would resolve now, before
    # monkeypatch restores the environment, and cache this test's env vars
    instrument_options.reset_instrument_options()


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    messages: list[str] = []
    monkeypatch.setattr(instrument_options.logger, "warning", messages.append)
    return messages


def test_public_type_matches_the_defaults_table():
    hints = get_type_hints(InstrumentOptions)
    assert set(hints) == set(OPTION_DEFAULTS)
    for instrument, option_type in hints.items():
        assert set(get_type_hints(option_type)) == set(OPTION_DEFAULTS[instrument])
        # Every key is an instrument users can enable or disable
        assert Instruments(instrument).value == instrument


def test_env_var_names():
    assert env_var_name("langchain", "record_metadata") == "LMNR_LANGCHAIN_RECORD_METADATA"
    assert env_var_name("langgraph", "record_graph") == "LMNR_LANGGRAPH_RECORD_GRAPH"
    assert env_var_name("temporal", "record_activity_args") == "LMNR_TEMPORAL_RECORD_ACTIVITY_ARGS"


def test_defaults():
    set_instrument_options(None)
    assert get_instrument_option("langchain", "record_metadata") is False
    assert get_instrument_option("langgraph", "record_graph") is True
    assert get_instrument_option("temporal", "create_activity_span") is True
    assert explicit_instrument_option("langgraph", "record_graph") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), (" TRUE ", True), ("1", True), ("yes", True), ("on", True)]
    + [("false", False), ("False", False), ("0", False), ("no", False), ("off", False)],
)
def test_env_values(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool):
    monkeypatch.setenv("LMNR_LANGGRAPH_RECORD_GRAPH", raw)
    set_instrument_options(None)
    assert explicit_instrument_option("langgraph", "record_graph") is expected


def test_invalid_env_value_is_ignored(monkeypatch: pytest.MonkeyPatch, warnings: list[str]):
    monkeypatch.setenv("LMNR_LANGGRAPH_RECORD_GRAPH", "nope")
    set_instrument_options(None)
    assert explicit_instrument_option("langgraph", "record_graph") is None
    assert get_instrument_option("langgraph", "record_graph") is True
    assert len(warnings) == 1
    assert "LMNR_LANGGRAPH_RECORD_GRAPH" in warnings[0]


def test_argument_wins_over_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LMNR_LANGCHAIN_RECORD_METADATA", "true")
    monkeypatch.setenv("LMNR_LANGGRAPH_RECORD_GRAPH", "false")
    set_instrument_options({"langchain": {"record_metadata": False}})
    # The argument wins for the option it sets; the env still applies to others
    assert get_instrument_option("langchain", "record_metadata") is False
    assert get_instrument_option("langgraph", "record_graph") is False


def test_enum_keys_are_accepted():
    options: dict[Any, Any] = {Instruments.LANGGRAPH: {"record_graph": False}}
    assert resolve_instrument_options(options) == {"langgraph": {"record_graph": False}}


def test_bad_options_are_ignored_with_a_warning(warnings: list[str]):
    options: dict[str, Any] = {
        "openai": {"record_metadata": True},
        "langgraph": False,
        "langchain": {"record_metdata": True, "record_metadata": "yes"},
        "temporal": {"record_activity_args": False},
    }
    assert resolve_instrument_options(options) == {"temporal": {"record_activity_args": False}}
    assert len(warnings) == 4


def test_lazily_resolved_from_env_before_initialize(monkeypatch: pytest.MonkeyPatch):
    instrument_options.reset_instrument_options()
    monkeypatch.setenv("LMNR_LANGCHAIN_RECORD_METADATA", "true")
    assert get_instrument_option("langchain", "record_metadata") is True

import json
import os
from pathlib import Path

import pytest

from lmnr.sdk.debug.config import build_debug_config

_DEBUG_ENV_KEYS = (
    "LMNR_DEBUG",
    "LMNR_DEBUG_SESSION_ID",
    "LMNR_DEBUG_REPLAY_TRACE_ID",
    "LMNR_DEBUG_CACHE_UNTIL",
)

_VECTORS = json.loads(
    (Path(__file__).parent / "data" / "debug" / "config_truth_table.json").read_text()
)["cases"]


@pytest.fixture(autouse=True)
def _clear_debug_env(monkeypatch):
    for key in _DEBUG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("case", _VECTORS, ids=[c["name"] for c in _VECTORS])
def test_config_truth_table(case, monkeypatch):
    for key, value in case["env"].items():
        monkeypatch.setenv(key, value)

    config = build_debug_config()
    expect = case["expect"]

    if expect is None:
        assert config is None
        return

    assert config is not None
    assert config.session_id == expect["session_id"]
    assert config.replay_trace_id == expect["replay_trace_id"]
    assert config.cache_until == expect["cache_until"]
    assert config.replay_enabled is expect["replay_enabled"]


def test_session_id_defaults_to_uuid(monkeypatch):
    monkeypatch.setenv("LMNR_DEBUG", "true")
    config = build_debug_config()
    assert config is not None
    # A generated uuid4 is 36 chars with 4 hyphens.
    assert len(config.session_id) == 36
    assert config.session_id.count("-") == 4


def test_disabled_when_env_absent():
    assert os.environ.get("LMNR_DEBUG") is None
    assert build_debug_config() is None

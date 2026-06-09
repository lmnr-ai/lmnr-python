import json
import os
from pathlib import Path

import pytest

from lmnr.sdk.debug.config import (
    build_debug_config,
    build_debug_config_from_context,
)
from lmnr.sdk.types import DebugContext, LaminarSpanContext

_DEBUG_ENV_KEYS = (
    "LMNR_DEBUG",
    "LMNR_DEBUG_SESSION_ID",
    "LMNR_DEBUG_REPLAY_TRACE_ID",
    "LMNR_DEBUG_CACHE_UNTIL",
    "LMNR_DEBUG_FROM_LAST_RUN",
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
    assert config.cache_until_span_id == expect.get("cache_until_span_id")
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


def _write_last_run(tmp_path: Path, payload: dict) -> None:
    pointer_dir = tmp_path / ".lmnr"
    pointer_dir.mkdir(parents=True, exist_ok=True)
    (pointer_dir / "last-run.json").write_text(json.dumps(payload))


def test_from_last_run_seeds_replay_from_pointer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_last_run(
        tmp_path,
        {
            "trace_id": "trace-abc",
            "session_id": "session-xyz",
            "cache_until": "0123-456789abcdef",
        },
    )
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_DEBUG_FROM_LAST_RUN", "true")

    config = build_debug_config()
    assert config is not None
    assert config.replay_trace_id == "trace-abc"
    assert config.session_id == "session-xyz"
    assert config.cache_until_span_id == "0123456789abcdef"
    assert config.replay_enabled is True


def test_from_last_run_env_overrides_per_field(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_last_run(
        tmp_path,
        {
            "trace_id": "trace-abc",
            "session_id": "session-xyz",
            "cache_until": "0123-456789abcdef",
        },
    )
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_DEBUG_FROM_LAST_RUN", "true")
    monkeypatch.setenv("LMNR_DEBUG_REPLAY_TRACE_ID", "trace-override")
    monkeypatch.setenv("LMNR_DEBUG_CACHE_UNTIL", "fedcba")

    config = build_debug_config()
    assert config is not None
    assert config.replay_trace_id == "trace-override"
    assert config.session_id == "session-xyz"
    assert config.cache_until_span_id == "fedcba"


def test_from_last_run_ignored_when_flag_falsey(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_last_run(tmp_path, {"trace_id": "trace-abc", "session_id": "session-xyz"})
    monkeypatch.setenv("LMNR_DEBUG", "true")

    config = build_debug_config()
    assert config is not None
    assert config.replay_trace_id is None
    assert config.session_id != "session-xyz"


def test_from_last_run_missing_file_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_DEBUG_FROM_LAST_RUN", "true")

    config = build_debug_config()
    assert config is not None
    assert config.replay_trace_id is None
    assert len(config.session_id) == 36


def test_local_origin_and_session_minted_from_env():
    config = build_debug_config_env_with({"LMNR_DEBUG": "true"})
    assert config is not None
    assert config.local_origin is True
    assert config.session_minted is True


def build_debug_config_env_with(env: dict):
    import os as _os

    saved = {k: _os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            _os.environ[k] = v
        return build_debug_config()
    finally:
        for k, v in saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v


def test_provided_session_id_not_minted(monkeypatch):
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_DEBUG_SESSION_ID", "sess-123")
    config = build_debug_config()
    assert config is not None
    assert config.session_minted is False


# --- DebugContext parsing (consumer side) ---

_SESSION = "00000000-0000-0000-0000-0000000000aa"
_REPLAY = "00000000-0000-0000-0000-0000000000bb"


def test_debug_context_parse_camel_case():
    ctx = DebugContext.deserialize(
        {
            "enabled": True,
            "sessionId": _SESSION,
            "replayTraceId": _REPLAY,
            "cacheUntil": "0123-456789abcdef",
        }
    )
    assert ctx.enabled is True
    assert ctx.session_id == _SESSION
    assert ctx.replay_trace_id == _REPLAY
    assert ctx.cache_until == "0123-456789abcdef"


def test_debug_context_parse_snake_case():
    ctx = DebugContext.deserialize(
        {
            "enabled": True,
            "session_id": _SESSION,
            "replay_trace_id": _REPLAY,
            "cache_until": "abcdef",
        }
    )
    assert ctx.session_id == _SESSION
    assert ctx.replay_trace_id == _REPLAY
    assert ctx.cache_until == "abcdef"


def test_debug_context_drops_unparseable_ids():
    ctx = DebugContext.deserialize(
        {"enabled": True, "session_id": "not-a-uuid", "replay_trace_id": "nope"}
    )
    assert ctx.session_id is None
    assert ctx.replay_trace_id is None


def test_laminar_span_context_parses_nested_debug():
    sc = LaminarSpanContext.deserialize(
        {
            "traceId": _SESSION,
            "spanId": _REPLAY,
            "debug": {"enabled": True, "sessionId": _SESSION},
        }
    )
    assert sc.debug is not None
    assert sc.debug.enabled is True
    assert sc.debug.session_id == _SESSION


def test_laminar_span_context_no_debug_is_none():
    sc = LaminarSpanContext.deserialize({"traceId": _SESSION, "spanId": _REPLAY})
    assert sc.debug is None


# --- from-context config builder (inherited path) ---


def test_build_from_context_armed():
    config = build_debug_config_from_context(
        DebugContext(
            enabled=True,
            session_id=_SESSION,
            replay_trace_id=_REPLAY,
            cache_until="0123-456789abcdef",
        )
    )
    assert config is not None
    assert config.session_id == _SESSION
    assert config.replay_trace_id == _REPLAY
    assert config.cache_until_span_id == "0123456789abcdef"
    assert config.local_origin is False
    assert config.session_minted is False
    assert config.replay_enabled is True


def test_build_from_context_none_when_disabled():
    assert (
        build_debug_config_from_context(
            DebugContext(enabled=False, session_id=_SESSION)
        )
        is None
    )


def test_build_from_context_none_when_no_session():
    assert (
        build_debug_config_from_context(DebugContext(enabled=True, session_id=None))
        is None
    )


def test_build_from_context_none_when_block_absent():
    assert build_debug_config_from_context(None) is None

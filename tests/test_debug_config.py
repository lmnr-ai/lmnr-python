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
)

_VECTORS = json.loads(
    (Path(__file__).parent / "data" / "debug" / "config_truth_table.json").read_text()
)["cases"]


@pytest.fixture(autouse=True)
def _clear_debug_env(monkeypatch):
    for key in _DEBUG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("case", _VECTORS, ids=[c["name"] for c in _VECTORS])
def test_config_truth_table(case, tmp_path, monkeypatch):
    # build_debug_config now reads `.lmnr/debug-session.json` from cwd
    # unconditionally; pin cwd to an empty temp dir so a stray file can't leak in.
    monkeypatch.chdir(tmp_path)
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


def test_session_id_defaults_to_uuid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMNR_DEBUG", "true")
    config = build_debug_config()
    assert config is not None
    # A generated uuid4 is 36 chars with 4 hyphens.
    assert len(config.session_id) == 36
    assert config.session_id.count("-") == 4


def test_disabled_when_env_absent():
    assert os.environ.get("LMNR_DEBUG") is None
    assert build_debug_config() is None


def _write_session_file(tmp_path: Path, payload: dict) -> None:
    session_dir = tmp_path / ".lmnr"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "debug-session.json").write_text(json.dumps(payload))


def test_session_file_rejoins_silently_continuation_not_minted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_session_file(
        tmp_path,
        {
            "session_id": "session-xyz",
            "trace_id": "trace-abc",
            "replay_trace_id": None,
            "cache_until": "0123-456789abcdef",
            "debugger_url": None,
            "started_at": "2026-01-01T00:00:00.000Z",
        },
    )
    monkeypatch.setenv("LMNR_DEBUG", "true")

    config = build_debug_config()
    assert config is not None
    # The session id comes from the file; the browser must NOT reopen.
    assert config.session_id == "session-xyz"
    assert config.session_minted is False
    assert config.local_origin is True
    # Continuation is NOT replay: the prior run's trace_id is NOT promoted into
    # replay_trace_id. Only an explicit replay_trace_id (file or env) arms replay.
    assert config.replay_trace_id is None
    # cache_until is read from the file when the env var is unset.
    assert config.cache_until_span_id == "0123456789abcdef"


def test_session_file_found_in_ancestor_joins_its_session(tmp_path, monkeypatch):
    # Nearest-ancestor resolution: a run started from a subdirectory of a
    # project joins the project's session, not a fresh one.
    _write_session_file(
        tmp_path,
        {
            "session_id": "session-ancestor",
            "trace_id": "trace-abc",
            "replay_trace_id": None,
            "cache_until": None,
            "debugger_url": None,
            "started_at": "2026-01-01T00:00:00.000Z",
        },
    )
    nested = tmp_path / "packages" / "app"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.setenv("LMNR_DEBUG", "true")

    config = build_debug_config()
    assert config is not None
    assert config.session_id == "session-ancestor"
    assert config.session_minted is False


def test_config_pins_session_dir_and_file_session_id(tmp_path, monkeypatch):
    # The anchor and the on-disk session id are captured at init for the
    # emit-side write/guard: session_dir is the resolved ancestor (chdir-safe
    # write target) and file_session_id is what the guard compares against
    # (so an env override still persists to an unchanged file).
    _write_session_file(
        tmp_path,
        {
            "session_id": "session-on-disk",
            "trace_id": None,
            "replay_trace_id": None,
            "cache_until": None,
            "debugger_url": None,
            "started_at": "2026-01-01T00:00:00.000Z",
        },
    )
    nested = tmp_path / "packages" / "app"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_DEBUG_SESSION_ID", "session-from-env")

    config = build_debug_config()
    assert config is not None
    assert config.session_id == "session-from-env"
    assert config.session_dir == str(tmp_path)
    assert config.file_session_id == "session-on-disk"


def test_session_file_reads_replay_and_cache_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_session_file(
        tmp_path,
        {
            "session_id": "session-xyz",
            "trace_id": "trace-abc",
            "replay_trace_id": "replay-from-file",
            "cache_until": "abcdef",
            "debugger_url": None,
            "started_at": "2026-01-01T00:00:00.000Z",
        },
    )
    monkeypatch.setenv("LMNR_DEBUG", "true")

    config = build_debug_config()
    assert config is not None
    assert config.replay_trace_id == "replay-from-file"
    assert config.cache_until_span_id == "abcdef"
    assert config.replay_enabled is True


def test_env_overrides_the_file_per_field(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_session_file(
        tmp_path,
        {
            "session_id": "session-xyz",
            "trace_id": "trace-abc",
            "replay_trace_id": "replay-from-file",
            "cache_until": "0123-456789abcdef",
            "debugger_url": None,
            "started_at": "2026-01-01T00:00:00.000Z",
        },
    )
    monkeypatch.setenv("LMNR_DEBUG", "true")
    monkeypatch.setenv("LMNR_DEBUG_SESSION_ID", "env-session")
    monkeypatch.setenv("LMNR_DEBUG_REPLAY_TRACE_ID", "env-replay")
    monkeypatch.setenv("LMNR_DEBUG_CACHE_UNTIL", "cafe")

    config = build_debug_config()
    assert config is not None
    # Explicit env session id wins over the file and is still a continuation.
    assert config.session_id == "env-session"
    assert config.session_minted is False
    assert config.replay_trace_id == "env-replay"
    assert config.cache_until_span_id == "cafe"


def test_mints_fresh_session_when_no_file_and_no_env_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMNR_DEBUG", "true")

    config = build_debug_config()
    assert config is not None
    assert len(config.session_id) == 36
    assert config.session_minted is True
    assert config.local_origin is True
    assert config.replay_trace_id is None


def test_local_origin_and_session_minted_from_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LMNR_DEBUG", "true")
    config = build_debug_config()
    assert config is not None
    assert config.local_origin is True
    assert config.session_minted is True


def test_provided_session_id_not_minted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
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


def test_debug_context_keeps_non_uuid_ids_verbatim():
    # `LMNR_DEBUG_SESSION_ID` may be an arbitrary string; the origin registers
    # and propagates that exact value, so the consumer must round-trip it
    # unchanged. Dropping non-UUID ids to None would make the downstream treat
    # the block as session-less and never join the run.
    ctx = DebugContext.deserialize(
        {
            "enabled": True,
            "session_id": "my-session",
            "replay_trace_id": "my-replay",
        }
    )
    assert ctx.session_id == "my-session"
    assert ctx.replay_trace_id == "my-replay"


def test_debug_context_empty_ids_become_none():
    ctx = DebugContext.deserialize(
        {"enabled": True, "session_id": "", "replay_trace_id": ""}
    )
    assert ctx.session_id is None
    assert ctx.replay_trace_id is None


def test_debug_context_non_boolean_enabled_never_arms():
    # The producer always emits a real boolean. A truthy non-True value (e.g.
    # the string "false", or 1) is a malformed/forged block and must parse to
    # enabled=False, never arming a downstream runtime.
    for enabled in ("false", "true", 1, {"x": 1}):
        ctx = DebugContext.deserialize(
            {"enabled": enabled, "session_id": "my-session"}
        )
        assert ctx.enabled is False


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

import json
import os

from lmnr.sdk.debug.pointer import (
    CONSOLE_PREFIX,
    POINTER_DIR,
    POINTER_FILE,
    build_pointer,
    emit_pointer,
)


def test_build_pointer_key_order_and_values():
    pointer = build_pointer(
        trace_id="trace-1",
        session_id="sess-1",
        replay_trace_id="replay-1",
        cache_until=3,
        debugger_url="https://www.lmnr.ai",
    )
    # Key order is part of the cross-language parity contract.
    assert list(pointer.keys()) == [
        "trace_id",
        "session_id",
        "replay_trace_id",
        "cache_until",
        "debugger_url",
        "started_at",
    ]
    assert pointer["trace_id"] == "trace-1"
    assert pointer["session_id"] == "sess-1"
    assert pointer["replay_trace_id"] == "replay-1"
    assert pointer["cache_until"] == 3
    assert pointer["debugger_url"] == "https://www.lmnr.ai"
    assert isinstance(pointer["started_at"], str) and pointer["started_at"]


def test_emit_pointer_prints_prefixed_compact_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    pointer = build_pointer("t", "s", None, 0, None)
    emit_pointer(pointer)

    out = capsys.readouterr().out.strip()
    assert out.startswith(CONSOLE_PREFIX)
    payload = out[len(CONSOLE_PREFIX):]
    # Compact JSON: no spaces after separators.
    assert ", " not in payload and ": " not in payload
    assert json.loads(payload) == pointer


def test_emit_pointer_writes_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pointer = build_pointer("t", "s", "r", 2, "https://www.lmnr.ai")
    emit_pointer(pointer)

    path = os.path.join(tmp_path, POINTER_DIR, POINTER_FILE)
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == pointer


def test_emit_pointer_best_effort_on_unwritable_dir(tmp_path, monkeypatch, capsys):
    # Make the working directory's .lmnr path un-creatable by pointing CWD at a
    # location where makedirs raises; emit_pointer must still print and not raise.
    monkeypatch.chdir(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("lmnr.sdk.debug.pointer.os.makedirs", boom)
    pointer = build_pointer("t", "s", None, 0, None)

    emit_pointer(pointer)  # must not raise

    out = capsys.readouterr().out.strip()
    assert out.startswith(CONSOLE_PREFIX)

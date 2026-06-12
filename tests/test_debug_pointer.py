import json
import os

from lmnr.sdk.debug.debug_session_file import (
    DEBUG_SESSION_DIR,
    DEBUG_SESSION_FILE,
    find_debug_session_dir,
    read_debug_session_file,
    resolve_debug_session_dir,
    write_debug_session_file,
)
from lmnr.sdk.debug.pointer import (
    CONSOLE_PREFIX,
    build_debug_session_file,
    emit_pointer,
)


def test_build_debug_session_file_key_order_and_values():
    file = build_debug_session_file(
        session_id="sess-1",
        trace_id="trace-1",
        replay_trace_id="replay-1",
        cache_until="0123456789abcdef",
        debugger_url="https://www.lmnr.ai",
    )
    # Key order is part of the cross-language parity contract; session_id is now
    # first AND primary.
    assert list(file.keys()) == [
        "session_id",
        "trace_id",
        "replay_trace_id",
        "cache_until",
        "debugger_url",
        "started_at",
    ]
    assert file["session_id"] == "sess-1"
    assert file["trace_id"] == "trace-1"
    assert file["replay_trace_id"] == "replay-1"
    assert file["cache_until"] == "0123456789abcdef"
    assert file["debugger_url"] == "https://www.lmnr.ai"
    assert isinstance(file["started_at"], str) and file["started_at"]


def test_build_debug_session_file_keeps_trace_id_null_for_fresh_session():
    file = build_debug_session_file(
        session_id="sess-1",
        trace_id=None,
        replay_trace_id=None,
        cache_until=None,
        debugger_url=None,
    )
    assert file["trace_id"] is None


def test_emit_prints_prefixed_compact_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    file = build_debug_session_file("s", "t", None, None, None)
    emit_pointer(file)

    out = capsys.readouterr().out.strip()
    assert out.startswith(CONSOLE_PREFIX)
    payload = out[len(CONSOLE_PREFIX):]
    # Compact JSON: no spaces after separators.
    assert ", " not in payload and ": " not in payload
    assert json.loads(payload) == file


def test_emit_always_writes_the_debug_session_file(tmp_path, monkeypatch):
    # Default-on now: no env gate. The file is always written.
    monkeypatch.chdir(tmp_path)
    file = build_debug_session_file(
        "s", "t", "r", "abcdef", "https://www.lmnr.ai"
    )
    emit_pointer(file)

    path = os.path.join(tmp_path, DEBUG_SESSION_DIR, DEBUG_SESSION_FILE)
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == file


def test_emit_best_effort_on_unwritable_dir(tmp_path, monkeypatch, capsys):
    # Make the working directory's .lmnr path un-creatable by making makedirs
    # raise; emit must still print and not raise.
    monkeypatch.chdir(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(
        "lmnr.sdk.debug.debug_session_file.os.makedirs", boom
    )
    file = build_debug_session_file("s", "t", None, None, None)

    emit_pointer(file)  # must not raise

    out = capsys.readouterr().out.strip()
    assert out.startswith(CONSOLE_PREFIX)


# --- read_debug_session_file / write_debug_session_file (shared helpers) ---

_SAMPLE = {
    "session_id": "sess-rt",
    "trace_id": "trace-rt",
    "replay_trace_id": "replay-rt",
    "cache_until": "abcdef",
    "debugger_url": "https://app.x/project/p/debugger-sessions/sess-rt",
    "started_at": "2026-06-11T00:00:00.000Z",
}


def _write_raw(directory, contents: str) -> None:
    target = os.path.join(directory, DEBUG_SESSION_DIR)
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, DEBUG_SESSION_FILE), "w", encoding="utf-8") as f:
        f.write(contents)


def test_session_file_round_trips_write_then_read(tmp_path):
    wrote = write_debug_session_file(_SAMPLE, str(tmp_path))
    assert wrote is True
    assert read_debug_session_file(str(tmp_path)) == _SAMPLE


def test_read_returns_none_for_missing_file(tmp_path):
    assert read_debug_session_file(str(tmp_path)) is None


def test_read_returns_none_for_malformed_file(tmp_path):
    _write_raw(str(tmp_path), "not json")
    assert read_debug_session_file(str(tmp_path)) is None


def test_read_returns_none_when_session_id_missing(tmp_path):
    _write_raw(str(tmp_path), json.dumps({"trace_id": "t"}))
    assert read_debug_session_file(str(tmp_path)) is None


def test_read_coerces_non_string_fields_to_none(tmp_path):
    _write_raw(
        str(tmp_path),
        json.dumps(
            {"session_id": "s", "trace_id": 123, "started_at": "2026-01-01T00:00:00Z"}
        ),
    )
    read = read_debug_session_file(str(tmp_path))
    assert read is not None
    assert read["session_id"] == "s"
    assert read["trace_id"] is None
    assert read["replay_trace_id"] is None
    assert read["cache_until"] is None
    assert read["debugger_url"] is None


# --- find_debug_session_dir / resolve_debug_session_dir (nearest-ancestor anchor) ---


def test_find_returns_none_when_no_ancestor_has_a_session(tmp_path):
    nested = tmp_path / "packages" / "app"
    nested.mkdir(parents=True)
    assert find_debug_session_dir(str(nested)) is None


def test_find_walks_up_to_the_nearest_ancestor_with_a_session(tmp_path):
    write_debug_session_file(_SAMPLE, str(tmp_path))
    nested = tmp_path / "packages" / "app"
    nested.mkdir(parents=True)
    assert find_debug_session_dir(str(nested)) == str(tmp_path)


def test_find_prefers_the_nearest_ancestor(tmp_path):
    write_debug_session_file(_SAMPLE, str(tmp_path))
    mid = tmp_path / "packages"
    nested = mid / "app"
    nested.mkdir(parents=True)
    write_debug_session_file(dict(_SAMPLE, session_id="sess-closer"), str(mid))
    assert find_debug_session_dir(str(nested)) == str(mid)


def test_find_ignores_an_unusable_file_and_keeps_walking(tmp_path):
    write_debug_session_file(_SAMPLE, str(tmp_path))
    nested = tmp_path / "packages" / "app"
    nested.mkdir(parents=True)
    # A malformed nested file (no usable session_id) must not stop the walk.
    _write_raw(str(nested), "not json")
    assert find_debug_session_dir(str(nested)) == str(tmp_path)


def test_resolve_falls_back_to_start_dir(tmp_path):
    nested = tmp_path / "packages" / "app"
    nested.mkdir(parents=True)
    assert resolve_debug_session_dir(str(nested)) == str(nested)


def test_emit_from_subdirectory_writes_back_to_the_ancestor_anchor(
    tmp_path, monkeypatch, capsys
):
    existing = build_debug_session_file("s", None, None, None, None)
    write_debug_session_file(existing, str(tmp_path))
    nested = tmp_path / "packages" / "app"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    updated = build_debug_session_file("s", "t-new", None, None, None)
    emit_pointer(updated)

    # The ancestor file was updated in place; no nested copy was created.
    ancestor_path = tmp_path / DEBUG_SESSION_DIR / DEBUG_SESSION_FILE
    with open(ancestor_path, encoding="utf-8") as f:
        assert json.load(f) == updated
    assert not (nested / DEBUG_SESSION_DIR).exists()
    capsys.readouterr()


def test_emit_skips_the_write_when_on_disk_session_differs(
    tmp_path, monkeypatch, capsys
):
    # A fresher session minted on disk while this run was in flight (e.g.
    # `lmnr-cli debug session new`) must not be clobbered.
    fresher = build_debug_session_file("sess-fresher", None, None, None, None)
    write_debug_session_file(fresher, str(tmp_path))
    monkeypatch.chdir(tmp_path)

    ours = build_debug_session_file("sess-ours", "t", None, None, None)
    emit_pointer(ours)

    # The console marker is always printed; the fresher file is preserved.
    out = capsys.readouterr().out.strip()
    assert out.startswith(CONSOLE_PREFIX)
    assert read_debug_session_file(str(tmp_path)) == fresher

"""Shared `.lmnr/debug-session.json` contract + best-effort fs helpers.

Single source of truth for the persisted debug-session record's SHAPE and
location. The SDK reads it at init (to decide "join existing session vs. mint a
new one") and writes it at shutdown.

Mirrors the TS split: the `DebugSessionFile` shape + filename consts live in
`@lmnr-ai/types` (`packages/types/src/debug-session.ts`), and the sync read/write
helpers live in the SDK's `src/debug/debug-session-file.ts`. Python is a single
package, so both halves live here. Keep the read/write helpers line-comparable
with the TS `readDebugSessionFile` / `writeDebugSessionFile`.

Persisted record at `${CWD}/.lmnr/debug-session.json`. Replaces the old
`.lmnr/last-run.json` pointer; it is the default persistence for a debug run (no
opt-in env var). Field order is `{session_id, trace_id, replay_trace_id,
cache_until, debugger_url, started_at}` — `session_id` is now FIRST AND primary
(the field read at startup), and `trace_id` is `str | None` (a freshly-minted
session has no trace yet).
"""

import datetime
import json
import os
from typing import Any

# Directory the debug-session file lives in, relative to the working dir.
DEBUG_SESSION_DIR = ".lmnr"
# Filename of the debug-session file inside DEBUG_SESSION_DIR.
DEBUG_SESSION_FILE = "debug-session.json"


def _str(value: Any) -> str | None:
    """Coerce an unknown field to a non-empty string, else None.

    The file is best-effort local state that an agent may hand-edit, so every
    field is treated defensively rather than trusted.
    """
    return value if isinstance(value, str) and len(value) > 0 else None


def read_debug_session_file(directory: str | None = None) -> dict[str, Any] | None:
    """Read `${dir ?? cwd}/.lmnr/debug-session.json`.

    Best-effort: returns None on a missing / unreadable / malformed file, or one
    with no usable `session_id` (the caller treats None as "no existing session"
    and mints a fresh one). Keep line-comparable with the TS `readDebugSessionFile`.
    """
    directory = directory if directory is not None else os.getcwd()
    try:
        path = os.path.join(directory, DEBUG_SESSION_DIR, DEBUG_SESSION_FILE)
        with open(path, "r", encoding="utf-8") as f:
            r = json.load(f)
        if not isinstance(r, dict):
            return None
        session_id = _str(r.get("session_id"))
        if not session_id:
            return None
        return {
            "session_id": session_id,
            "trace_id": _str(r.get("trace_id")),
            "replay_trace_id": _str(r.get("replay_trace_id")),
            "cache_until": _str(r.get("cache_until")),
            "debugger_url": _str(r.get("debugger_url")),
            "started_at": (
                _str(r.get("started_at"))
                or datetime.datetime.now(datetime.timezone.utc).isoformat()
            ),
        }
    except Exception:
        return None


def find_debug_session_dir(start_dir: str | None = None) -> str | None:
    """Find the nearest directory (walking up from `start_dir` to the
    filesystem root) whose `.lmnr/debug-session.json` holds a usable session
    record, so a debug run started from a subdirectory of a project joins the
    project's session. Returns None when no ancestor (including `start_dir`)
    has one. Keep line-comparable with the TS `findDebugSessionDir`.
    """
    directory = os.path.abspath(start_dir if start_dir is not None else os.getcwd())
    while True:
        if read_debug_session_file(directory) is not None:
            return directory
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def resolve_debug_session_dir(start_dir: str | None = None) -> str:
    """The directory the debug-session file should be read from AND written to:
    the nearest ancestor (incl. `start_dir`) that already has one, else
    `start_dir` itself. Read and write MUST share this anchor — reading from an
    ancestor but writing to cwd would strand the ancestor's copy stale and
    shadow it with a nested one. Keep line-comparable with the TS
    `resolveDebugSessionDir`.
    """
    found = find_debug_session_dir(start_dir)
    if found is not None:
        return found
    return os.path.abspath(start_dir if start_dir is not None else os.getcwd())


def write_debug_session_file(
    file: dict[str, Any], directory: str | None = None
) -> bool:
    """Write the debug-session file (mkdir -p first). Best-effort.

    Swallows any IO error so a read-only working directory never breaks a run.
    Keep line-comparable with the TS `writeDebugSessionFile`.
    """
    directory = directory if directory is not None else os.getcwd()
    try:
        target_dir = os.path.join(directory, DEBUG_SESSION_DIR)
        os.makedirs(target_dir, exist_ok=True)
        with open(
            os.path.join(target_dir, DEBUG_SESSION_FILE), "w", encoding="utf-8"
        ) as f:
            f.write(json.dumps(file, separators=(",", ":")))
        return True
    except Exception:
        return False

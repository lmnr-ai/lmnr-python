"""Emit the debug-session record (§5, §I).

On a debug run we always print one console line prefixed with `LMNR_DEBUG_RUN `
followed by compact JSON, AND write the same payload to
`${CWD}/.lmnr/debug-session.json` (default-on now — there is no longer an opt-in
env gate). The file write is best-effort (any IO error is swallowed so a
read-only working directory never breaks the run) and delegates to the local
`write_debug_session_file` helper, which writes the SAME shape (the
`DebugSessionFile` contract + filename consts) the rest of the tooling reads, so
the on-disk file stays consistent across writers.

Part of the cross-language parity surface — keep line-comparable with the TS
`pointer.ts` (identical prefix and best-effort semantics).
"""

import datetime
import json
from typing import Any

from lmnr.sdk.debug.debug_session_file import write_debug_session_file

# Console marker the orchestrating tooling greps for. Must match the TS SDK.
CONSOLE_PREFIX = "LMNR_DEBUG_RUN "


def build_debug_session_file(
    session_id: str,
    trace_id: str | None,
    replay_trace_id: str | None,
    cache_until: str | None,
    debugger_url: str | None,
    started_at: str | None = None,
) -> dict[str, Any]:
    """Build the persisted debug-session record. Key order matches the TS SDK.

    `started_at` is the run's start time, captured by `DebugRuntime` at SDK init
    so the record reflects when tracing began rather than when it was emitted
    (shutdown). Defaults to now for standalone callers.
    """
    return {
        "session_id": session_id,
        "trace_id": trace_id,
        "replay_trace_id": replay_trace_id,
        "cache_until": cache_until,
        "debugger_url": debugger_url,
        "started_at": (
            started_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
        ),
    }


def emit_pointer(file: dict[str, Any]) -> None:
    """Print the console line, then best-effort write the debug-session file."""
    print(f"{CONSOLE_PREFIX}{json.dumps(file, separators=(',', ':'))}", flush=True)
    write_debug_session_file(file)

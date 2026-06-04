"""Emit the debug-run pointer (§5, §I).

On startup of a debug run we always print one console line prefixed with
`LMNR_DEBUG_RUN ` followed by compact JSON, then best-effort write the same
payload to `${CWD}/.lmnr/last-run.json`. The file write is best-effort: any IO
error is swallowed so a read-only working directory never breaks the run.

Part of the cross-language parity surface — keep line-comparable with the TS
`pointer.ts` (identical key order, prefix, and best-effort semantics).
"""

import datetime
import json
import os
from typing import Any

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

# Console marker the orchestrating tooling greps for. Must match the TS SDK.
CONSOLE_PREFIX = "LMNR_DEBUG_RUN "

# Pointer file location, relative to the current working directory.
POINTER_DIR = ".lmnr"
POINTER_FILE = "last-run.json"


def build_pointer(
    trace_id: str,
    session_id: str,
    replay_trace_id: str | None,
    cache_until: int,
    debugger_url: str | None,
    started_at: str | None = None,
) -> dict[str, Any]:
    """Build the pointer payload. Key order matches the TS SDK.

    `started_at` is the run's start time, captured by `DebugRuntime` at SDK init
    so the pointer reflects when tracing began rather than when it was emitted
    (shutdown). Defaults to now for standalone callers.
    """
    return {
        "trace_id": trace_id,
        "session_id": session_id,
        "replay_trace_id": replay_trace_id,
        "cache_until": cache_until,
        "debugger_url": debugger_url,
        "started_at": (
            started_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
        ),
    }


def emit_pointer(pointer: dict[str, Any]) -> None:
    """Print the console line, then best-effort write the pointer file."""
    payload = json.dumps(pointer, separators=(",", ":"))
    print(f"{CONSOLE_PREFIX}{payload}", flush=True)
    if os.getenv("LMNR_DEBUG_WRITE_LAST_RUN_TO_FILE"):
        _write_pointer_file(payload)


def _write_pointer_file(payload: str) -> None:
    try:
        directory = os.path.join(os.getcwd(), POINTER_DIR)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, POINTER_FILE), "w", encoding="utf-8") as f:
            f.write(payload)
    except Exception as exc:  # pragma: no cover - best-effort only
        logger.debug("Could not write debug pointer file: %s", exc)

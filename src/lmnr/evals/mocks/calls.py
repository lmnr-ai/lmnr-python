"""`ctx.mocks` in `verify`: the mock server's call log, read on the host."""

from __future__ import annotations

import json
from typing import Any

from lmnr.evals.artifacts import ARTIFACTS_DIR, Artifacts
from lmnr.evals.errors import EvalConfigError

# Written into /logs/artifacts/ in `collect`, one JSON object per call.
MOCK_LOG = "mocks.jsonl"


class MockCalls:
    """Every call the app made to the runner's mocks, oldest first. Each is a
    dict: `seq`, `ts`, `type` (`function` / `http`), `key`, `index`,
    `request`, `outcome` (`respond`, `raise`, `passthrough`, `unmatched`,
    `handler_error`), `response`, `duration_ms`, `trace_id`, `span_id`.

    Read from the trial's artifacts, so it works in both verifier modes; code
    in a separate grading container finds the file at
    `/logs/artifacts/mocks.jsonl`.
    """

    def __init__(self, artifacts: Artifacts, keys: list[str]):
        self._artifacts: Artifacts = artifacts
        self._keys: list[str] = list(keys)

    def calls(self, key: str) -> list[dict[str, Any]]:
        """The calls of one mock, by its key in `mocks`."""
        if key not in self._keys:
            declared = ", ".join(repr(k) for k in self._keys) or "none"
            raise KeyError(f"{key!r} is not in the runner's mocks (mocks: {declared}).")
        return [call for call in self.all() if call.get("key") == key]

    def unmatched(self) -> list[dict[str, Any]]:
        """Requests no handler answered: HTTP mocks that returned None, and
        function keys the server doesn't have."""
        return [call for call in self.all() if call.get("outcome") == "unmatched"]

    def all(self) -> list[dict[str, Any]]:
        if not self._keys:
            raise EvalConfigError("The runner declares no mocks, so there is no call log.")
        path = f"{ARTIFACTS_DIR}/{MOCK_LOG}"
        if not self._artifacts.exists(path):
            raise FileNotFoundError(
                f"No {path} in this trial's artifacts: collect didn't save the mock call log "
                + "(see the agent phase's error)."
            )
        text = self._artifacts.read_text(path)
        return [json.loads(line) for line in text.splitlines() if line.strip()]

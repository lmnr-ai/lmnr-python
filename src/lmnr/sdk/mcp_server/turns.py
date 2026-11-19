"""Trace-id selection: one trace per user turn.

First that applies (contract shared with the TS SDK and app-server):
1. the client's turn id (`_meta["io.modelcontextprotocol/aiInvocation"].turnId`),
   hashed with the session id, so every server copy picks the same trace;
2. the trace id of a valid W3C `traceparent` in `_meta` (the span stays a root
   and links to the remote span);
3. the per-session `TurnTracker`: a call joins the session's current trace when
   another call of the session is in flight or the last activity was at most
   `turn_gap` ago;
4. a random trace id.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import INVALID_TRACE_ID

DEFAULT_TURN_GAP_SECONDS = 20.0
# Signals wait until a trace has been quiet for 30 s before analysing it, so a
# turn gap at or above that would let a Signal run mid-turn.
MAX_TURN_GAP_SECONDS = 25.0
MAX_TRACKED_SESSIONS = 10_000

_ids = RandomIdGenerator()


def turn_trace_id(session_id: str | None, turn_id: str) -> int:
    """The first 16 bytes of SHA-256(`"{session_id or ''}\\n{turn_id}"`)."""
    digest = hashlib.sha256(f"{session_id or ''}\n{turn_id}".encode()).digest()
    trace_id = int.from_bytes(digest[:16], "big")
    # an all-zero trace id is invalid; astronomically unlikely, but stay valid
    return trace_id if trace_id != INVALID_TRACE_ID else _ids.generate_trace_id()


@dataclass
class _Turn:
    trace_id: int
    last_activity: float
    in_flight: int = 0


class TurnTracker:
    """In-process turn state per session id. Thread-safe.

    Every call `begin`s and `end`s; both count as activity. Idle sessions are
    evicted lazily, and the map is capped at `max_sessions`.
    """

    def __init__(
        self,
        turn_gap_seconds: float = DEFAULT_TURN_GAP_SECONDS,
        max_sessions: int = MAX_TRACKED_SESSIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.turn_gap_seconds: float = turn_gap_seconds
        self._max_sessions: int = max_sessions
        self._clock: Callable[[], float] = clock
        self._lock: threading.Lock = threading.Lock()
        self._turns: dict[str, _Turn] = {}

    def begin(self, session_id: str, trace_id: int | None = None) -> int:
        """Start a call of `session_id` and return its trace id.

        `trace_id` (from a turn id or a `traceparent`) wins over the tracker
        and becomes the session's current trace, so later calls of the same
        turn that don't carry one still join it.
        """
        with self._lock:
            now = self._clock()
            turn = self._turns.get(session_id)
            if turn is None:
                self._evict(now)
                turn = _Turn(trace_id=trace_id or _ids.generate_trace_id(), last_activity=now)
                self._turns[session_id] = turn
            elif trace_id is not None:
                turn.trace_id = trace_id
            elif turn.in_flight == 0 and now - turn.last_activity > self.turn_gap_seconds:
                turn.trace_id = _ids.generate_trace_id()
            turn.in_flight += 1
            turn.last_activity = now
            return turn.trace_id

    def end(self, session_id: str) -> None:
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is None:
                return
            turn.in_flight = max(0, turn.in_flight - 1)
            turn.last_activity = self._clock()

    def __len__(self) -> int:
        return len(self._turns)

    def _evict(self, now: float) -> None:
        # called with the lock held, before adding a session
        if len(self._turns) < self._max_sessions:
            return
        for key in [
            key
            for key, turn in self._turns.items()
            if turn.in_flight == 0 and now - turn.last_activity > self.turn_gap_seconds
        ]:
            del self._turns[key]
        if len(self._turns) < self._max_sessions:
            return
        # still full: drop the least recently active idle sessions, then the oldest
        by_age = sorted(
            self._turns.items(), key=lambda item: (item[1].in_flight > 0, item[1].last_activity)
        )
        for key, _ in by_age[: len(self._turns) - self._max_sessions + 1]:
            del self._turns[key]

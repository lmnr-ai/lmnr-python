"""Collect spans into an lmnr-atif trajectory file (`LMNR_TRAJECTORY_PATH`).

`LaminarSpanProcessor` feeds every span it would export to a
`TrajectoryCollector`, which keeps a `SpanRecord` per finished span and
rewrites the file:

- on `force_flush` / `shutdown` (so `Laminar.flush()`, `Laminar.shutdown()`
  and the atexit flush all leave a final file), and
- as a snapshot, at most every `snapshot_interval` seconds while spans keep
  ending, so an agent killed on a timeout still leaves a recent file. A file
  written while spans were open has `complete: false`.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
import weakref
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, cast

from opentelemetry.sdk.trace import ReadableSpan, Span

from lmnr.opentelemetry_lib.tracing.attributes import SPAN_INPUT, SPAN_OUTPUT, SPAN_TYPE
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.trajectory.builder import BuildOptions, SpanRecord, format_time
from lmnr.sdk.trajectory.writer import TRACE_ID, write_trajectories

LOG = get_default_logger(__name__)

DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 2.0

_SESSION_ID = "lmnr.association.properties.session_id"
_RELEASE = "lmnr.span.release"
# Legacy indexed prompt attributes: only the system prompt is read (the OpenAI
# Responses API records `instructions` nowhere else).
_LEGACY_SYSTEM = ("gen_ai.prompt.0.role", "gen_ai.prompt.0.content")
# The AI SDK's TOOL span attributes the builder matches tool calls by (the
# TypeScript SDK records them; kept here so both collectors stay alike).
_AI_SDK_TOOL_CALL = ("ai.toolCall.id", "ai.toolCall.name")


# app-server's `SpanType` enum (`db/spans.rs`).
_SPAN_TYPES = frozenset(
    (
        "DEFAULT",
        "LLM",
        "PIPELINE",
        "EXECUTOR",
        "EVALUATOR",
        "HUMAN_EVALUATOR",
        "EVALUATION",
        "TOOL",
        "CACHED",
    )
)
_OPENROUTER_SPAN_TYPE = "span.type"
_OPENROUTER_METADATA_PREFIX = "trace.metadata.openrouter."


def span_type(attributes: Mapping[str, Any], name: str = "") -> str:
    """The span's type as app-server sees it (`traces/spans.rs`), so the file
    and the Laminar UI agree on what is an LLM or tool call: `span_type()`,
    plus the two rules applied after it, `is_ai_sdk_tool_call_span` (TOOL) and
    the cached-LLM case of `is_llm_span` (a debug-replay hit is still an LLM
    call)."""
    if name == "ai.toolCall" or any(k in attributes for k in _AI_SDK_TOOL_CALL):
        return "TOOL"
    kind = _base_span_type(attributes)
    if kind == "CACHED" and attributes.get("lmnr.span.original_type") == "LLM":
        return "LLM"
    return kind


def _base_span_type(attributes: Mapping[str, Any]) -> str:
    """`span_type()` in app-server's `traces/spans.rs`."""
    if SPAN_TYPE in attributes:
        explicit = attributes[SPAN_TYPE]
        return explicit if isinstance(explicit, str) and explicit in _SPAN_TYPES else "DEFAULT"
    # OpenRouter Broadcast types its own spans.
    openrouter_type = attributes.get(_OPENROUTER_SPAN_TYPE)
    if (
        isinstance(openrouter_type, str)
        and openrouter_type != "generation"
        and any(k.startswith(_OPENROUTER_METADATA_PREFIX) for k in attributes)
    ):
        return "DEFAULT"
    operation = attributes.get("gen_ai.operation.name")
    if operation in ("chat", "text_completion", "embeddings", "generate_content"):
        return "LLM"
    if operation == "execute_tool":
        return "TOOL"
    if operation == "invoke_agent":
        return "DEFAULT"
    if "gen_ai.tool.call.arguments" in attributes or "gen_ai.tool.call.result" in attributes:
        return "TOOL"
    if any(
        k in attributes for k in ("gen_ai.system", "gen_ai.request.model", "gen_ai.response.model")
    ):
        return "LLM"
    return "DEFAULT"


def _keep(key: str, attributes: Mapping[str, Any], kind: str) -> bool:
    """Whether the builder reads this attribute. Records live as long as the
    process, so whole prompts are not kept twice."""
    if key.startswith(("gen_ai.prompt.", "gen_ai.completion.")):
        return key in _LEGACY_SYSTEM
    if key.startswith("gen_ai."):
        return True
    if key == SPAN_INPUT:
        return not (kind == "LLM" and "gen_ai.input.messages" in attributes)
    if key == SPAN_OUTPUT:
        return not (kind == "LLM" and "gen_ai.output.messages" in attributes)
    return key in (_SESSION_ID, _RELEASE, SPAN_TYPE, *_AI_SDK_TOOL_CALL)


def _value(value: Any) -> Any:
    # OTel stores sequences as tuples; the record format is JSON.
    return list(cast(tuple[Any, ...], value)) if isinstance(value, tuple) else value


def _error(span: ReadableSpan) -> dict[str, str] | None:
    """`{type, message[, stacktrace]}` from the span's last `exception` event.
    Like app-server, a span without one didn't fail."""
    for event in reversed(span.events or ()):
        if event.name != "exception":
            continue
        attributes = event.attributes or {}
        error = {
            "type": str(attributes.get("exception.type") or "Error"),
            "message": str(attributes.get("exception.message") or ""),
        }
        if attributes.get("exception.stacktrace"):
            error["stacktrace"] = str(attributes["exception.stacktrace"])
        return error
    return None


def span_record(span: ReadableSpan, end_time_ns: int | None = None) -> SpanRecord | None:
    """The builder's view of a span. `end_time_ns` stands in for the end of a
    span that is still open. None if the span can't be read."""
    context = span.context
    start = span.start_time
    end = span.end_time if span.end_time is not None else end_time_ns
    if context is None or start is None or end is None:
        return None
    attributes = dict(span.attributes or {})
    kind = span_type(attributes, span.name)
    return SpanRecord(
        trace_id=str(uuid.UUID(int=context.trace_id)),
        span_id=str(uuid.UUID(int=context.span_id)),
        parent_span_id=str(uuid.UUID(int=span.parent.span_id)) if span.parent else None,
        name=span.name,
        span_type=kind,
        start_time=format_time(start),
        end_time=format_time(max(end, start)),
        attributes={k: _value(v) for k, v in attributes.items() if _keep(k, attributes, kind)},
        error=_error(span),
    )


@dataclass
class _EndingRun:
    """A run `new_run` ended while some of its spans were still open."""

    run: int
    records: dict[str, SpanRecord]
    # Ids of its spans that are still open.
    open: set[int]


class TrajectoryCollector:
    """Turns the spans of this process into lmnr-atif trajectory files.

    `path` may contain `{trace_id}` (one file per trace) and `{pid}`. Without
    `{trace_id}`, every trace of the process goes into one file.
    """

    def __init__(
        self,
        path: str,
        exported: bool,
        release: str | None = None,
        snapshot_interval: float = DEFAULT_SNAPSHOT_INTERVAL_SECONDS,
    ):
        self.path: str = path
        self.exported: bool = exported
        self.release: str | None = release
        self.snapshot_interval: float = snapshot_interval
        self._reset()
        # Fork handlers can't be unregistered: hold the collector weakly.
        # `register_at_fork` doesn't exist on Windows (no fork there either).
        if hasattr(os, "register_at_fork"):
            ref = weakref.WeakMethod(self._reset)
            os.register_at_fork(after_in_child=lambda: (method := ref()) and method())

    def _reset(self) -> None:
        """Fresh state. Also run in a forked child: locks held by another
        thread at fork time stay locked forever, and the parent's spans are the
        parent's to write."""
        self._lock: threading.Lock = threading.Lock()
        self._write_lock: threading.Lock = threading.Lock()
        self._records: dict[str, SpanRecord] = {}
        self._open: dict[int, Span] = {}
        # Runs are numbered; `new_run` starts the next one.
        self._run: int = 0
        # Ended runs whose spans are still open. Those spans stay out of later
        # runs, and once the last one ends, the run's files are rewritten
        # complete.
        self._endings: list[_EndingRun] = []
        # While a run is ending: each file's key (`_file_key`) mapped to the
        # latest run that recorded a span for it. An ended run doesn't rewrite
        # a file a later run has written.
        self._claimed: dict[str | None, int] = {}
        self._dirty: bool = False
        self._wake: threading.Event = threading.Event()
        self._stopped: bool = False
        self._thread: threading.Thread | None = None
        self._warned: bool = False

    def on_start(self, span: Span) -> None:
        if span.context is None:
            return
        with self._lock:
            self._open[span.context.span_id] = span

    def new_run(self) -> None:
        """Drop every span, finished or open, so the next file holds only what
        comes after (`Laminar.force_flush` between Lambda invocations). Write
        the file first: this run's spans are gone afterwards. A span still open
        here (an `@observe` handler that calls `force_flush` before returning)
        stays out of the next run. When the last of them ends, the ended run's
        files are rewritten complete, except those a later run has written."""
        with self._lock:
            if self._open:
                self._endings.append(_EndingRun(self._run, self._records, set(self._open)))
            self._run += 1
            self._records = {}
            self._open = {}
            self._dirty = False

    def forget(self, span: ReadableSpan) -> None:
        """Stop tracking a span that won't be recorded."""
        if span.context is None:
            return
        span_id = span.context.span_id
        ended: list[SpanRecord] | None = None
        with self._lock:
            _ = self._open.pop(span_id, None)
            ending = self._ending_run(span_id)
            if ending is not None:
                ended = self._end_span(ending, span_id, None)
        if ended:
            with self._write_lock:
                _ = self._write(ended, (), retry=False)

    def on_end(self, span: ReadableSpan) -> None:
        record = span_record(span)
        ended: list[SpanRecord] | None = None
        with self._lock:
            span_id = span.context.span_id if span.context is not None else None
            if span_id is not None:
                _ = self._open.pop(span_id, None)
            ending = self._ending_run(span_id) if span_id is not None else None
            if ending is not None and span_id is not None:
                ended = self._end_span(ending, span_id, record)
            elif record is not None:
                self._records[record.span_id] = record
                if self._endings:
                    self._claimed[self._file_key(record)] = self._run
                self._dirty = True
                if self._thread is None and not self._stopped and self.snapshot_interval > 0:
                    self._thread = threading.Thread(
                        target=self._snapshots, name="LmnrTrajectorySnapshot", daemon=True
                    )
                    self._thread.start()
        # `write` takes `_write_lock` before `_lock`, never the other way round.
        if ended:
            with self._write_lock:
                _ = self._write(ended, (), retry=False)

    def _file_key(self, record: SpanRecord) -> str | None:
        """Which file `record` goes into: its trace's, or the one shared file."""
        return record.trace_id if TRACE_ID in self.path else None

    def _ending_run(self, span_id: int) -> _EndingRun | None:
        """The ended run `span_id` is a still-open span of. Holds `_lock`."""
        return next((e for e in self._endings if span_id in e.open), None)

    def _end_span(
        self, ending: _EndingRun, span_id: int, record: SpanRecord | None
    ) -> list[SpanRecord] | None:
        """Record a span of an ended run. Once its last open span has ended,
        returns the records of its files that no later run has written.
        Holds `_lock`."""
        ending.open.discard(span_id)
        if record is not None:
            ending.records[record.span_id] = record
        if ending.open:
            return None
        self._endings.remove(ending)
        ended = [
            r
            for r in ending.records.values()
            if self._claimed.get(self._file_key(r), ending.run) <= ending.run
        ]
        if not self._endings:
            self._claimed = {}
        return ended

    def _snapshots(self) -> None:
        while not self._wake.wait(self.snapshot_interval):
            if self._dirty:
                _ = self.write()

    def write(self) -> bool:
        """Rebuild and write the file(s) now. Never raises."""
        with self._write_lock:
            now = time.time_ns()
            with self._lock:
                self._dirty = False
                records = list(self._records.values())
                open_spans = list(self._open.values())
            # Open spans keep the structure of a snapshot (the root's name, the
            # agent a call belongs to); an open LLM call has no output yet.
            for span in open_spans:
                try:
                    record = span_record(span, end_time_ns=now)
                except RuntimeError:  # attributes changed while being copied
                    continue
                if record is not None and record.span_type != "LLM":
                    records.append(record)
            open_trace_ids = {
                str(uuid.UUID(int=span.context.trace_id))
                for span in open_spans
                if span.context is not None
            }
            return self._write(records, open_trace_ids, retry=True)

    def _write(
        self, records: list[SpanRecord], open_trace_ids: Collection[str], retry: bool
    ) -> bool:
        """Write `records`. Never raises. Holds `_write_lock`."""
        options = BuildOptions(release=self.release, exported=self.exported)
        try:
            _ = write_trajectories(records, self.path, options, open_trace_ids)
        except Exception as e:
            if retry:
                # Retry at the next snapshot even if no other span ends: the
                # last snapshot is what a killed agent leaves behind.
                with self._lock:
                    self._dirty = True
            if not self._warned:
                self._warned = True
                LOG.warning(f"Could not write the trajectory to {self.path}: {e}")
            else:
                LOG.debug("Could not write the trajectory", exc_info=True)
            return False
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._stopped = True
            thread = self._thread
        self._wake.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        _ = self.write()

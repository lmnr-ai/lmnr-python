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
from collections.abc import Mapping
from typing import Any, cast

from opentelemetry.sdk.trace import ReadableSpan, Span

from lmnr.opentelemetry_lib.tracing.attributes import SPAN_INPUT, SPAN_OUTPUT, SPAN_TYPE
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.trajectory.builder import BuildOptions, SpanRecord, format_time
from lmnr.sdk.trajectory.writer import write_trajectories

LOG = get_default_logger(__name__)

DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 2.0

_SESSION_ID = "lmnr.association.properties.session_id"
_RELEASE = "lmnr.span.release"
# Legacy indexed prompt attributes: only the system prompt is read (the OpenAI
# Responses API records `instructions` nowhere else).
_LEGACY_SYSTEM = ("gen_ai.prompt.0.role", "gen_ai.prompt.0.content")


def span_type(attributes: Mapping[str, Any]) -> str:
    """The span's type as app-server infers it (`traces/spans.rs::span_type`),
    so the file and the Laminar UI agree on what is an LLM or tool call."""
    explicit = attributes.get(SPAN_TYPE)
    if isinstance(explicit, str) and explicit:
        return explicit
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
    return key in (_SESSION_ID, _RELEASE, SPAN_TYPE)


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
    kind = span_type(attributes)
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
        the file first: this run's spans are gone afterwards."""
        with self._lock:
            self._records = {}
            self._open = {}
            self._dirty = False

    def forget(self, span: ReadableSpan) -> None:
        """Stop tracking a span that won't be recorded."""
        if span.context is not None:
            with self._lock:
                _ = self._open.pop(span.context.span_id, None)

    def on_end(self, span: ReadableSpan) -> None:
        record = span_record(span)
        with self._lock:
            if span.context is not None:
                _ = self._open.pop(span.context.span_id, None)
            if record is None:
                return
            self._records[record.span_id] = record
            self._dirty = True
            if self._thread is None and not self._stopped and self.snapshot_interval > 0:
                self._thread = threading.Thread(
                    target=self._snapshots, name="LmnrTrajectorySnapshot", daemon=True
                )
                self._thread.start()

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
            options = BuildOptions(release=self.release, exported=self.exported)
            try:
                _ = write_trajectories(records, self.path, options, open_trace_ids)
            except Exception as e:
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

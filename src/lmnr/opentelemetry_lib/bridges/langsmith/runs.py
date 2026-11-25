"""Turns the runs LangSmith's client sends into Laminar spans.

LangSmith sends each run in two halves: a create when it starts and an
update when it ends. `LangSmithRunRecorder` keeps the create half (its
inputs already converted, so later mutations of the caller's objects don't
leak in) and emits one finished span on Laminar's processor when the update
arrives. A run sent complete in its create (deferred inputs, hand-built
`RunTree`s posted after `end()`) is emitted right away, and a later update
for it is ignored.

IDs are deterministic, so a run's children (and the other half of a
distributed trace) find it without coordination: the trace id is the
LangSmith trace UUID, the span id the FNV-1a hash of the run UUID. A
LangSmith root started under a Laminar span joins that span's trace instead.
"""

from __future__ import annotations

import datetime
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span as SDKSpan
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags
from opentelemetry.util.types import AttributeValue

from lmnr.opentelemetry_lib.bridges.base import Bridge
from lmnr.opentelemetry_lib.bridges.langsmith.translate import (
    LLM_RUN_TYPES,
    association_attributes,
    exception_attributes,
    input_attributes,
    is_glue_run,
    is_langchain_run,
    output_attributes,
    request_attributes,
    response_attributes,
    span_type,
    tool_attributes,
    usage_attributes,
)
from lmnr.opentelemetry_lib.tracing import span_limits
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    PARENT_SPAN_IDS_PATH,
    PARENT_SPAN_PATH,
    SPAN_BRIDGE_NAME,
    SPAN_BRIDGE_VERSION,
    SPAN_IDS_PATH,
    SPAN_PATH,
    SPAN_TYPE,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

# The LangSmith run a span came from, to find it in LangSmith.
LANGSMITH_RUN_ID = "langsmith.run.id"
LANGSMITH_RUN_TYPE = "langsmith.run.type"
# Set on a run that had not ended when Laminar shut down or the process
# exited; its span ends then (`end_open_runs`).
LANGSMITH_RUN_INCOMPLETE = "langsmith.run.incomplete"

# Open runs, and recently finished ones (for children that start after their
# parent ended, and updates that follow a complete create). Both bounded: a
# run whose update never comes is dropped when evicted. At shutdown and
# process exit, the runs still open are emitted as incomplete.
MAX_OPEN_RUNS = 10_000
MAX_FINISHED_RUNS = 10_000

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK_64 = 0xFFFFFFFFFFFFFFFF


def span_id_for(run_id: str) -> int:
    """64-bit FNV-1a of the run UUID's canonical text. Same as the TS SDK."""
    value = _FNV_OFFSET
    for byte in str(run_id).lower().encode():
        value = ((value ^ byte) * _FNV_PRIME) & _MASK_64
    return value or 1


def trace_id_for(trace_uuid: str) -> int:
    """A LangSmith trace UUID as a 128-bit OTel trace id."""
    try:
        return uuid.UUID(str(trace_uuid)).int or 1
    except ValueError:
        return span_id_for(trace_uuid)


def _ids_path_entry(span_id: int) -> str:
    # Laminar writes span ids into `ids_path` in UUID form.
    return str(uuid.UUID(int=span_id))


def _time_ns(value: Any) -> int | None:
    if isinstance(value, datetime.datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.datetime.fromisoformat(value)
        except ValueError:
            return None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        # Epoch milliseconds (the JS SDK's end times).
        return int(value * 1_000_000)
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    delta = dt - datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def _dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


class _BridgeSpan(SDKSpan):
    """A span built from a finished run. The SDK only lets subclasses of
    `Span` be constructed directly; tracers can't take explicit ids."""


@dataclass
class _Node:
    """Where a run sits in Laminar's tree; what its children need.

    A skipped (glue) run takes its parent's place, so its children nest under
    the nearest recorded ancestor."""

    trace_id: int
    span_id: int
    path: list[str]
    ids_path: list[str]


@dataclass
class _OpenRun:
    node: _Node
    parent_span_id: int | None
    parent_path: list[str]
    parent_ids_path: list[str]
    context: Context
    skipped: bool
    run: dict[str, Any]
    input_attributes: dict[str, AttributeValue] = field(default_factory=dict)
    extra_attributes: dict[str, AttributeValue] = field(default_factory=dict)


class LangSmithRunRecorder:
    """Merges each run's create and update halves into one Laminar span."""

    def __init__(
        self,
        span_processor: SpanProcessor,
        tracer_provider: TracerProvider,
        bridge_version: str | None,
    ) -> None:
        self._span_processor: SpanProcessor = span_processor
        self._resource: Resource = tracer_provider.resource
        self._bridge_version: str | None = bridge_version
        self._scope: InstrumentationScope = InstrumentationScope("langsmith", bridge_version)
        self._lock: threading.Lock = threading.Lock()
        self._open: OrderedDict[str, _OpenRun] = OrderedDict()
        self._finished: OrderedDict[str, _Node] = OrderedDict()

    def _node(self, run_id: str) -> _Node | None:
        with self._lock:
            open_run = self._open.get(run_id)
            if open_run is not None:
                return open_run.node
            return self._finished.get(run_id)

    def on_create(
        self, run: dict[str, Any], context: Context, topology: dict[str, str] | None = None
    ) -> None:
        """A run started (or arrived complete). `run` is a JSON snapshot of
        what LangSmith sends; `context` is Laminar's context where it
        started."""
        run_id = str(run.get("id") or "")
        if not run_id:
            return
        with self._lock:
            if run_id in self._open or run_id in self._finished:
                return
        name = str(run.get("name") or "run")
        run_type = run.get("run_type")
        extra = _dict(run.get("extra"))
        tags = run.get("tags")
        is_langchain = is_langchain_run(extra)
        parent_run_id = run.get("parent_run_id")

        parent = self._node(str(parent_run_id)) if parent_run_id else None
        parent_span_id: int | None
        if parent is not None:
            trace_id = parent.trace_id
            parent_span_id = parent.span_id
            parent_path, parent_ids_path = list(parent.path), list(parent.ids_path)
        elif parent_run_id:
            # Continued from another service (`langsmith-trace` headers), or
            # the parent started before the bridge was installed: the same
            # ids it got (or will get) there, without its path.
            trace_id = trace_id_for(str(run.get("trace_id") or run_id))
            parent_span_id = span_id_for(str(parent_run_id))
            parent_path, parent_ids_path = [], []
        else:
            laminar_span = trace.get_current_span(context)
            span_context = laminar_span.get_span_context()
            if span_context.is_valid:
                # A LangSmith trace started under a Laminar span joins its trace.
                trace_id = span_context.trace_id
                parent_span_id = span_context.span_id
                attrs = cast(Mapping[str, Any], getattr(laminar_span, "attributes", None) or {})
                parent_path = [str(p) for p in attrs.get(SPAN_PATH) or ()]
                parent_ids_path = [str(p) for p in attrs.get(SPAN_IDS_PATH) or ()]
            else:
                trace_id = trace_id_for(str(run.get("trace_id") or run_id))
                parent_span_id = None
                parent_path, parent_ids_path = [], []

        skipped = parent is not None and is_glue_run(name, tags, is_langchain)
        if skipped:
            assert parent is not None
            node = _Node(trace_id, parent.span_id, parent_path, parent_ids_path)
        else:
            span_id = span_id_for(run_id)
            node = _Node(
                trace_id,
                span_id,
                [*parent_path, name],
                [*parent_ids_path, _ids_path_entry(span_id)],
            )
        open_run = _OpenRun(
            node=node,
            parent_span_id=parent_span_id,
            parent_path=parent_path,
            parent_ids_path=parent_ids_path,
            context=context,
            skipped=skipped,
            run=run,
        )
        if not skipped:
            invocation_params = extra.get("invocation_params")
            open_run.input_attributes = input_attributes(
                run_type, run.get("inputs"), invocation_params, is_langchain
            )
            if run_type in LLM_RUN_TYPES:
                open_run.input_attributes.update(
                    request_attributes(run.get("inputs"), metadata_of(run), invocation_params)
                )
            # LangGraph's topology (published by Laminar's langgraph
            # instrumentation around each graph step) belongs on the
            # graph's own run, the first chain run of the step.
            if topology and is_langchain and run_type == "chain":
                if not parent_run_id:
                    for key, value in topology.items():
                        open_run.extra_attributes[f"{ASSOCIATION_PROPERTIES}.{key}"] = value
                topology.clear()
        # Only the snapshot's other fields are needed from here on.
        run.pop("inputs", None)

        evicted: _OpenRun | None = None
        with self._lock:
            self._open[run_id] = open_run
            if len(self._open) > MAX_OPEN_RUNS:
                _, evicted = self._open.popitem(last=False)
        if evicted is not None:
            logger.debug("LangSmith bridge: dropped a run that never ended")
        if run.get("end_time") is not None:
            self.on_update(run)

    def on_update(self, update: dict[str, Any]) -> None:
        """A run ended: merge the update into its create and emit the span."""
        run_id = str(update.get("id") or "")
        if not run_id:
            return
        with self._lock:
            open_run = self._open.pop(run_id, None)
            known = open_run is None and run_id in self._finished
        if open_run is None:
            if known:
                # Already emitted from a complete create.
                return
            # The create came before the bridge was installed (or was never
            # seen): the update carries the rest of the run in current
            # LangSmith versions.
            self.on_create(dict(update), Context())
            return
        with self._lock:
            self._finished[run_id] = open_run.node
            if len(self._finished) > MAX_FINISHED_RUNS:
                _ = self._finished.popitem(last=False)
        if open_run.skipped:
            return
        run = open_run.run
        for key, value in update.items():
            if value is not None:
                run[key] = value
        try:
            self._emit(run_id, open_run, run)
        except Exception:
            logger.debug("LangSmith bridge failed to record a run", exc_info=True)

    def end_open_runs(self) -> int:
        """Emit every run that hasn't ended, ending now and marked
        `langsmith.run.incomplete`. For shutdown and process exit, after
        which its update can't arrive: LangSmith keeps such a run as in
        progress, and without this Laminar would have no span for it. Latest
        first, so children end before their parents. Returns how many spans
        were emitted."""
        with self._lock:
            runs = list(reversed(self._open.items()))
            self._open.clear()
            for run_id, open_run in runs:
                self._finished[run_id] = open_run.node
                if len(self._finished) > MAX_FINISHED_RUNS:
                    _ = self._finished.popitem(last=False)
        emitted = 0
        for run_id, open_run in runs:
            if open_run.skipped:
                continue
            open_run.extra_attributes[LANGSMITH_RUN_INCOMPLETE] = True
            try:
                self._emit(run_id, open_run, open_run.run)
                emitted += 1
            except Exception:
                logger.debug("LangSmith bridge failed to record an open run", exc_info=True)
        return emitted

    def _emit(self, run_id: str, open_run: _OpenRun, run: dict[str, Any]) -> None:
        run_type = run.get("run_type")
        extra = _dict(run.get("extra"))
        metadata = _dict(extra.get("metadata"))
        invocation_params = extra.get("invocation_params")
        is_langchain = is_langchain_run(extra)
        outputs = run.get("outputs")
        node = open_run.node

        attributes: dict[str, AttributeValue] = {
            SPAN_TYPE: span_type(cast(str | None, run_type)),
            SPAN_BRIDGE_NAME: Bridge.LANGSMITH.value,
            LANGSMITH_RUN_ID: run_id,
        }
        if isinstance(run_type, str):
            attributes[LANGSMITH_RUN_TYPE] = run_type
        if self._bridge_version:
            attributes[SPAN_BRIDGE_VERSION] = self._bridge_version
        if open_run.parent_path:
            attributes[PARENT_SPAN_PATH] = open_run.parent_path
            attributes[PARENT_SPAN_IDS_PATH] = open_run.parent_ids_path
        input_attrs = open_run.input_attributes
        if not input_attrs and run.get("inputs"):
            # LangChain sends deferred inputs (streamed into a runnable) with
            # the update.
            input_attrs = input_attributes(
                run_type, run.get("inputs"), invocation_params, is_langchain
            )
            if run_type in LLM_RUN_TYPES:
                input_attrs.update(
                    request_attributes(run.get("inputs"), metadata, invocation_params)
                )
        attributes.update(input_attrs)
        attributes.update(output_attributes(run_type, outputs, is_langchain))
        if run_type in LLM_RUN_TYPES:
            attributes.update(response_attributes(outputs))
            attributes.update(usage_attributes(outputs, metadata))
        attributes.update(
            tool_attributes(cast(str | None, run_type), str(run.get("name") or "run"), extra)
        )
        attributes.update(
            association_attributes(
                metadata,
                run.get("tags"),
                is_root=not run.get("parent_run_id"),
                invocation_params=invocation_params,
            )
        )
        attributes.update(open_run.extra_attributes)

        start_ns = _time_ns(run.get("start_time")) or time.time_ns()
        end_ns = _time_ns(run.get("end_time")) or time.time_ns()
        parent = (
            SpanContext(
                trace_id=node.trace_id,
                span_id=open_run.parent_span_id,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            if open_run.parent_span_id is not None
            else None
        )
        span = _BridgeSpan(
            name=str(run.get("name") or "run"),
            context=SpanContext(
                trace_id=node.trace_id,
                span_id=node.span_id,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            ),
            parent=parent,
            resource=self._resource,
            attributes=attributes,
            span_processor=self._span_processor,
            kind=SpanKind.INTERNAL,
            instrumentation_scope=self._scope,
            limits=span_limits(),
        )
        # Laminar's `on_start` reads session / user / metadata set through
        # Laminar from the context the run started in.
        span.start(start_time=start_ns, parent_context=open_run.context)
        error = run.get("error")
        if isinstance(error, str) and error:
            # Laminar marks a span failed only by an `exception` event.
            span.add_event("exception", exception_attributes(error), timestamp=end_ns)
            span.set_status(Status(StatusCode.ERROR, error.split("\n", 1)[0]))
        span.end(end_time=max(end_ns, start_ns))

    def clear(self) -> None:
        with self._lock:
            dropped = len(self._open)
            self._open.clear()
            self._finished.clear()
        if dropped:
            logger.debug("LangSmith bridge: dropped %d runs that had not ended", dropped)


def metadata_of(run: Mapping[str, Any]) -> dict[str, Any]:
    """A run's `extra.metadata`."""
    return _dict(_dict(run.get("extra")).get("metadata"))


__all__ = ["LangSmithRunRecorder", "span_id_for", "trace_id_for"]

"""Where the bridge listens to the LangSmith SDK.

Every way of tracing with LangSmith (`@traceable`, `wrap_openai` /
`wrap_anthropic`, LangChain's and LangGraph's tracer, the
`langsmith.integrations.*` hooks, hand-built `RunTree`s) ends in
`RunTree.post()` / `patch()`, which call the client's `create_run` /
`update_run`. Those sample the run, apply the client's redaction
(`hide_inputs` / `hide_outputs` / `hide_metadata`, the anonymizer) and hand
the result to `Client._create_run` / `_update_run` (directly, or after
`process_buffered_run_ops`), which queue it for sending. The bridge taps
there: it sees exactly what LangSmith sends, after the customer's sampling
and redaction, and nothing else.

The tap copies the run dict before calling through (LangSmith's
serialization pops fields from it) and converts it after the call returns,
so LangSmith's queueing is never delayed or changed by the bridge.

With write replicas (`LANGSMITH_RUNS_ENDPOINTS`, `tracing_context(replicas=
...)`), `RunTree.post()` / `patch()` call the client once per replica, with
remapped ids for some. A wrapper around them collects those calls and keeps
one: the one with the run's own ids, else the first. Replicas are inherited
by child runs, so every run of a tree keeps the same one.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, cast

import orjson
from opentelemetry.context import Context

from lmnr.opentelemetry_lib.bridges.langsmith.runs import LangSmithRunRecorder
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.context_keys import (
    LANGGRAPH_TOPOLOGY,
)
from lmnr.opentelemetry_lib.tracing.context import get_current_context
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import json_dumps

logger = get_default_logger(__name__)


@dataclass
class _Write:
    run: dict[str, Any]
    context: Context
    topology: dict[str, str] | None


@dataclass
class _Scope:
    """One `RunTree.post()` / `patch()` call and the client writes it made."""

    run_id: str
    writes: list[_Write] = field(default_factory=list)


_scope: ContextVar[_Scope | None] = ContextVar("lmnr_langsmith_scope", default=None)


def _snapshot_dumps() -> Callable[[Any], bytes]:
    """LangSmith's own JSON serializer (so Laminar gets the same values
    LangSmith stored), or Laminar's when it moved."""
    try:
        from langsmith._internal._serde import dumps_json

        return cast(Callable[[Any], bytes], dumps_json)
    except Exception:
        return lambda value: json_dumps(value).encode()


class LangSmithTap:
    def __init__(self, recorder: LangSmithRunRecorder) -> None:
        self._recorder: LangSmithRunRecorder = recorder
        self._dumps: Callable[[Any], bytes] = _snapshot_dumps()
        # (class, attribute, our wrapper, what it replaced)
        self._patched: list[tuple[type, str, Callable[..., Any], Callable[..., Any]]] = []
        self.active: bool = False

    # --- capture --------------------------------------------------------------

    def _snapshot(self, run: dict[str, Any]) -> dict[str, Any] | None:
        try:
            data = orjson.loads(self._dumps(run))
        except Exception:
            logger.debug("LangSmith bridge could not serialize a run", exc_info=True)
            return None
        return cast(dict[str, Any], data) if isinstance(data, dict) else None

    def _write(self, kind: str, run: dict[str, Any], context: Context) -> None:
        topology = LANGGRAPH_TOPOLOGY.get() if kind == "create" else None
        scope = _scope.get()
        write = _Write(run, context, topology)
        if scope is not None:
            scope.writes.append(write)
        else:
            self._dispatch(kind, write)

    def _dispatch(self, kind: str, write: _Write) -> None:
        snapshot = self._snapshot(write.run)
        if snapshot is None:
            return
        if kind == "create":
            self._recorder.on_create(snapshot, write.context, write.topology)
        else:
            self._recorder.on_update(snapshot)

    def _end_scope(self, kind: str, scope: _Scope) -> None:
        if not scope.writes:
            return
        own = [w for w in scope.writes if str(w.run.get("id")) == scope.run_id]
        self._dispatch(kind, own[0] if own else scope.writes[0])

    # --- wrappers -------------------------------------------------------------

    def _wrap_client_write(self, kind: str, original: Callable[..., Any]) -> Callable[..., Any]:
        tap = self

        @functools.wraps(original)
        def wrapper(client: Any, run: Any, *args: Any, **kwargs: Any) -> Any:
            copy: dict[str, Any] | None = None
            context: Context | None = None
            if tap.active and isinstance(run, dict):
                try:
                    copy = dict(cast(dict[str, Any], run))
                    # Files (images, audio) aren't recorded; skip serializing them.
                    _ = copy.pop("attachments", None)
                    context = get_current_context()
                except Exception:
                    copy = None
            try:
                return original(client, run, *args, **kwargs)
            finally:
                if copy is not None and context is not None:
                    try:
                        tap._write(kind, copy, context)
                    except Exception:
                        logger.debug("LangSmith bridge failed to read a run", exc_info=True)

        return wrapper

    def _wrap_run_tree_send(self, kind: str, original: Callable[..., Any]) -> Callable[..., Any]:
        tap = self

        @functools.wraps(original)
        def wrapper(run_tree: Any, *args: Any, **kwargs: Any) -> Any:
            if not tap.active:
                return original(run_tree, *args, **kwargs)
            scope = _Scope(run_id=str(getattr(run_tree, "id", "")))
            token = _scope.set(scope)
            try:
                return original(run_tree, *args, **kwargs)
            finally:
                _scope.reset(token)
                try:
                    tap._end_scope(kind, scope)
                except Exception:
                    logger.debug("LangSmith bridge failed to read a run", exc_info=True)

        return wrapper

    # --- install --------------------------------------------------------------

    def install(self) -> bool:
        from langsmith.client import Client
        from langsmith.run_trees import RunTree

        targets: list[tuple[type, str, Callable[..., Any], Callable[..., Any]]] = []
        for cls, name, kind, wrap in (
            (Client, "_create_run", "create", self._wrap_client_write),
            (Client, "_update_run", "update", self._wrap_client_write),
            (RunTree, "post", "create", self._wrap_run_tree_send),
            (RunTree, "patch", "update", self._wrap_run_tree_send),
        ):
            original = cls.__dict__.get(name)
            if not callable(original):
                logger.warning(
                    "The LangSmith bridge could not find langsmith's %s.%s; "
                    + "LangSmith runs will not be sent to Laminar.",
                    cls.__name__,
                    name,
                )
                return False
            original = cast(Callable[..., Any], original)
            targets.append((cls, name, wrap(kind, original), original))
        for cls, name, wrapper, original in targets:
            setattr(cls, name, wrapper)
            self._patched.append((cls, name, wrapper, original))
        self.active = True
        return True

    def uninstall(self) -> None:
        # A wrapper someone else put on top of ours stays, and ours inside it
        # just calls through.
        self.active = False
        for cls, name, wrapper, original in self._patched:
            if cls.__dict__.get(name) is wrapper:
                setattr(cls, name, original)
            else:
                logger.debug("LangSmith bridge: %s.%s was re-wrapped", cls.__name__, name)
        self._patched.clear()
        self._recorder.clear()


__all__ = ["LangSmithTap"]

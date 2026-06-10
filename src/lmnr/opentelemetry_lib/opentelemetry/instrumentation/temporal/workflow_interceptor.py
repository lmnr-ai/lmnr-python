"""Workflow-side Laminar interceptor — runs inside the Temporal sandbox.

This module is loaded into the deterministic workflow sandbox (the Python
equivalent of the TS V8 isolate). The hard rules there: no real ``uuid``, no
wall-clock time, no I/O. So unlike the client/activity interceptors this file
NEVER imports ``lmnr`` and NEVER creates a Laminar span — it only forwards the
already-serialized trace headers that the client injected, so activities and
child workflows scheduled *from inside* a workflow stay parented to the
workflow-start trace (or to a signal/update handler's trace when one is
running).

Mirrors the TS ``workflow-interceptors.ts``. The only sandbox-safe primitives
used are :mod:`temporalio.worker` base classes (``temporalio`` is a sandbox
passthrough module), the pure-constant :mod:`.consts`, and ``contextvars`` —
a ``ContextVar`` is isolated per asyncio task, so a signal/update handler's
headers never leak into the main workflow path or into other concurrent
handlers, and reading/setting it is deterministic across replay.
"""

from __future__ import annotations

import contextvars
from typing import Any, NoReturn

import temporalio.api.common.v1
import temporalio.worker

from .consts import LAMINAR_SPAN_CONTEXT_HEADER, TRACEPARENT_HEADER

_Headers = dict[str, temporalio.api.common.v1.Payload]

# Trace headers scoped to the currently-running signal/update handler coroutine.
# Each handler runs as its own asyncio task (which copies the context), so this
# stays isolated from the main workflow path and from other concurrent handlers
# — overwriting shared instance state would corrupt those interleaved paths.
_handler_headers: contextvars.ContextVar[_Headers | None] = contextvars.ContextVar(
    "lmnr_temporal_handler_headers", default=None
)


def _has_trace_headers(headers: _Headers) -> bool:
    """True when the headers map actually carries an injected trace context.

    A signal or update may arrive with no injected context; in that case we must
    NOT enter the handler-headers scope, or ``_active_headers`` would resolve to
    that empty map and drop the workflow-start trace from any activity /
    child-workflow scheduled inside the handler.
    """
    return (
        headers.get(LAMINAR_SPAN_CONTEXT_HEADER) is not None
        or headers.get(TRACEPARENT_HEADER) is not None
    )


class LaminarWorkflowInboundInterceptor(temporalio.worker.WorkflowInboundInterceptor):
    """Forwards trace headers from the workflow-start (and signal/update) call to
    outbound activity / child-workflow scheduling. Creates no spans."""

    def __init__(
        self, next: temporalio.worker.WorkflowInboundInterceptor
    ) -> None:
        super().__init__(next)
        # Per-workflow-instance: each sandbox builds its own interceptor, so this
        # never crosses between concurrent workflows on the same worker.
        self._start_headers: _Headers = {}

    def init(self, outbound: temporalio.worker.WorkflowOutboundInterceptor) -> None:
        super().init(_LaminarWorkflowOutboundInterceptor(outbound, self))

    async def execute_workflow(
        self, input: temporalio.worker.ExecuteWorkflowInput
    ) -> Any:
        self._start_headers = dict(input.headers or {})
        return await super().execute_workflow(input)

    async def handle_signal(
        self, input: temporalio.worker.HandleSignalInput
    ) -> None:
        headers = dict(input.headers or {})
        if not _has_trace_headers(headers):
            return await super().handle_signal(input)
        token = _handler_headers.set(headers)
        try:
            return await super().handle_signal(input)
        finally:
            _handler_headers.reset(token)

    async def handle_update_handler(
        self, input: temporalio.worker.HandleUpdateInput
    ) -> Any:
        headers = dict(input.headers or {})
        if not _has_trace_headers(headers):
            return await super().handle_update_handler(input)
        token = _handler_headers.set(headers)
        try:
            return await super().handle_update_handler(input)
        finally:
            _handler_headers.reset(token)

    def _active_headers(self) -> _Headers:
        """The headers outbound calls propagate from: the active signal/update
        handler's headers when one is running, otherwise the workflow-start
        headers."""
        scoped = _handler_headers.get()
        return scoped if scoped is not None else self._start_headers


class _LaminarWorkflowOutboundInterceptor(
    temporalio.worker.WorkflowOutboundInterceptor
):
    def __init__(
        self,
        next: temporalio.worker.WorkflowOutboundInterceptor,
        root: LaminarWorkflowInboundInterceptor,
    ) -> None:
        super().__init__(next)
        self.root = root

    def _merge(self, input: Any) -> None:
        # Caller-supplied headers win over the propagated trace headers.
        input.headers = {**self.root._active_headers(), **dict(input.headers or {})}

    def start_activity(
        self, input: temporalio.worker.StartActivityInput
    ) -> temporalio.workflow.ActivityHandle[Any]:
        self._merge(input)
        return super().start_activity(input)

    def start_local_activity(
        self, input: temporalio.worker.StartLocalActivityInput
    ) -> temporalio.workflow.ActivityHandle[Any]:
        self._merge(input)
        return super().start_local_activity(input)

    async def start_child_workflow(
        self, input: temporalio.worker.StartChildWorkflowInput
    ) -> temporalio.workflow.ChildWorkflowHandle[Any, Any]:
        self._merge(input)
        return await super().start_child_workflow(input)

    def continue_as_new(
        self, input: temporalio.worker.ContinueAsNewInput
    ) -> NoReturn:
        self._merge(input)
        super().continue_as_new(input)

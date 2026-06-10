"""Laminar Temporal interceptors (client + activity sides).

A single :class:`LaminarTracingInterceptor` implements both
``temporalio.client.Interceptor`` and ``temporalio.worker.Interceptor``. Injected
at the client level, it is automatically inherited by any worker built from that
client (see ``temporalio.worker._worker`` — client interceptors are prepended to
worker interceptors), so one injection covers the client, activity and workflow
paths. This collapses the three-way split the TypeScript SDK needs (separate
client patch, worker patch, and bundled workflow module) into one object.

- Client side: a workflow-lifecycle Laminar span wraps ``start_workflow`` and its
  serialized context is injected into Temporal headers; ``signal`` / ``query`` /
  ``update`` calls forward the caller's active span context instead.
- Activity side: the propagated context is read back out of the headers and the
  activity runs under a Laminar span parented to it. Passing the context as
  ``parent_span_context`` is what restores the debugger context too —
  ``Laminar.start_span`` parses the nested ``debug`` block and arms the
  downstream debug runtime for free.
- Workflow side: see :mod:`.workflow_interceptor` (runs in the sandbox).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import temporalio.activity
import temporalio.client
import temporalio.worker

from lmnr.opentelemetry_lib.tracing.span import LaminarSpan
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger

from .helpers import build_headers, restore_context_from_headers
from .workflow_interceptor import LaminarWorkflowInboundInterceptor

logger = get_default_logger(__name__)


@dataclass
class LaminarTemporalInterceptorOptions:
    """Options controlling the Laminar Temporal interceptor.

    Mirrors the TypeScript ``LaminarTemporalInterceptorOptions``.
    """

    #: Wrap each activity execution in a Laminar span named after the activity
    #: type. When ``False``, only context restoration happens (so your own
    #: ``observe`` calls act as roots inside the activity).
    create_activity_span: bool = True
    #: Record the activity's arguments as the span input. Ignored when
    #: ``create_activity_span`` is ``False``.
    record_activity_args: bool = True
    #: Record the activity's return value as the span output. Ignored when
    #: ``create_activity_span`` is ``False``.
    record_activity_output: bool = True


class LaminarTracingInterceptor(
    temporalio.client.Interceptor, temporalio.worker.Interceptor
):
    """Unified Laminar interceptor for Temporal clients and workers."""

    def __init__(
        self, options: LaminarTemporalInterceptorOptions | None = None
    ) -> None:
        self.options = options or LaminarTemporalInterceptorOptions()

    def intercept_client(
        self, next: temporalio.client.OutboundInterceptor
    ) -> temporalio.client.OutboundInterceptor:
        return _LaminarClientOutboundInterceptor(next, self)

    def intercept_activity(
        self, next: temporalio.worker.ActivityInboundInterceptor
    ) -> temporalio.worker.ActivityInboundInterceptor:
        return _LaminarActivityInboundInterceptor(next, self)

    def workflow_interceptor_class(
        self, input: temporalio.worker.WorkflowInterceptorClassInput
    ) -> type[temporalio.worker.WorkflowInboundInterceptor]:
        return LaminarWorkflowInboundInterceptor


class _LaminarClientOutboundInterceptor(temporalio.client.OutboundInterceptor):
    def __init__(
        self,
        next: temporalio.client.OutboundInterceptor,
        root: LaminarTracingInterceptor,
    ) -> None:
        super().__init__(next)
        self.root = root

    async def start_workflow(
        self, input: temporalio.client.StartWorkflowInput
    ) -> temporalio.client.WorkflowHandle[Any, Any]:
        # A dedicated workflow-lifecycle span: it nests under any active Laminar
        # span (start_span uses the current context as parent), and its context
        # is what worker-side activities parent to.
        span = Laminar.start_span(
            name=f"temporal.workflow.{input.workflow}",
            input=getattr(input, "args", None),
        )
        span_context = Laminar.get_laminar_span_context(span)
        input.headers = build_headers(dict(input.headers or {}), span_context)
        try:
            handle = await super().start_workflow(input)
        except Exception as e:
            span.record_exception(e)
            span.end()
            raise
        _wrap_workflow_handle(handle, span)
        return handle

    async def signal_workflow(
        self, input: temporalio.client.SignalWorkflowInput
    ) -> None:
        input.headers = build_headers(
            dict(input.headers or {}), Laminar.get_laminar_span_context()
        )
        return await super().signal_workflow(input)

    async def query_workflow(
        self, input: temporalio.client.QueryWorkflowInput
    ) -> Any:
        input.headers = build_headers(
            dict(input.headers or {}), Laminar.get_laminar_span_context()
        )
        return await super().query_workflow(input)

    async def start_workflow_update(
        self, input: temporalio.client.StartWorkflowUpdateInput
    ) -> temporalio.client.WorkflowUpdateHandle[Any]:
        input.headers = build_headers(
            dict(input.headers or {}), Laminar.get_laminar_span_context()
        )
        return await super().start_workflow_update(input)


def _wrap_workflow_handle(
    handle: temporalio.client.WorkflowHandle[Any, Any], span: Any
) -> None:
    """Wrap a workflow handle so the lifecycle span ends on the FIRST terminal
    call — ``result()`` resolving/raising, ``cancel()`` or ``terminate()``.

    ``WorkflowHandle`` instances have a ``__dict__``, so assigning to
    ``handle.result`` shadows the bound class method on this instance only.
    """
    state = {"closed": False}

    def close() -> None:
        if state["closed"]:
            return
        state["closed"] = True
        span.end()

    orig_result = handle.result
    orig_cancel = handle.cancel
    orig_terminate = handle.terminate

    async def result(*args: Any, **kwargs: Any) -> Any:
        if state["closed"]:
            return await orig_result(*args, **kwargs)
        try:
            res = await orig_result(*args, **kwargs)
        except Exception as e:
            span.record_exception(e)
            close()
            raise
        if isinstance(span, LaminarSpan):
            try:
                span.set_output(res)
            except Exception as e:
                logger.debug(f"failed to set workflow span output: {e}")
        close()
        return res

    def _wrap_terminating(name: str, orig: Any) -> Any:
        async def terminating(*args: Any, **kwargs: Any) -> Any:
            if state["closed"]:
                return await orig(*args, **kwargs)
            child = Laminar.start_span(
                name=name,
                parent_span_context=Laminar.get_laminar_span_context(span),
            )
            try:
                res = await orig(*args, **kwargs)
            except Exception as e:
                child.record_exception(e)
                child.end()
                span.record_exception(e)
                close()
                raise
            child.end()
            close()
            return res

        return terminating

    handle.result = result
    handle.cancel = _wrap_terminating("temporal.workflow.cancel", orig_cancel)
    handle.terminate = _wrap_terminating("temporal.workflow.terminate", orig_terminate)


class _LaminarActivityInboundInterceptor(
    temporalio.worker.ActivityInboundInterceptor
):
    def __init__(
        self,
        next: temporalio.worker.ActivityInboundInterceptor,
        root: LaminarTracingInterceptor,
    ) -> None:
        super().__init__(next)
        self.root = root

    async def execute_activity(
        self, input: temporalio.worker.ExecuteActivityInput
    ) -> Any:
        restored = restore_context_from_headers(dict(input.headers or {}))
        if not self.root.options.create_activity_span or restored is None:
            return await super().execute_activity(input)

        info = temporalio.activity.info()
        name = info.activity_type or "temporal.activity"
        # Explicit parent_span_context wins over any ambient worker context, so
        # the activity span always parents to the propagated remote context.
        span = Laminar.start_span(
            name=name,
            parent_span_context=restored,
            input=input.args if self.root.options.record_activity_args else None,
        )
        with Laminar.use_span(span, end_on_exit=True):
            res = await super().execute_activity(input)
            if self.root.options.record_activity_output and isinstance(
                span, LaminarSpan
            ):
                try:
                    span.set_output(res)
                except Exception as e:
                    logger.debug(f"failed to set activity span output: {e}")
            return res

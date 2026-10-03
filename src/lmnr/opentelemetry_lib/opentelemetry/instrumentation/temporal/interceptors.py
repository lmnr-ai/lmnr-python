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
  downstream debug runtime for free. When ``create_activity_span`` is disabled,
  the context is still activated (via ``Laminar.use_span_context``) so spans
  created inside the activity nest under the workflow trace without a wrapper.
- Workflow side: see :mod:`.workflow_interceptor` (runs in the sandbox).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import CoroutineType
from typing import Any

import temporalio.activity
import temporalio.client
import temporalio.worker
from opentelemetry.trace import Span
from typing_extensions import TypeVar, override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.temporal.helpers import (
    build_headers,
    restore_context_from_headers,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.temporal.workflow_interceptor import (
    LaminarWorkflowInboundInterceptor,
)
from lmnr.opentelemetry_lib.tracing.span import LaminarSpan
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

T = TypeVar("T")

@dataclass
class LaminarTemporalInterceptorOptions:
    """Options controlling the Laminar Temporal interceptor.

    Mirrors the TypeScript ``LaminarTemporalInterceptorOptions``.
    """

    #: Wrap each activity execution in a Laminar span named after the activity
    #: type. When ``False``, no wrapper span is created but the propagated trace
    #: context is still activated, so your own ``observe`` calls (and any
    #: auto-instrumented spans) inside the activity nest under the workflow /
    #: client trace rather than starting a detached trace.
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
        super().__init__()
        self.options: LaminarTemporalInterceptorOptions = options or LaminarTemporalInterceptorOptions()

    @override
    def intercept_client(
        self, next: temporalio.client.OutboundInterceptor
    ) -> temporalio.client.OutboundInterceptor:
        return _LaminarClientOutboundInterceptor(next, self)

    @override
    def intercept_activity(
        self, next: temporalio.worker.ActivityInboundInterceptor
    ) -> temporalio.worker.ActivityInboundInterceptor:
        return _LaminarActivityInboundInterceptor(next, self)

    @override
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
        self.root: LaminarTracingInterceptor = root

    @override
    async def start_workflow(
        self, input: temporalio.client.StartWorkflowInput
    ) -> temporalio.client.WorkflowHandle[Any, Any]:  # pyright: ignore[reportExplicitAny]
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

    @override
    async def signal_workflow(
        self, input: temporalio.client.SignalWorkflowInput
    ) -> None:
        input.headers = build_headers(
            dict(input.headers or {}), Laminar.get_laminar_span_context()
        )
        return await super().signal_workflow(input)

    @override
    async def query_workflow(  # pyright: ignore[reportAny]
        self, input: temporalio.client.QueryWorkflowInput
    ) -> Any:  # pyright: ignore[reportExplicitAny]
        input.headers = build_headers(
            dict(input.headers or {}), Laminar.get_laminar_span_context()
        )
        return await super().query_workflow(input)  # pyright: ignore[reportAny]

    @override
    async def start_workflow_update(
        self, input: temporalio.client.StartWorkflowUpdateInput
    ) -> temporalio.client.WorkflowUpdateHandle[Any]:  # pyright: ignore[reportExplicitAny]
        input.headers = build_headers(
            dict(input.headers or {}), Laminar.get_laminar_span_context()
        )
        return await super().start_workflow_update(input)


def _wrap_workflow_handle(
    handle: temporalio.client.WorkflowHandle[Any, Any], span: Span,  # pyright: ignore[reportExplicitAny]
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

    async def result(*args: Any, **kwargs: Any) -> Any:  # pyright: ignore[reportAny, reportExplicitAny]
        if state["closed"]:
            return await orig_result(*args, **kwargs)  # pyright: ignore[reportAny]
        try:
            res = await orig_result(*args, **kwargs)  # pyright: ignore[reportAny]
        except Exception as e:
            span.record_exception(e)
            close()
            raise
        if isinstance(span, LaminarSpan):
            try:
                span.set_output(res)
            except Exception:
                logger.debug("failed to set workflow span output", exc_info=True)
        close()
        return res  # pyright: ignore[reportAny]

    def _wrap_terminating(
        name: str,
        orig: Callable[..., CoroutineType[Any, Any, T]],  # pyright: ignore[reportExplicitAny]
    ) -> Callable[..., CoroutineType[Any, Any, T]]:  # pyright: ignore[reportExplicitAny]
        async def terminating(*args: Any, **kwargs: Any) -> T:  # pyright: ignore[reportAny, reportExplicitAny]
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
        self.root: LaminarTracingInterceptor = root

    @override
    async def execute_activity(  # pyright: ignore[reportAny]
        self, input: temporalio.worker.ExecuteActivityInput
    ) -> Any:  # pyright: ignore[reportExplicitAny]
        restored = restore_context_from_headers(dict(input.headers or {}))
        if restored is None:
            return await super().execute_activity(input)  # pyright: ignore[reportAny]

        # When the wrapper span is disabled, still activate the propagated
        # context as the parent so spans created inside the activity (manual
        # `observe`, auto-instrumented LLM calls, etc.) nest under the workflow /
        # client trace — and a propagated debug block still arms the runtime.
        if not self.root.options.create_activity_span:
            with Laminar.use_span_context(restored):
                return await super().execute_activity(input)  # pyright: ignore[reportAny]

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
            res = await super().execute_activity(input)  # pyright: ignore[reportAny]
            if self.root.options.record_activity_output and isinstance(
                span, LaminarSpan
            ):
                try:
                    span.set_output(res)
                except Exception:
                    logger.debug("failed to set activity span output", exc_info=True)
            return res  # pyright: ignore[reportAny]

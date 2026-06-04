"""
Laminar Temporal interceptors for distributed trace context propagation.

Usage — Option A (explicit, always works):

    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.temporal import (
        LaminarTemporalInterceptor,
    )
    from temporalio.worker import Worker
    from temporalio.client import Client

    worker = await Worker.create(
        interceptors=[LaminarTemporalInterceptor()],
        ...
    )

    client = await Client.connect(
        "localhost:7233",
        interceptors=[LaminarTemporalInterceptor()],
    )

Usage — Option B (auto-patch via Laminar.initialize()):

    import temporalio.worker as temporal_worker
    import temporalio.client as temporal_client
    from lmnr import Laminar

    Laminar.initialize(
        instruments={},  # or your normal instrument set
        temporal_modules={
            "worker": temporal_worker,
            "client": temporal_client,
        },
    )

    # Worker.create() and Client.connect() now include Laminar interceptors.
    worker = await temporal_worker.Worker.create(...)
    client = await temporal_client.Client.connect(...)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, TraceFlags

from lmnr.opentelemetry_lib.tracing.context import (
    attach_context,
    detach_context,
    get_current_context,
)
from lmnr.opentelemetry_lib.tracing import TracerWrapper as TracerManager
from lmnr.sdk.types import LaminarSpanContext

logger = logging.getLogger(__name__)

# Header keys
LAMINAR_SPAN_CONTEXT_HEADER = "x-lmnr-span-context"
TRACEPARENT_HEADER = "traceparent"


# ─── Payload codec helpers ─────────────────────────────────────────────────────
# Temporal headers are Mapping[str, temporalio.api.common.v1.Payload].
# We avoid importing temporalio.api directly to keep it as a soft dep.

def _encode_payload(value: str) -> Any:
    """Encode a string as a Temporal Payload (JSON-encoded)."""
    data = json.dumps(value).encode("utf-8")
    return _make_payload(b"json/plain", data)


def _make_payload(encoding: bytes, data: bytes) -> Any:
    """Build a minimal Temporal Payload-compatible dict or proto object."""
    try:
        from google.protobuf import struct_pb2  # noqa: F401
        from temporalio.api.common.v1 import Payload

        p = Payload()
        p.metadata["encoding"] = encoding
        p.data = data
        return p
    except ImportError:
        # Fall back to a plain dict that mirrors the Payload protobuf shape.
        return {"metadata": {"encoding": encoding}, "data": data}


def _decode_payload(payload: Any) -> str | None:
    """Decode a Temporal Payload back to its string value."""
    if payload is None:
        return None
    try:
        data: bytes | None = None
        if hasattr(payload, "data"):
            data = payload.data
        elif isinstance(payload, dict):
            data = payload.get("data")
        if not data:
            return None
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


# ─── Context propagation helpers ───────────────────────────────────────────────

def _build_headers(existing: dict[str, Any]) -> dict[str, Any]:
    """
    Read the currently active Laminar span and encode its context into a
    Temporal headers dict.  Writes both `laminar-span-context` (full Laminar
    JSON) and `traceparent` (W3C, for interop).
    """
    from lmnr.sdk.laminar import Laminar

    headers = dict(existing)
    laminar_ctx = Laminar.get_laminar_span_context()
    if laminar_ctx is None:
        # Try OTel active span for traceparent-only propagation
        span = trace.get_current_span()
        if span.get_span_context().is_valid:
            sc = span.get_span_context()
            trace_hex = format(sc.trace_id, "032x")
            span_hex = format(sc.span_id, "016x")
            headers[TRACEPARENT_HEADER] = _encode_payload(
                f"00-{trace_hex}-{span_hex}-01"
            )
        return headers

    headers[LAMINAR_SPAN_CONTEXT_HEADER] = _encode_payload(str(laminar_ctx))

    trace_hex = str(laminar_ctx.trace_id).replace("-", "")
    span_hex = str(laminar_ctx.span_id).replace("-", "")[16:]
    headers[TRACEPARENT_HEADER] = _encode_payload(
        f"00-{trace_hex}-{span_hex}-01"
    )
    return headers


def _restore_context_from_headers(
    headers: dict[str, Any] | None,
) -> tuple[LaminarSpanContext, Any] | None:
    """
    Read `laminar-span-context` (preferred) or `traceparent` (fallback) from
    Temporal headers and restore the parent context onto Laminar's context stack.

    Returns a (LaminarSpanContext, token) pair so the caller can detach the
    context when the activity finishes, or None if no usable trace context was
    found.
    """
    if not headers:
        return None

    # Preferred: full Laminar context header
    laminar_raw = _decode_payload(headers.get(LAMINAR_SPAN_CONTEXT_HEADER))
    if laminar_raw:
        try:
            from lmnr.sdk.laminar import Laminar

            ctx = Laminar.deserialize_span_context(laminar_raw)
            token = _push_laminar_context(ctx)
            if token is not None:
                return ctx, token
        except Exception as e:
            logger.warning(
                "[Laminar] Could not restore %s: %s",
                LAMINAR_SPAN_CONTEXT_HEADER,
                e,
            )

    # Fallback: W3C traceparent
    traceparent = _decode_payload(headers.get(TRACEPARENT_HEADER))
    if traceparent:
        parts = traceparent.split("-")
        if len(parts) >= 3:
            _, trace_hex, span_hex = parts[0], parts[1], parts[2]
            try:
                import uuid

                trace_id = uuid.UUID(trace_hex)
                span_id = uuid.UUID(span_hex.zfill(32))
                ctx = LaminarSpanContext(
                    trace_id=trace_id,
                    span_id=span_id,
                    is_remote=True,
                    span_path=[],
                    span_ids_path=[],
                )
                token = _push_laminar_context(ctx)
                if token is not None:
                    return ctx, token
            except Exception as e:
                logger.warning(
                    "[Laminar] Could not restore traceparent: %s", e
                )

    return None


def _push_laminar_context(laminar_ctx: LaminarSpanContext) -> Any | None:
    """Push a LaminarSpanContext onto Laminar's context stack as a remote parent.

    Returns the detach token so the caller can clean up after the activity.
    """
    from lmnr.opentelemetry_lib.tracing.context import set_association_prop_context
    from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor

    otel_span_ctx = LaminarSpanContext.try_to_otel_span_context(laminar_ctx)
    if otel_span_ctx is None:
        return None

    # Register parent path info in the span processor so descendant spans
    # inherit the correct span path hierarchy.
    tracer_wrapper = getattr(TracerManager, "instance", None)
    if tracer_wrapper is not None:
        processor = getattr(tracer_wrapper, "_span_processor", None)
        if (
            processor is not None
            and isinstance(processor, LaminarSpanProcessor)
            and laminar_ctx.span_path
            and laminar_ctx.span_ids_path
        ):
            processor.set_parent_path_info(
                otel_span_ctx.span_id,  # int in Python SDK
                laminar_ctx.span_path,
                [str(sid) for sid in laminar_ctx.span_ids_path],
            )

    # Build an OTel context with the remote parent span and association props.
    base_ctx = get_current_context()
    parent_span = NonRecordingSpan(
        trace.SpanContext(
            trace_id=otel_span_ctx.trace_id,
            span_id=otel_span_ctx.span_id,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    )
    ctx_with_span = trace.set_span_in_context(parent_span, base_ctx)
    ctx_with_props = set_association_prop_context(
        user_id=laminar_ctx.user_id,
        session_id=laminar_ctx.session_id,
        trace_type=laminar_ctx.trace_type,
        metadata=laminar_ctx.metadata,
        context=ctx_with_span,
        attach=False,
    )
    return attach_context(ctx_with_props)


# ─── Interceptors ──────────────────────────────────────────────────────────────

class LaminarTemporalInterceptor:
    """
    A single Temporal interceptor class that handles both client-side header
    injection and worker-side context restoration for Laminar trace propagation.

    Register it on both your Client and Worker:

        client = await Client.connect(
            "localhost:7233",
            interceptors=[LaminarTemporalInterceptor()],
        )

        worker = await Worker.create(
            interceptors=[LaminarTemporalInterceptor(create_activity_span=True)],
            ...
        )
    """

    def __init__(self, *, create_activity_span: bool = True) -> None:
        self._create_activity_span = create_activity_span

    # ── Client-side: Workflow outbound calls ──

    def intercept_client(self, next_interceptor: Any) -> Any:
        """Return a client outbound interceptor that injects Laminar headers."""
        return _LaminarClientOutboundInterceptor(next_interceptor)

    # ── Worker-side: Activity inbound calls ──

    def intercept_activity(self, next_interceptor: Any) -> Any:
        """Return an activity inbound interceptor that restores Laminar context."""
        return _LaminarActivityInboundInterceptor(
            next_interceptor,
            create_activity_span=self._create_activity_span,
        )


class _LaminarClientOutboundInterceptor:
    """Injects Laminar span context into outgoing workflow start/signal/query calls."""

    def __init__(self, next_interceptor: Any) -> None:
        self._next = next_interceptor

    def __getattr__(self, name: str) -> Any:
        return getattr(self._next, name)

    async def start_workflow(self, input: Any) -> Any:  # type: ignore[override]
        try:
            input.headers = _build_headers(dict(input.headers or {}))
        except Exception as e:
            logger.debug("[Laminar] Could not inject headers into start_workflow: %s", e)
        return await self._next.start_workflow(input)

    async def signal_with_start_workflow(self, input: Any) -> Any:
        try:
            input.headers = _build_headers(dict(input.headers or {}))
        except Exception as e:
            logger.debug(
                "[Laminar] Could not inject headers into signal_with_start_workflow: %s", e
            )
        return await self._next.signal_with_start_workflow(input)

    async def query_workflow(self, input: Any) -> Any:
        try:
            input.headers = _build_headers(dict(input.headers or {}))
        except Exception as e:
            logger.debug(
                "[Laminar] Could not inject headers into query_workflow: %s", e
            )
        return await self._next.query_workflow(input)

    async def start_workflow_update(self, input: Any) -> Any:
        try:
            input.headers = _build_headers(dict(input.headers or {}))
        except Exception as e:
            logger.debug(
                "[Laminar] Could not inject headers into start_workflow_update: %s", e
            )
        return await self._next.start_workflow_update(input)


class _LaminarActivityInboundInterceptor:
    """Restores Laminar span context from Temporal headers before activity execution."""

    def __init__(
        self,
        next_interceptor: Any,
        *,
        create_activity_span: bool = True,
    ) -> None:
        self._next = next_interceptor
        self._create_activity_span = create_activity_span

    def __getattr__(self, name: str) -> Any:
        return getattr(self._next, name)

    async def execute_activity(self, input: Any) -> Any:
        headers = getattr(input, "headers", None) or {}
        restored = _restore_context_from_headers(dict(headers))

        if restored is None:
            return await self._next.execute_activity(input)

        restored_ctx, ctx_token = restored

        try:
            if not self._create_activity_span:
                return await self._next.execute_activity(input)

            from lmnr.sdk.laminar import Laminar

            activity_name: str = "temporal.activity"
            try:
                import temporalio.activity as _ta

                info = _ta.info()
                activity_name = info.activity_type
            except Exception:
                pass

            with Laminar.start_as_current_span(
                activity_name,
                parent_span_context=restored_ctx,
            ):
                return await self._next.execute_activity(input)
        finally:
            try:
                detach_context(ctx_token)
            except Exception:
                pass


# ─── Auto-patch helpers ────────────────────────────────────────────────────────

def patch_temporal_worker(
    worker_module: Any,
    *,
    create_activity_span: bool = True,
) -> None:
    """
    Patch a `temporalio.worker` module so that every `Worker.create()` call
    automatically includes `LaminarTemporalInterceptor`.

    Called by Laminar.initialize() when `temporal_modules` is provided.
    """
    original_create = worker_module.Worker.create

    @staticmethod  # type: ignore[misc]
    async def patched_create(*args: Any, **kwargs: Any) -> Any:
        interceptors: list[Any] = list(kwargs.pop("interceptors", []) or [])
        interceptors.insert(0, LaminarTemporalInterceptor(
            create_activity_span=create_activity_span,
        ))
        return await original_create(*args, interceptors=interceptors, **kwargs)

    worker_module.Worker.create = patched_create


def patch_temporal_client(
    client_module: Any,
    *,
    create_activity_span: bool = True,
) -> None:
    """
    Patch a `temporalio.client` module so that every `Client.connect()` call
    automatically includes `LaminarTemporalInterceptor`.

    Called by Laminar.initialize() when `temporal_modules` is provided.
    """
    original_connect = client_module.Client.connect

    @staticmethod  # type: ignore[misc]
    async def patched_connect(*args: Any, **kwargs: Any) -> Any:
        interceptors: list[Any] = list(kwargs.pop("interceptors", []) or [])
        interceptors.insert(0, LaminarTemporalInterceptor(
            create_activity_span=create_activity_span,
        ))
        return await original_connect(*args, interceptors=interceptors, **kwargs)

    client_module.Client.connect = patched_connect

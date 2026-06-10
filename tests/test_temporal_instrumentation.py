"""Unit tests for the Laminar Temporal instrumentation.

These exercise the interceptors directly with lightweight fake ``input`` objects
and fake ``next`` interceptors, so no Temporal server (or sandbox) is needed.
The pieces under test are pure: header codec, span-context propagation
(including the nested debug block), client-side header injection, activity-side
context restoration + span creation, and workflow-side header forwarding.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.temporal.consts import (
    LAMINAR_SPAN_CONTEXT_HEADER,
    TRACEPARENT_HEADER,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.temporal.helpers import (
    build_headers,
    decode_payload,
    encode_payload,
    restore_context_from_headers,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.temporal.interceptors import (
    LaminarTemporalInterceptorOptions,
    LaminarTracingInterceptor,
    _LaminarActivityInboundInterceptor,
    _LaminarClientOutboundInterceptor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.temporal.workflow_interceptor import (  # noqa: E501
    LaminarWorkflowInboundInterceptor,
)
from lmnr.sdk.types import DebugContext, LaminarSpanContext


def _span_context(**kwargs) -> LaminarSpanContext:
    """A LaminarSpanContext with a valid 64-bit span id.

    OTel span ids are 64-bit; `LaminarSpanContext` stores them padded into a
    UUID (this is exactly what `get_laminar_span_context` produces from a real
    span). A full random uuid4 span id is out of OTel's 64-bit range and yields
    an INVALID span context that span creation silently drops, so tests that
    assert parent-trace inheritance must mint a valid one.
    """
    return LaminarSpanContext(
        trace_id=uuid.uuid4(),
        span_id=uuid.UUID(int=uuid.uuid4().int & ((1 << 64) - 1)),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeClientNext:
    """Records what the wrapped client interceptor forwarded."""

    def __init__(self, handle=None):
        self.handle = handle or SimpleNamespace()
        self.start_input = None
        self.signal_input = None
        self.query_input = None
        self.update_input = None

    async def start_workflow(self, input):
        self.start_input = input
        return self.handle

    async def signal_workflow(self, input):
        self.signal_input = input

    async def query_workflow(self, input):
        self.query_input = input
        return "query-result"

    async def start_workflow_update(self, input):
        self.update_input = input
        return SimpleNamespace()


class _FakeActivityNext:
    def __init__(self, result="activity-result"):
        self.result = result
        self.executed_input = None

    async def execute_activity(self, input):
        self.executed_input = input
        return self.result


class _RecordingWorkflowNext:
    """Inbound + outbound recorder for the workflow interceptor chain."""

    def __init__(self):
        self.outbound = None
        self.start_activity_input = None
        self.start_local_activity_input = None
        self.start_child_workflow_input = None
        self.continue_as_new_input = None

    # inbound
    def init(self, outbound):
        self.outbound = outbound

    async def execute_workflow(self, input):
        # Schedule an activity from inside the workflow body via the outbound
        # interceptor the chain handed us.
        self.outbound.start_activity(
            SimpleNamespace(activity="act", headers={})
        )
        return "wf-result"

    async def handle_signal(self, input):
        self.outbound.start_activity(
            SimpleNamespace(activity="act", headers={})
        )

    async def handle_update_handler(self, input):
        self.outbound.start_activity(
            SimpleNamespace(activity="act", headers={})
        )
        return "update-result"

    # outbound
    def start_activity(self, input):
        self.start_activity_input = input
        return SimpleNamespace()

    def start_local_activity(self, input):
        self.start_local_activity_input = input
        return SimpleNamespace()

    async def start_child_workflow(self, input):
        self.start_child_workflow_input = input
        return SimpleNamespace()

    def continue_as_new(self, input):
        self.continue_as_new_input = input


# --------------------------------------------------------------------------- #
# helpers / codec
# --------------------------------------------------------------------------- #
def test_payload_roundtrip():
    payload = encode_payload("hello")
    assert payload.metadata["encoding"] == b"json/plain"
    assert decode_payload(payload) == "hello"


def test_decode_payload_none():
    assert decode_payload(None) is None


def test_build_headers_none_context_returns_existing():
    existing = {"k": encode_payload("v")}
    out = build_headers(existing, None)
    assert out == existing
    # A copy, not the same object — callers mutate it.
    assert out is not existing


def test_build_and_restore_roundtrip():
    ctx = _span_context()
    headers = build_headers({}, ctx)
    assert LAMINAR_SPAN_CONTEXT_HEADER in headers
    assert TRACEPARENT_HEADER in headers

    restored = restore_context_from_headers(headers)
    assert restored is not None
    assert restored.trace_id == ctx.trace_id
    assert restored.span_id == ctx.span_id


def test_restore_preserves_debug_block():
    debug = DebugContext(
        enabled=True,
        session_id=str(uuid.uuid4()),
        replay_trace_id=str(uuid.uuid4()),
        cache_until="abcdef",
    )
    ctx = LaminarSpanContext(
        trace_id=uuid.uuid4(), span_id=uuid.uuid4(), debug=debug
    )
    headers = build_headers({}, ctx)
    restored = restore_context_from_headers(headers)
    assert restored is not None
    assert restored.debug is not None
    assert restored.debug.enabled is True
    assert restored.debug.cache_until == "abcdef"


def test_restore_falls_back_to_traceparent():
    trace_id = uuid.uuid4()
    span_hex = uuid.uuid4().hex[16:]
    headers = {
        TRACEPARENT_HEADER: encode_payload(f"00-{trace_id.hex}-{span_hex}-01")
    }
    restored = restore_context_from_headers(headers)
    assert restored is not None
    assert restored.trace_id == trace_id
    assert restored.is_remote is True


def test_restore_empty_headers_returns_none():
    assert restore_context_from_headers({}) is None
    assert restore_context_from_headers(None) is None


# --------------------------------------------------------------------------- #
# client outbound interceptor
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_start_workflow_injects_headers_and_spans(
    span_exporter: InMemorySpanExporter,
):
    span_exporter.clear()
    root = LaminarTracingInterceptor()
    next_ = _FakeClientNext()
    interceptor = _LaminarClientOutboundInterceptor(next_, root)

    handle = SimpleNamespace(
        result=lambda *a, **k: None,
        cancel=lambda *a, **k: None,
        terminate=lambda *a, **k: None,
    )
    next_ = _FakeClientNext(handle=handle)
    interceptor = _LaminarClientOutboundInterceptor(next_, root)

    input = SimpleNamespace(workflow="MyWorkflow", args=[1, 2], headers={})
    returned = await interceptor.start_workflow(input)

    # Headers were injected on the forwarded input.
    assert LAMINAR_SPAN_CONTEXT_HEADER in next_.start_input.headers
    assert TRACEPARENT_HEADER in next_.start_input.headers
    assert returned is handle

    # Restored context from the injected headers should be valid.
    restored = restore_context_from_headers(next_.start_input.headers)
    assert restored is not None

    # The lifecycle span hasn't ended yet (it wraps the handle until result()).
    names = [s.name for s in span_exporter.get_finished_spans()]
    assert "temporal.workflow.MyWorkflow" not in names


@pytest.mark.asyncio
async def test_signal_forwards_active_context(span_exporter: InMemorySpanExporter):
    span_exporter.clear()
    root = LaminarTracingInterceptor()
    next_ = _FakeClientNext()
    interceptor = _LaminarClientOutboundInterceptor(next_, root)

    with Laminar.start_as_current_span("caller"):
        input = SimpleNamespace(signal="sig", headers={})
        await interceptor.signal_workflow(input)

    # Active caller span context was forwarded.
    assert LAMINAR_SPAN_CONTEXT_HEADER in next_.signal_input.headers


@pytest.mark.asyncio
async def test_workflow_handle_lifecycle_span_ends_on_result(
    span_exporter: InMemorySpanExporter,
):
    span_exporter.clear()
    root = LaminarTracingInterceptor()

    async def _orig_result(*a, **k):
        return "the-output"

    handle = SimpleNamespace(
        result=_orig_result,
        cancel=lambda *a, **k: None,
        terminate=lambda *a, **k: None,
    )
    next_ = _FakeClientNext(handle=handle)
    interceptor = _LaminarClientOutboundInterceptor(next_, root)

    input = SimpleNamespace(workflow="W", args=["in"], headers={})
    h = await interceptor.start_workflow(input)
    result = await h.result()
    assert result == "the-output"

    spans = span_exporter.get_finished_spans()
    wf = [s for s in spans if s.name == "temporal.workflow.W"]
    assert len(wf) == 1


# --------------------------------------------------------------------------- #
# activity inbound interceptor
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_activity_restores_context_and_spans(
    span_exporter: InMemorySpanExporter, monkeypatch
):
    span_exporter.clear()
    root = LaminarTracingInterceptor()
    next_ = _FakeActivityNext(result="out")
    interceptor = _LaminarActivityInboundInterceptor(next_, root)

    parent = _span_context()
    headers = build_headers({}, parent)

    monkeypatch.setattr(
        "temporalio.activity.info",
        lambda: SimpleNamespace(activity_type="MyActivity"),
    )

    input = SimpleNamespace(args=["a"], headers=headers)
    result = await interceptor.execute_activity(input)
    assert result == "out"

    spans = span_exporter.get_finished_spans()
    act = [s for s in spans if s.name == "MyActivity"]
    assert len(act) == 1
    # Parented to the propagated remote trace.
    assert act[0].context.trace_id == parent.trace_id.int


@pytest.mark.asyncio
async def test_activity_no_span_when_disabled(
    span_exporter: InMemorySpanExporter, monkeypatch
):
    span_exporter.clear()
    root = LaminarTracingInterceptor(
        LaminarTemporalInterceptorOptions(create_activity_span=False)
    )
    next_ = _FakeActivityNext(result="out")
    interceptor = _LaminarActivityInboundInterceptor(next_, root)

    parent = _span_context()
    headers = build_headers({}, parent)
    input = SimpleNamespace(args=["a"], headers=headers)

    result = await interceptor.execute_activity(input)
    assert result == "out"
    assert next_.executed_input is input
    assert span_exporter.get_finished_spans() == ()


@pytest.mark.asyncio
async def test_activity_no_span_without_context(
    span_exporter: InMemorySpanExporter,
):
    span_exporter.clear()
    root = LaminarTracingInterceptor()
    next_ = _FakeActivityNext(result="out")
    interceptor = _LaminarActivityInboundInterceptor(next_, root)

    input = SimpleNamespace(args=["a"], headers={})
    result = await interceptor.execute_activity(input)
    assert result == "out"
    assert span_exporter.get_finished_spans() == ()


# --------------------------------------------------------------------------- #
# workflow inbound/outbound interceptor (sandbox-side header forwarding)
# --------------------------------------------------------------------------- #
def _wf_interceptor():
    next_ = _RecordingWorkflowNext()
    inbound = LaminarWorkflowInboundInterceptor(next_)
    inbound.init(next_)
    return inbound, next_


@pytest.mark.asyncio
async def test_workflow_forwards_start_headers_to_activity():
    inbound, next_ = _wf_interceptor()
    ctx = _span_context()
    start_headers = build_headers({}, ctx)

    await inbound.execute_workflow(
        SimpleNamespace(headers=start_headers)
    )

    forwarded = next_.start_activity_input.headers
    assert LAMINAR_SPAN_CONTEXT_HEADER in forwarded
    assert forwarded[LAMINAR_SPAN_CONTEXT_HEADER] == start_headers[
        LAMINAR_SPAN_CONTEXT_HEADER
    ]


@pytest.mark.asyncio
async def test_workflow_signal_scopes_handler_headers():
    inbound, next_ = _wf_interceptor()
    start_ctx = _span_context()
    await inbound.execute_workflow(
        SimpleNamespace(headers=build_headers({}, start_ctx))
    )

    signal_ctx = _span_context()
    signal_headers = build_headers({}, signal_ctx)
    await inbound.handle_signal(SimpleNamespace(headers=signal_headers))

    # The activity scheduled inside the signal handler used the SIGNAL trace,
    # not the workflow-start trace.
    forwarded = next_.start_activity_input.headers
    assert (
        forwarded[LAMINAR_SPAN_CONTEXT_HEADER]
        == signal_headers[LAMINAR_SPAN_CONTEXT_HEADER]
    )


@pytest.mark.asyncio
async def test_workflow_signal_without_context_uses_start_headers():
    inbound, next_ = _wf_interceptor()
    start_ctx = _span_context()
    start_headers = build_headers({}, start_ctx)
    await inbound.execute_workflow(SimpleNamespace(headers=start_headers))

    # Signal with no injected trace context — must fall back to start headers.
    await inbound.handle_signal(SimpleNamespace(headers={}))
    forwarded = next_.start_activity_input.headers
    assert (
        forwarded[LAMINAR_SPAN_CONTEXT_HEADER]
        == start_headers[LAMINAR_SPAN_CONTEXT_HEADER]
    )


@pytest.mark.asyncio
async def test_workflow_caller_headers_win_on_merge():
    inbound, next_ = _wf_interceptor()
    ctx = _span_context()
    inbound._start_headers = build_headers({}, ctx)

    # The outbound interceptor the chain built is what user workflow code calls;
    # init() handed it to next_.init, so next_.outbound is our wrapping outbound.
    outbound = next_.outbound
    caller_payload = encode_payload("caller-wins")
    outbound.start_activity(
        SimpleNamespace(
            activity="act",
            headers={LAMINAR_SPAN_CONTEXT_HEADER: caller_payload},
        )
    )

    # Caller-supplied header for the same key wins over the propagated one.
    forwarded = next_.start_activity_input.headers
    assert forwarded[LAMINAR_SPAN_CONTEXT_HEADER] is caller_payload


@pytest.mark.asyncio
async def test_workflow_child_and_continue_as_new_forward_headers():
    inbound, next_ = _wf_interceptor()
    ctx = _span_context()
    inbound._start_headers = build_headers({}, ctx)
    outbound = next_.outbound

    await outbound.start_child_workflow(
        SimpleNamespace(workflow="child", headers={})
    )
    outbound.start_local_activity(
        SimpleNamespace(activity="local", headers={})
    )
    outbound.continue_as_new(SimpleNamespace(workflow="W", headers={}))

    assert LAMINAR_SPAN_CONTEXT_HEADER in next_.start_child_workflow_input.headers
    assert LAMINAR_SPAN_CONTEXT_HEADER in next_.start_local_activity_input.headers
    assert LAMINAR_SPAN_CONTEXT_HEADER in next_.continue_as_new_input.headers

"""
This test primarily tests our patching of the OpenTelemetry context to fix
DataDog's broken Span context.

See lmnr.opentelemetry_lib.opentelemetry.instrumentation.opentelemetry
for more details.
"""

import asyncio
import time
import uuid

import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from typing_extensions import override

from lmnr import Laminar, observe

SPAN_ID = 369


class MyBrokenSpanContext(SpanContext):
    @property
    @override
    def trace_flags(self) -> TraceFlags:
        return 1  # pyright: ignore[reportReturnType] purposefully model a bug ddtrace used to have


def test_broken_span(span_exporter: InMemorySpanExporter):
    if not Laminar.is_initialized():
        Laminar.initialize(project_api_key="test", disable_batch=True)

    with Laminar.start_as_current_span("outer"):
        trace_id = trace.get_current_span().get_span_context().trace_id
        span = NonRecordingSpan(MyBrokenSpanContext(trace_id, SPAN_ID, False))
        ctx = trace.set_span_in_context(span, context.get_current())
        ctx_token = context.attach(ctx)

        with Laminar.start_as_current_span("inner"):
            pass

    span.end()
    time.sleep(0.5)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    inner_span = next(span for span in spans if span.name == "inner")
    outer_span = next(span for span in spans if span.name == "outer")
    assert outer_span.name == "outer"
    outer_attrs = outer_span.attributes or {}
    outer_ctx = outer_span.get_span_context()
    inner_ctx = inner_span.get_span_context()
    assert outer_ctx is not None
    assert inner_ctx is not None
    assert outer_attrs["lmnr.span.instrumentation_source"] == "python"
    assert outer_attrs["lmnr.span.path"] == ("outer",)
    assert outer_attrs["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_ctx.span_id)),
    )

    assert inner_ctx.trace_id == outer_ctx.trace_id

    context.detach(ctx_token)


def test_broken_span_observe(span_exporter: InMemorySpanExporter):
    @observe()
    def test():
        return 1

    @observe()
    def outer():
        trace_id = trace.get_current_span().get_span_context().trace_id
        span = NonRecordingSpan(MyBrokenSpanContext(trace_id, SPAN_ID, False))
        ctx = trace.set_span_in_context(span, context.get_current())
        _token = context.attach(ctx)

        result = test()
        return result

    result = outer()
    time.sleep(0.5)
    assert result == 1

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    inner_span = next(span for span in spans if span.name == "test")
    outer_span = next(span for span in spans if span.name == "outer")
    assert inner_span.name == "test"
    inner_attrs = inner_span.attributes or {}
    outer_attrs = outer_span.attributes or {}
    outer_ctx = outer_span.get_span_context()
    inner_ctx = inner_span.get_span_context()
    assert outer_ctx is not None
    assert inner_ctx is not None
    assert inner_span.parent is not None
    assert inner_attrs["lmnr.span.instrumentation_source"] == "python"
    assert inner_attrs["lmnr.span.path"] == (
        "outer",
        "test",
    )
    assert inner_attrs["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_ctx.span_id)),
        str(uuid.UUID(int=inner_ctx.span_id)),
    )
    assert inner_span.parent.span_id == outer_ctx.span_id
    assert inner_span.parent.trace_flags.sampled

    assert outer_span.name == "outer"
    assert outer_attrs["lmnr.span.instrumentation_source"] == "python"
    assert outer_attrs["lmnr.span.path"] == ("outer",)
    assert outer_attrs["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_ctx.span_id)),
    )

    assert inner_ctx.trace_id == outer_ctx.trace_id


@pytest.mark.asyncio
async def test_broken_span_observe_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def test():
        return 1

    @observe()
    async def outer():
        trace_id = trace.get_current_span().get_span_context().trace_id
        span = NonRecordingSpan(MyBrokenSpanContext(trace_id, SPAN_ID, False))
        ctx = trace.set_span_in_context(span, context.get_current())
        _token = context.attach(ctx)

        result = await test()
        return result

    result = await outer()
    await asyncio.sleep(0.5)
    assert result == 1

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    inner_span = next(span for span in spans if span.name == "test")
    outer_span = next(span for span in spans if span.name == "outer")
    assert inner_span.name == "test"
    inner_attrs = inner_span.attributes or {}
    outer_attrs = outer_span.attributes or {}
    outer_ctx = outer_span.get_span_context()
    inner_ctx = inner_span.get_span_context()
    assert outer_ctx is not None
    assert inner_ctx is not None
    assert inner_span.parent is not None
    assert inner_attrs["lmnr.span.instrumentation_source"] == "python"
    assert inner_attrs["lmnr.span.path"] == (
        "outer",
        "test",
    )
    assert inner_attrs["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_ctx.span_id)),
        str(uuid.UUID(int=inner_ctx.span_id)),
    )
    assert inner_span.parent.span_id == outer_ctx.span_id
    assert inner_span.parent.trace_flags.sampled

    assert outer_span.name == "outer"
    assert outer_attrs["lmnr.span.instrumentation_source"] == "python"
    assert outer_attrs["lmnr.span.path"] == ("outer",)
    assert outer_attrs["lmnr.span.ids_path"] == (
        str(uuid.UUID(int=outer_ctx.span_id)),
    )

    assert inner_ctx.trace_id == outer_ctx.trace_id

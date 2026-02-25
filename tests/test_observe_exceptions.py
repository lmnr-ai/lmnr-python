"""
Tests verifying that all internal Laminar decorator failures are silently swallowed
and never propagate to the end-user. Each group targets a specific try/except block
in opentelemetry_lib/decorators/__init__.py and sdk/decorators.py.
"""

import pytest
from unittest.mock import patch, PropertyMock

import opentelemetry.context as otel_context

from lmnr import observe
from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.opentelemetry_lib.tracing.span import LaminarSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


# =============================================================================
# _setup_span failures (get_tracer_with_context raises)
# The outer try/except in _setup_span returns None; wrap falls back to fn(*args).
# Result: 0 spans, function still returns correctly.
# =============================================================================


def test_setup_span_failure_sync(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        return "foo"

    with patch(
        "lmnr.opentelemetry_lib.decorators.get_tracer_with_context",
        side_effect=RuntimeError("tracer exploded"),
    ):
        result = observed_foo()

    assert result == "foo"
    assert len(span_exporter.get_finished_spans()) == 0


@pytest.mark.asyncio
async def test_setup_span_failure_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    with patch(
        "lmnr.opentelemetry_lib.decorators.get_tracer_with_context",
        side_effect=RuntimeError("tracer exploded"),
    ):
        result = await observed_foo()

    assert result == "foo"
    assert len(span_exporter.get_finished_spans()) == 0


# =============================================================================
# TracerWrapper() construction failures
# The try/except around `wrapper = TracerWrapper()` falls back to fn(*args).
# Result: 0 spans, function still returns correctly.
# =============================================================================


def test_tracer_wrapper_creation_failure_sync(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        return "foo"

    with patch("lmnr.opentelemetry_lib.decorators.TracerWrapper") as mock_tw:
        mock_tw.verify_initialized.return_value = True
        mock_tw.side_effect = RuntimeError("wrapper exploded")
        result = observed_foo()

    assert result == "foo"
    assert len(span_exporter.get_finished_spans()) == 0


@pytest.mark.asyncio
async def test_tracer_wrapper_creation_failure_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    with patch("lmnr.opentelemetry_lib.decorators.TracerWrapper") as mock_tw:
        mock_tw.verify_initialized.return_value = True
        mock_tw.side_effect = RuntimeError("wrapper exploded")
        result = await observed_foo()

    assert result == "foo"
    assert len(span_exporter.get_finished_spans()) == 0


# =============================================================================
# _process_input failures — get_input_from_func_args raises
# The try/except inside _process_input swallows it.
# Result: 1 span without lmnr.span.input, function returns correctly.
# =============================================================================


def test_process_input_get_func_args_failure_sync(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo(x):
        return "foo"

    with patch(
        "lmnr.opentelemetry_lib.decorators.get_input_from_func_args",
        side_effect=RuntimeError("args exploded"),
    ):
        result = observed_foo(42)

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert "lmnr.span.input" not in spans[0].attributes


@pytest.mark.asyncio
async def test_process_input_get_func_args_failure_async(
    span_exporter: InMemorySpanExporter,
):
    @observe()
    async def observed_foo(x):
        return "foo"

    with patch(
        "lmnr.opentelemetry_lib.decorators.get_input_from_func_args",
        side_effect=RuntimeError("args exploded"),
    ):
        result = await observed_foo(42)

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert "lmnr.span.input" not in spans[0].attributes


# =============================================================================
# _process_input failures — span.set_input raises
# Same try/except swallows it.
# Result: 1 span without lmnr.span.input.
# =============================================================================


def test_process_input_set_input_failure_sync(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo(x):
        return "foo"

    with patch.object(LaminarSpan, "set_input", side_effect=RuntimeError("set_input exploded")):
        result = observed_foo(42)

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert "lmnr.span.input" not in spans[0].attributes


@pytest.mark.asyncio
async def test_process_input_set_input_failure_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo(x):
        return "foo"

    with patch.object(LaminarSpan, "set_input", side_effect=RuntimeError("set_input exploded")):
        result = await observed_foo(42)

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert "lmnr.span.input" not in spans[0].attributes


# =============================================================================
# _process_output failures — span.set_output raises
# The try/except inside _process_output swallows it.
# Result: 1 span without lmnr.span.output, function returns correctly.
# =============================================================================


def test_process_output_set_output_failure_sync(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        return "foo"

    with patch.object(LaminarSpan, "set_output", side_effect=RuntimeError("set_output exploded")):
        result = observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert "lmnr.span.output" not in spans[0].attributes


@pytest.mark.asyncio
async def test_process_output_set_output_failure_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    with patch.object(LaminarSpan, "set_output", side_effect=RuntimeError("set_output exploded")):
        result = await observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert "lmnr.span.output" not in spans[0].attributes


# =============================================================================
# _cleanup_span failures — span.end() raises
# The try/except inside _cleanup_span swallows it.
# span.end() is never reached on the SDK span so it is not exported, but the
# user sees no exception and gets the correct return value.
# =============================================================================


def test_cleanup_span_end_failure_sync(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        return "foo"

    with patch.object(LaminarSpan, "end", side_effect=RuntimeError("end exploded")):
        result = observed_foo()

    assert result == "foo"


@pytest.mark.asyncio
async def test_cleanup_span_end_failure_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    with patch.object(LaminarSpan, "end", side_effect=RuntimeError("end exploded")):
        result = await observed_foo()

    assert result == "foo"


# =============================================================================
# _cleanup_span failures — wrapper.pop_span_context() raises
# span.end() has already run and exported the span; the subsequent
# pop_span_context failure is caught by the same try/except in _cleanup_span.
# Result: 1 span, function returns correctly.
# =============================================================================


def test_cleanup_span_pop_context_failure_sync(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        return "foo"

    with patch.object(
        TracerWrapper, "pop_span_context", side_effect=RuntimeError("pop exploded")
    ):
        result = observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


@pytest.mark.asyncio
async def test_cleanup_span_pop_context_failure_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    with patch.object(
        TracerWrapper, "pop_span_context", side_effect=RuntimeError("pop exploded")
    ):
        result = await observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


# =============================================================================
# Context setup failures — wrapper.push_span_context raises
# The outer try/except around the entire context-setup block catches it.
# Execution proceeds; the span is still created and properly closed.
# Result: 1 span, function returns correctly.
# =============================================================================


def test_push_span_context_failure_sync(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        return "foo"

    with patch.object(
        TracerWrapper, "push_span_context", side_effect=RuntimeError("push exploded")
    ):
        result = observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


@pytest.mark.asyncio
async def test_push_span_context_failure_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    with patch.object(
        TracerWrapper, "push_span_context", side_effect=RuntimeError("push exploded")
    ):
        result = await observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


# =============================================================================
# Context setup failures — context_api.attach raises
# Same outer try/except catches it; ctx_token stays None.
# Result: 1 span, function returns correctly.
# =============================================================================


def test_context_api_attach_failure_sync(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        return "foo"

    with patch.object(otel_context, "attach", side_effect=RuntimeError("otel attach exploded")):
        result = observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


@pytest.mark.asyncio
async def test_context_api_attach_failure_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    with patch.object(otel_context, "attach", side_effect=RuntimeError("otel attach exploded")):
        result = await observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


# =============================================================================
# Context setup failures — lmnr isolated attach_context raises
# Same outer try/except catches it; isolated_ctx_token stays None.
# Result: 1 span, function returns correctly.
# =============================================================================


def test_attach_context_failure_sync(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        return "foo"

    with patch(
        "lmnr.opentelemetry_lib.decorators.attach_context",
        side_effect=RuntimeError("attach_context exploded"),
    ):
        result = observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


@pytest.mark.asyncio
async def test_attach_context_failure_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    with patch(
        "lmnr.opentelemetry_lib.decorators.attach_context",
        side_effect=RuntimeError("attach_context exploded"),
    ):
        result = await observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


# =============================================================================
# Context teardown failures — context_api.detach raises in finally block
# The individual try/except around context_api.detach(ctx_token) catches it.
# Result: 1 span, function returns correctly.
# =============================================================================


def test_context_api_detach_failure_sync(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        return "foo"

    with patch.object(otel_context, "detach", side_effect=RuntimeError("otel detach exploded")):
        result = observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


@pytest.mark.asyncio
async def test_context_api_detach_failure_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    with patch.object(otel_context, "detach", side_effect=RuntimeError("otel detach exploded")):
        result = await observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


# =============================================================================
# Context teardown failures — lmnr detach_context raises in finally block
# The individual try/except around detach_context(isolated_ctx_token) catches it.
# Result: 1 span, function returns correctly.
# =============================================================================


def test_detach_context_failure_sync(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        return "foo"

    with patch(
        "lmnr.opentelemetry_lib.decorators.detach_context",
        side_effect=RuntimeError("detach_context exploded"),
    ):
        result = observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


@pytest.mark.asyncio
async def test_detach_context_failure_async(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        return "foo"

    with patch(
        "lmnr.opentelemetry_lib.decorators.detach_context",
        side_effect=RuntimeError("detach_context exploded"),
    ):
        result = await observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


# =============================================================================
# _process_exception failures — span.record_exception raises
# The try/except inside _process_exception catches it.
# The *user's* original exception is still re-raised as expected.
# Result: 1 span (created + ended), user's ValueError propagates.
# =============================================================================


def test_process_exception_record_exception_failure_sync(
    span_exporter: InMemorySpanExporter,
):
    @observe()
    def observed_foo():
        raise ValueError("user error")

    with patch.object(
        LaminarSpan, "record_exception", side_effect=RuntimeError("record_exception exploded")
    ):
        with pytest.raises(ValueError, match="user error"):
            observed_foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


@pytest.mark.asyncio
async def test_process_exception_record_exception_failure_async(
    span_exporter: InMemorySpanExporter,
):
    @observe()
    async def observed_foo():
        raise ValueError("user error")

    with patch.object(
        LaminarSpan, "record_exception", side_effect=RuntimeError("record_exception exploded")
    ):
        with pytest.raises(ValueError, match="user error"):
            await observed_foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


# =============================================================================
# _process_exception failures — span.set_status raises
# Same try/except inside _process_exception catches it.
# The user's exception is still re-raised.
# =============================================================================


def test_process_exception_set_status_failure_sync(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        raise ValueError("user error")

    with patch.object(
        LaminarSpan, "set_status", side_effect=RuntimeError("set_status exploded")
    ):
        with pytest.raises(ValueError, match="user error"):
            observed_foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


@pytest.mark.asyncio
async def test_process_exception_set_status_failure_async(
    span_exporter: InMemorySpanExporter,
):
    @observe()
    async def observed_foo():
        raise ValueError("user error")

    with patch.object(
        LaminarSpan, "set_status", side_effect=RuntimeError("set_status exploded")
    ):
        with pytest.raises(ValueError, match="user error"):
            await observed_foo()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


# =============================================================================
# metadata.copy() failure at decoration time (sdk/decorators.py)
# The try/except around `metadata.copy()` catches it; merged_metadata stays {}.
# Result: 1 span without metadata attributes, function returns correctly.
# =============================================================================


class _BadCopyMetadata(dict):
    """Simulates a broken metadata object whose .copy() always raises."""

    def copy(self):
        raise RuntimeError("copy exploded deliberately")

    def __bool__(self):
        return True


def test_metadata_copy_failure_sync(span_exporter: InMemorySpanExporter):
    @observe(metadata=_BadCopyMetadata())
    def observed_foo():
        return "foo"

    result = observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert not any(
        k.startswith("lmnr.association.properties.metadata") for k in spans[0].attributes
    )


@pytest.mark.asyncio
async def test_metadata_copy_failure_async(span_exporter: InMemorySpanExporter):
    @observe(metadata=_BadCopyMetadata())
    async def observed_foo():
        return "foo"

    result = await observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert not any(
        k.startswith("lmnr.association.properties.metadata") for k in spans[0].attributes
    )


# =============================================================================
# set_association_props_in_context internal failures
# The try/except inside set_association_props_in_context returns None;
# the caller (wrap) handles None gracefully.
# Result: 1 span, function returns correctly.
# =============================================================================


def test_set_association_props_internal_failure_sync(span_exporter: InMemorySpanExporter):
    @observe(session_id="123")
    def observed_foo():
        return "foo"

    with patch.object(
        LaminarSpan,
        "laminar_association_properties",
        new_callable=PropertyMock,
        side_effect=RuntimeError("props access exploded"),
    ):
        result = observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


@pytest.mark.asyncio
async def test_set_association_props_internal_failure_async(
    span_exporter: InMemorySpanExporter,
):
    @observe(session_id="123")
    async def observed_foo():
        return "foo"

    with patch.object(
        LaminarSpan,
        "laminar_association_properties",
        new_callable=PropertyMock,
        side_effect=RuntimeError("props access exploded"),
    ):
        result = await observed_foo()

    assert result == "foo"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


# =============================================================================
# Generator: _cleanup_span failures — pop_span_context raises in generator finally
# span.end() runs first (exports the span); pop_span_context failure is swallowed.
# Result: 1 span, all values yielded, no exception.
# =============================================================================


def test_generator_cleanup_pop_context_failure(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        yield "foo"
        yield "bar"

    with patch.object(
        TracerWrapper, "pop_span_context", side_effect=RuntimeError("pop exploded")
    ):
        results = list(observed_foo())

    assert results == ["foo", "bar"]
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


@pytest.mark.asyncio
async def test_async_generator_cleanup_pop_context_failure(
    span_exporter: InMemorySpanExporter,
):
    @observe()
    async def observed_foo():
        yield "foo"
        yield "bar"

    with patch.object(
        TracerWrapper, "pop_span_context", side_effect=RuntimeError("pop exploded")
    ):
        results = [r async for r in observed_foo()]

    assert results == ["foo", "bar"]
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"


# =============================================================================
# Generator: _process_output failures — set_output raises in generator finally
# The try/except inside _process_output swallows it.
# Result: 1 span without lmnr.span.output, all values yielded, no exception.
# =============================================================================


def test_generator_process_output_failure(span_exporter: InMemorySpanExporter):
    @observe()
    def observed_foo():
        yield "foo"
        yield "bar"

    with patch.object(LaminarSpan, "set_output", side_effect=RuntimeError("set_output exploded")):
        results = list(observed_foo())

    assert results == ["foo", "bar"]
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert "lmnr.span.output" not in spans[0].attributes


@pytest.mark.asyncio
async def test_async_generator_process_output_failure(span_exporter: InMemorySpanExporter):
    @observe()
    async def observed_foo():
        yield "foo"
        yield "bar"

    with patch.object(LaminarSpan, "set_output", side_effect=RuntimeError("set_output exploded")):
        results = [r async for r in observed_foo()]

    assert results == ["foo", "bar"]
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "observed_foo"
    assert "lmnr.span.output" not in spans[0].attributes

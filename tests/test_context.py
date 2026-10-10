from contextvars import Token

from opentelemetry.context import Context
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.span import INVALID_SPAN_ID

from lmnr import Laminar, observe

TokenStack = tuple[Token[Context], ...]
Result = tuple[Context, TokenStack, TokenStack, TokenStack]


def test_clear_context_observe(span_exporter: InMemorySpanExporter):
    @observe()
    def inner():
        Laminar.set_trace_user_id("test_user_id_2")
        return "inner"

    @observe()
    def outer():
        Laminar.set_trace_user_id("test_user_id_1")
        _success = Laminar.force_flush()
        return inner()

    _ = outer()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    outer_span = next(s for s in spans if s.name == "outer")
    inner_span = next(s for s in spans if s.name == "inner")
    assert (outer_span.attributes or {})[
        "lmnr.association.properties.user_id"
    ] == "test_user_id_1"
    assert (inner_span.attributes or {})[
        "lmnr.association.properties.user_id"
    ] == "test_user_id_2"

    assert inner_span.parent is None or inner_span.parent.span_id == INVALID_SPAN_ID
    inner_context = inner_span.get_span_context()
    outer_context = outer_span.get_span_context()
    assert inner_context is not None
    assert outer_context is not None
    assert inner_context.trace_id != outer_context.trace_id


def test_clear_context_start_as_current_span(span_exporter: InMemorySpanExporter):
    with Laminar.start_as_current_span("outer"):
        Laminar.set_trace_user_id("test_user_id_1")
        _flush_success = Laminar.force_flush()
        with Laminar.start_as_current_span("inner"):
            Laminar.set_trace_user_id("test_user_id_2")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    outer_span = next(s for s in spans if s.name == "outer")
    inner_span = next(s for s in spans if s.name == "inner")
    assert (outer_span.attributes or {})[
        "lmnr.association.properties.user_id"
    ] == "test_user_id_1"
    assert (inner_span.attributes or {})[
        "lmnr.association.properties.user_id"
    ] == "test_user_id_2"

    assert inner_span.parent is None or inner_span.parent.span_id == INVALID_SPAN_ID
    inner_context = inner_span.get_span_context()
    outer_context = outer_span.get_span_context()
    assert inner_context is not None
    assert outer_context is not None
    assert inner_context.trace_id != outer_context.trace_id


def test_clear_context_start_active_span(span_exporter: InMemorySpanExporter):
    span = Laminar.start_active_span("outer")
    Laminar.set_trace_user_id("test_user_id_1")
    _ = Laminar.force_flush()
    span2 = Laminar.start_active_span("inner")
    Laminar.set_trace_user_id("test_user_id_2")
    span2.end()
    span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    outer_span = next(s for s in spans if s.name == "outer")
    inner_span = next(s for s in spans if s.name == "inner")
    assert (outer_span.attributes or {})[
        "lmnr.association.properties.user_id"
    ] == "test_user_id_1"
    assert (inner_span.attributes or {})[
        "lmnr.association.properties.user_id"
    ] == "test_user_id_2"

    assert inner_span.parent is None or inner_span.parent.span_id == INVALID_SPAN_ID
    inner_context = inner_span.get_span_context()
    outer_context = outer_span.get_span_context()
    assert inner_context is not None
    assert outer_context is not None
    assert inner_context.trace_id != outer_context.trace_id


def test_isolated_context_is_readable_from_a_fresh_contextvars_context():
    """The ContextVars need real defaults: a value set at import time only exists
    in the importing context, so a fresh context (a thread not created through
    the patched Thread.__init__, an executor, ...) would raise LookupError."""
    import _thread
    import contextvars
    import threading

    from lmnr.opentelemetry_lib.tracing.context import (
        get_current_context,
        get_token_stack,
        pop_span_context,
        push_span_context,
    )

    def exercise() -> Result:
        before = get_token_stack()
        push_span_context(Context())
        pushed = get_token_stack()
        pop_span_context()
        return get_current_context(), before, pushed, get_token_stack()

    def check(result: Result | BaseException):
        assert not isinstance(result, BaseException), result
        context, before, pushed, after = result
        assert context == Context()
        assert before == ()
        assert len(pushed) == 1
        assert after == ()

    check(contextvars.Context().run(exercise))

    # `_thread` bypasses the patched `threading.Thread.__init__`, so the new
    # thread starts with an empty contextvars context.
    results: list[Result | BaseException] = []
    done = threading.Event()

    def target():
        try:
            results.append(exercise())
        except Exception as e:
            results.append(e)
        finally:
            done.set()

    _thread_id = _thread.start_new_thread(target, ())
    assert done.wait(5)
    check(results[0])

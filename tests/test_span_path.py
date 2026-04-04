"""
Comprehensive tests for span_path and span_ids_path propagation in LaminarSpanProcessor.

Covers:
  1. Regular span path propagation (nested spans)
  2. Spans inheriting from an already ended span
  3. Continuing from a remote span (LaminarSpanContext)
  4. Multiple traces in parallel
  5. Corner cases (deeply nested, empty paths, etc.)
"""
import uuid
import threading
import concurrent.futures

from lmnr import Laminar
from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.sdk.types import LaminarSpanContext
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


# ---------- helpers ----------

def _span_by_name(spans, name):
    matches = [s for s in spans if s.name == name]
    assert len(matches) == 1, f"Expected 1 span named '{name}', found {len(matches)}"
    return matches[0]


def _span_uuid(span) -> str:
    return str(uuid.UUID(int=span.get_span_context().span_id))


# ==========================================================================
# 1. Regular span path propagation
# ==========================================================================


def test_single_span_path(span_exporter: InMemorySpanExporter):
    """A single root span should have path=[name] and ids_path=[own_id]."""
    with Laminar.start_as_current_span("root"):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    root = spans[0]
    assert root.attributes["lmnr.span.path"] == ("root",)
    assert root.attributes["lmnr.span.ids_path"] == (_span_uuid(root),)


def test_two_level_nested_path(span_exporter: InMemorySpanExporter):
    """Parent -> child nesting should accumulate paths."""
    with Laminar.start_as_current_span("parent"):
        with Laminar.start_as_current_span("child"):
            pass

    spans = span_exporter.get_finished_spans()
    parent = _span_by_name(spans, "parent")
    child = _span_by_name(spans, "child")

    assert parent.attributes["lmnr.span.path"] == ("parent",)
    assert child.attributes["lmnr.span.path"] == ("parent", "child")
    assert child.attributes["lmnr.span.ids_path"] == (
        _span_uuid(parent),
        _span_uuid(child),
    )


def test_three_level_nested_path(span_exporter: InMemorySpanExporter):
    """Grandparent -> parent -> child nesting should accumulate."""
    with Laminar.start_as_current_span("gp"):
        with Laminar.start_as_current_span("p"):
            with Laminar.start_as_current_span("c"):
                pass

    spans = span_exporter.get_finished_spans()
    gp = _span_by_name(spans, "gp")
    p = _span_by_name(spans, "p")
    c = _span_by_name(spans, "c")

    assert gp.attributes["lmnr.span.path"] == ("gp",)
    assert p.attributes["lmnr.span.path"] == ("gp", "p")
    assert c.attributes["lmnr.span.path"] == ("gp", "p", "c")
    assert c.attributes["lmnr.span.ids_path"] == (
        _span_uuid(gp),
        _span_uuid(p),
        _span_uuid(c),
    )


def test_siblings_share_parent_path(span_exporter: InMemorySpanExporter):
    """Sibling spans under same parent should share the same prefix."""
    with Laminar.start_as_current_span("parent"):
        with Laminar.start_as_current_span("child_a"):
            pass
        with Laminar.start_as_current_span("child_b"):
            pass

    spans = span_exporter.get_finished_spans()
    parent = _span_by_name(spans, "parent")
    a = _span_by_name(spans, "child_a")
    b = _span_by_name(spans, "child_b")

    assert a.attributes["lmnr.span.path"] == ("parent", "child_a")
    assert b.attributes["lmnr.span.path"] == ("parent", "child_b")
    # Both share the parent's id in their ids_path
    assert a.attributes["lmnr.span.ids_path"][0] == _span_uuid(parent)
    assert b.attributes["lmnr.span.ids_path"][0] == _span_uuid(parent)


def test_start_span_manual_path(span_exporter: InMemorySpanExporter):
    """start_span (non-context-managed) should also build correct paths."""
    with Laminar.start_as_current_span("parent"):
        span = Laminar.start_span("manual_child")
        span.end()

    spans = span_exporter.get_finished_spans()
    parent = _span_by_name(spans, "parent")
    child = _span_by_name(spans, "manual_child")

    assert child.attributes["lmnr.span.path"] == ("parent", "manual_child")
    assert child.attributes["lmnr.span.ids_path"] == (
        _span_uuid(parent),
        _span_uuid(child),
    )


def test_start_active_span_path(span_exporter: InMemorySpanExporter):
    """start_active_span should correctly set up paths for children."""
    outer = Laminar.start_active_span("outer")
    with Laminar.start_as_current_span("inner"):
        pass
    outer.end()

    spans = span_exporter.get_finished_spans()
    outer_span = _span_by_name(spans, "outer")
    inner_span = _span_by_name(spans, "inner")

    assert outer_span.attributes["lmnr.span.path"] == ("outer",)
    assert inner_span.attributes["lmnr.span.path"] == ("outer", "inner")
    assert inner_span.attributes["lmnr.span.ids_path"] == (
        _span_uuid(outer_span),
        _span_uuid(inner_span),
    )


# ==========================================================================
# 2. Spans inheriting from an already ended span
# ==========================================================================


def test_child_from_ended_parent_via_context(span_exporter: InMemorySpanExporter):
    """A child span should inherit path from an already-ended parent
    when parent_span_context is passed from LaminarSpanContext."""
    parent = Laminar.start_span("ended_parent")
    parent_ctx = Laminar.get_laminar_span_context(parent)
    parent.end()

    # Parent is now ended. Start a child using its context.
    with Laminar.start_as_current_span("child", parent_span_context=parent_ctx):
        pass

    spans = span_exporter.get_finished_spans()
    parent_span = _span_by_name(spans, "ended_parent")
    child_span = _span_by_name(spans, "child")

    assert child_span.attributes["lmnr.span.path"] == ("ended_parent", "child")
    assert child_span.attributes["lmnr.span.ids_path"] == (
        _span_uuid(parent_span),
        _span_uuid(child_span),
    )


def test_child_from_ended_parent_via_serialized_context(span_exporter: InMemorySpanExporter):
    """Same as above but using serialized/deserialized context (simulates cross-service)."""
    parent = Laminar.start_span("ser_parent")
    serialized = Laminar.serialize_span_context(parent)
    parent.end()

    # Clear processor cache to simulate cross-service scenario
    TracerWrapper.instance._span_processor.clear()

    deserialized = Laminar.deserialize_span_context(serialized)
    with Laminar.start_as_current_span("remote_child", parent_span_context=deserialized):
        pass

    spans = span_exporter.get_finished_spans()
    parent_span = _span_by_name(spans, "ser_parent")
    child_span = _span_by_name(spans, "remote_child")

    assert child_span.attributes["lmnr.span.path"] == ("ser_parent", "remote_child")
    assert child_span.attributes["lmnr.span.ids_path"] == (
        _span_uuid(parent_span),
        _span_uuid(child_span),
    )


def test_grandchild_from_ended_parent(span_exporter: InMemorySpanExporter):
    """Grandchild should still have correct paths even if intermediate parent ended."""
    root = Laminar.start_span("root")
    root_ctx = Laminar.get_laminar_span_context(root)
    root.end()

    with Laminar.start_as_current_span("mid", parent_span_context=root_ctx):
        with Laminar.start_as_current_span("leaf"):
            pass

    spans = span_exporter.get_finished_spans()
    root_span = _span_by_name(spans, "root")
    mid_span = _span_by_name(spans, "mid")
    leaf_span = _span_by_name(spans, "leaf")

    assert leaf_span.attributes["lmnr.span.path"] == ("root", "mid", "leaf")
    assert leaf_span.attributes["lmnr.span.ids_path"] == (
        _span_uuid(root_span),
        _span_uuid(mid_span),
        _span_uuid(leaf_span),
    )


# ==========================================================================
# 3. Continuing from a remote span (LaminarSpanContext)
# ==========================================================================


def test_remote_span_context_path(span_exporter: InMemorySpanExporter):
    """A remote LaminarSpanContext should supply parent path for local span."""
    remote_trace_id = "01234567-89ab-cdef-0123-456789abcdef"
    remote_span_id = "00000000-0000-0000-fedc-ba9876543210"
    remote_ctx = LaminarSpanContext(
        trace_id=remote_trace_id,
        span_id=remote_span_id,
        is_remote=True,
        span_path=["service_a", "handler"],
        span_ids_path=["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", remote_span_id],
    )

    with Laminar.start_as_current_span("local_span", parent_span_context=remote_ctx):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    local = spans[0]

    assert local.attributes["lmnr.span.path"] == ("service_a", "handler", "local_span")
    assert local.attributes["lmnr.span.ids_path"] == (
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        remote_span_id,
        _span_uuid(local),
    )
    # The trace_id should be from the remote context
    assert local.get_span_context().trace_id == uuid.UUID(remote_trace_id).int


def test_remote_span_context_nested_children(span_exporter: InMemorySpanExporter):
    """Children nested under a remote-context span should inherit the full path."""
    remote_ctx = LaminarSpanContext(
        trace_id="11111111-1111-1111-1111-111111111111",
        span_id="22222222-2222-2222-2222-222222222222",
        is_remote=True,
        span_path=["remote_root"],
        span_ids_path=["22222222-2222-2222-2222-222222222222"],
    )

    with Laminar.start_as_current_span("local_parent", parent_span_context=remote_ctx):
        with Laminar.start_as_current_span("local_child"):
            pass

    spans = span_exporter.get_finished_spans()
    parent = _span_by_name(spans, "local_parent")
    child = _span_by_name(spans, "local_child")

    assert parent.attributes["lmnr.span.path"] == ("remote_root", "local_parent")
    assert child.attributes["lmnr.span.path"] == ("remote_root", "local_parent", "local_child")
    assert child.attributes["lmnr.span.ids_path"] == (
        "22222222-2222-2222-2222-222222222222",
        _span_uuid(parent),
        _span_uuid(child),
    )


def test_remote_span_context_serialized_roundtrip(span_exporter: InMemorySpanExporter):
    """Test path propagation across full serialize/deserialize cycle."""
    # Create a local span, serialize its context
    with Laminar.start_as_current_span("origin"):
        ctx_str = Laminar.serialize_span_context()

    # Simulate a remote service receiving the serialized context
    # Clear processor to simulate different process
    TracerWrapper.instance._span_processor.clear()

    remote_ctx = Laminar.deserialize_span_context(ctx_str)
    with Laminar.start_as_current_span("remote_handler", parent_span_context=remote_ctx):
        with Laminar.start_as_current_span("remote_inner"):
            pass

    spans = span_exporter.get_finished_spans()
    origin = _span_by_name(spans, "origin")
    handler = _span_by_name(spans, "remote_handler")
    inner = _span_by_name(spans, "remote_inner")

    assert handler.attributes["lmnr.span.path"] == ("origin", "remote_handler")
    assert inner.attributes["lmnr.span.path"] == ("origin", "remote_handler", "remote_inner")
    assert inner.attributes["lmnr.span.ids_path"] == (
        _span_uuid(origin),
        _span_uuid(handler),
        _span_uuid(inner),
    )


def test_remote_span_context_with_start_span(span_exporter: InMemorySpanExporter):
    """Remote context via start_span (non-context-managed)."""
    remote_ctx = LaminarSpanContext(
        trace_id="33333333-3333-3333-3333-333333333333",
        span_id="44444444-4444-4444-4444-444444444444",
        is_remote=True,
        span_path=["remote_svc"],
        span_ids_path=["44444444-4444-4444-4444-444444444444"],
    )

    span = Laminar.start_span("local_child", parent_span_context=remote_ctx)
    span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    local = spans[0]

    assert local.attributes["lmnr.span.path"] == ("remote_svc", "local_child")
    assert local.attributes["lmnr.span.ids_path"] == (
        "44444444-4444-4444-4444-444444444444",
        _span_uuid(local),
    )


# ==========================================================================
# 4. Multiple traces in parallel
# ==========================================================================


def test_parallel_traces_have_independent_paths(span_exporter: InMemorySpanExporter):
    """Concurrent traces should not interfere with each other's paths."""
    results = {}
    barrier = threading.Barrier(2)

    def trace_a():
        barrier.wait()
        with Laminar.start_as_current_span("trace_a_root"):
            with Laminar.start_as_current_span("trace_a_child"):
                pass

    def trace_b():
        barrier.wait()
        with Laminar.start_as_current_span("trace_b_root"):
            with Laminar.start_as_current_span("trace_b_child"):
                pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        fa = executor.submit(trace_a)
        fb = executor.submit(trace_b)
        fa.result()
        fb.result()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4

    a_root = _span_by_name(spans, "trace_a_root")
    a_child = _span_by_name(spans, "trace_a_child")
    b_root = _span_by_name(spans, "trace_b_root")
    b_child = _span_by_name(spans, "trace_b_child")

    # Trace A paths
    assert a_root.attributes["lmnr.span.path"] == ("trace_a_root",)
    assert a_child.attributes["lmnr.span.path"] == ("trace_a_root", "trace_a_child")
    assert a_child.attributes["lmnr.span.ids_path"] == (
        _span_uuid(a_root),
        _span_uuid(a_child),
    )

    # Trace B paths
    assert b_root.attributes["lmnr.span.path"] == ("trace_b_root",)
    assert b_child.attributes["lmnr.span.path"] == ("trace_b_root", "trace_b_child")
    assert b_child.attributes["lmnr.span.ids_path"] == (
        _span_uuid(b_root),
        _span_uuid(b_child),
    )

    # Different trace IDs
    assert a_root.get_span_context().trace_id != b_root.get_span_context().trace_id


def test_many_parallel_traces(span_exporter: InMemorySpanExporter):
    """Multiple (10) concurrent traces should maintain correct paths."""
    n_traces = 10
    barrier = threading.Barrier(n_traces)

    def run_trace(i):
        barrier.wait()
        with Laminar.start_as_current_span(f"root_{i}"):
            with Laminar.start_as_current_span(f"child_{i}"):
                pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_traces) as executor:
        futures = [executor.submit(run_trace, i) for i in range(n_traces)]
        for f in futures:
            f.result()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2 * n_traces

    for i in range(n_traces):
        root = _span_by_name(spans, f"root_{i}")
        child = _span_by_name(spans, f"child_{i}")
        assert root.attributes["lmnr.span.path"] == (f"root_{i}",)
        assert child.attributes["lmnr.span.path"] == (f"root_{i}", f"child_{i}")
        assert child.attributes["lmnr.span.ids_path"] == (
            _span_uuid(root),
            _span_uuid(child),
        )


# ==========================================================================
# 5. Corner cases
# ==========================================================================


def test_deeply_nested_spans(span_exporter: InMemorySpanExporter):
    """Test deeply nested spans (10 levels) accumulate path correctly."""
    depth = 10
    names = [f"level_{i}" for i in range(depth)]

    def nest(remaining_names):
        if not remaining_names:
            return
        with Laminar.start_as_current_span(remaining_names[0]):
            nest(remaining_names[1:])

    nest(names)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == depth

    for i, name in enumerate(names):
        span = _span_by_name(spans, name)
        expected_path = tuple(names[:i + 1])
        assert span.attributes["lmnr.span.path"] == expected_path, (
            f"Span '{name}' path mismatch: {span.attributes['lmnr.span.path']} != {expected_path}"
        )


def test_root_span_ids_path_is_single_entry(span_exporter: InMemorySpanExporter):
    """A root span's ids_path should contain exactly its own span ID."""
    with Laminar.start_as_current_span("lone_root"):
        pass

    spans = span_exporter.get_finished_spans()
    root = spans[0]
    assert len(root.attributes["lmnr.span.ids_path"]) == 1
    assert root.attributes["lmnr.span.ids_path"][0] == _span_uuid(root)


def test_two_sequential_traces_have_clean_paths(span_exporter: InMemorySpanExporter):
    """Sequential traces should not leak paths from one to the next."""
    with Laminar.start_as_current_span("trace1_root"):
        with Laminar.start_as_current_span("trace1_child"):
            pass

    span_exporter.clear()
    TracerWrapper.clear()

    with Laminar.start_as_current_span("trace2_root"):
        with Laminar.start_as_current_span("trace2_child"):
            pass

    spans = span_exporter.get_finished_spans()
    root = _span_by_name(spans, "trace2_root")
    child = _span_by_name(spans, "trace2_child")

    # trace2 should NOT have trace1 paths
    assert root.attributes["lmnr.span.path"] == ("trace2_root",)
    assert child.attributes["lmnr.span.path"] == ("trace2_root", "trace2_child")


def test_use_span_path_propagation(span_exporter: InMemorySpanExporter):
    """use_span should correctly propagate paths to children."""
    outer = Laminar.start_span("outer")
    with Laminar.use_span(outer):
        with Laminar.start_as_current_span("inner"):
            pass
    outer.end()

    spans = span_exporter.get_finished_spans()
    outer_span = _span_by_name(spans, "outer")
    inner_span = _span_by_name(spans, "inner")

    assert inner_span.attributes["lmnr.span.path"] == ("outer", "inner")
    assert inner_span.attributes["lmnr.span.ids_path"] == (
        _span_uuid(outer_span),
        _span_uuid(inner_span),
    )


def test_multiple_children_after_parent_ended(span_exporter: InMemorySpanExporter):
    """Multiple children created from an ended parent should all inherit correctly."""
    parent = Laminar.start_span("parent")
    ctx = Laminar.get_laminar_span_context(parent)
    parent.end()

    for i in range(3):
        with Laminar.start_as_current_span(f"child_{i}", parent_span_context=ctx):
            pass

    spans = span_exporter.get_finished_spans()
    parent_span = _span_by_name(spans, "parent")

    for i in range(3):
        child = _span_by_name(spans, f"child_{i}")
        assert child.attributes["lmnr.span.path"] == ("parent", f"child_{i}")
        assert child.attributes["lmnr.span.ids_path"] == (
            _span_uuid(parent_span),
            _span_uuid(child),
        )


def test_empty_remote_span_path(span_exporter: InMemorySpanExporter):
    """A remote context with empty span_path should still create a valid local path."""
    remote_ctx = LaminarSpanContext(
        trace_id="55555555-5555-5555-5555-555555555555",
        span_id="66666666-6666-6666-6666-666666666666",
        is_remote=True,
        span_path=[],
        span_ids_path=[],
    )

    with Laminar.start_as_current_span("local", parent_span_context=remote_ctx):
        pass

    spans = span_exporter.get_finished_spans()
    local = spans[0]
    assert local.attributes["lmnr.span.path"] == ("local",)
    assert local.attributes["lmnr.span.ids_path"] == (_span_uuid(local),)

import asyncio
import concurrent.futures
import pytest
import threading
import time

from lmnr import Laminar
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


# =============================================================================
# ASYNCIO CONCURRENCY TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_asyncio_parallel_spans_separate_traces(exporter: InMemorySpanExporter):
    """Test multiple parallel async spans live in separate traces."""

    async def task_a():
        with Laminar.start_as_current_span("task_a"):
            await asyncio.sleep(0.01)
            Laminar.set_span_output("task_a")
            return "task_a"

    async def task_b():
        with Laminar.start_as_current_span("task_b"):
            await asyncio.sleep(0.01)
            Laminar.set_span_output("task_b")
            return "task_b"

    async def task_c():
        with Laminar.start_as_current_span("task_c"):
            await asyncio.sleep(0.01)
            Laminar.set_span_output("task_c")
            return "task_c"

    # Run tasks concurrently
    results = await asyncio.gather(task_a(), task_b(), task_c())

    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    assert set(results) == {"task_a", "task_b", "task_c"}

    # Check all spans have different trace IDs (separate traces)
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 3, "All spans should have different trace IDs"

    # Check no span has a parent (all are root spans)
    for span in spans:
        assert span.parent is None or span.parent.span_id == 0


@pytest.mark.asyncio
async def test_asyncio_parallel_spans_same_parent(exporter: InMemorySpanExporter):
    """Test multiple parallel async spans within one parent share the same trace."""

    async def child_task(task_id: str):
        with Laminar.start_as_current_span("child_task"):
            await asyncio.sleep(0.01)
            result = f"child_{task_id}"
            Laminar.set_span_output(result)
            return result

    async def parent_task():
        with Laminar.start_as_current_span("parent_task"):
            Laminar.set_trace_session_id("parent_session")
            # Run child tasks concurrently within the parent context
            results = await asyncio.gather(
                child_task("a"), child_task("b"), child_task("c")
            )
            Laminar.set_span_output(results)
            return results

    result = await parent_task()

    spans = exporter.get_finished_spans()
    assert len(spans) == 4  # 1 parent + 3 children
    assert result == ["child_a", "child_b", "child_c"]

    # Find parent and child spans
    parent_span = [s for s in spans if s.name == "parent_task"][0]
    child_spans = [s for s in spans if s.name == "child_task"]

    assert len(child_spans) == 3
    assert (
        parent_span.attributes["lmnr.association.properties.session_id"]
        == "parent_session"
    )

    # Check all spans share the same trace ID
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 1, "All spans should share the same trace ID"

    # Check all child spans have the parent as their parent
    for child_span in child_spans:
        assert child_span.parent.span_id == parent_span.context.span_id


@pytest.mark.asyncio
async def test_asyncio_deeply_nested_with_parallelism(exporter: InMemorySpanExporter):
    """Test deeply nested async spans with some parallelism."""

    async def leaf_task(task_id: str):
        with Laminar.start_as_current_span("leaf_task"):
            await asyncio.sleep(0.01)
            result = f"leaf_{task_id}"
            Laminar.set_span_output(result)
            return result

    async def branch_task(branch_id: str):
        with Laminar.start_as_current_span("branch_task"):
            # Each branch runs some leaf tasks in parallel
            results = await asyncio.gather(
                leaf_task(f"{branch_id}_1"), leaf_task(f"{branch_id}_2")
            )
            Laminar.set_span_output(results)
            return results

    async def root_task():
        with Laminar.start_as_current_span("root_task"):
            Laminar.set_trace_session_id("root_session")
            # Run multiple branches in parallel
            results = await asyncio.gather(
                branch_task("branch_a"), branch_task("branch_b")
            )
            Laminar.set_span_output(results)
            return results

    await root_task()

    spans = exporter.get_finished_spans()
    assert len(spans) == 7  # 1 root + 2 branches + 4 leaves

    # Find spans by name
    root_span = [s for s in spans if s.name == "root_task"][0]
    branch_spans = [s for s in spans if s.name == "branch_task"]
    leaf_spans = [s for s in spans if s.name == "leaf_task"]

    assert len(branch_spans) == 2
    assert len(leaf_spans) == 4

    # All spans should be in the same trace
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 1

    # Check hierarchy
    for branch_span in branch_spans:
        assert branch_span.parent.span_id == root_span.context.span_id

    # Each leaf should have a branch as parent
    for leaf_span in leaf_spans:
        assert leaf_span.parent.span_id in [b.context.span_id for b in branch_spans]


# =============================================================================
# THREADING.THREAD CONCURRENCY TESTS
# =============================================================================


def test_threading_parallel_spans_separate_traces(exporter: InMemorySpanExporter):
    """Test multiple parallel thread spans live in separate traces."""

    results = []

    def task_worker(task_id: str):
        with Laminar.start_as_current_span("task_worker"):
            time.sleep(0.01)
            result = f"task_{task_id}"
            results.append(result)
            Laminar.set_span_output(result)
            return result

    # Create and start threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=task_worker, args=(str(i),))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    assert len(results) == 3

    # Check all spans have different trace IDs (separate traces)
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 3, "All spans should have different trace IDs"

    # Check no span has a parent (all are root spans)
    for span in spans:
        assert span.parent is None or span.parent.span_id == 0


def test_threading_parallel_spans_same_parent(exporter: InMemorySpanExporter):
    """Test multiple parallel thread spans within one parent share the same trace."""

    child_results = []

    def child_worker(task_id: str):
        with Laminar.start_as_current_span("child_worker"):
            time.sleep(0.01)
            result = f"child_{task_id}"
            child_results.append(result)
            Laminar.set_span_output(result)
            return result

    def parent_task():
        with Laminar.start_as_current_span("parent_task"):
            Laminar.set_trace_session_id("parent_session")
            # Create child threads within parent context
            threads = []
            for i in range(3):
                thread = threading.Thread(target=child_worker, args=(str(i),))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            Laminar.set_span_output(child_results)
            return child_results

    parent_task()

    spans = exporter.get_finished_spans()
    assert len(spans) == 4  # 1 parent + 3 children
    assert len(child_results) == 3

    # Find parent and child spans
    parent_span = [s for s in spans if s.name == "parent_task"][0]
    child_spans = [s for s in spans if s.name == "child_worker"]

    assert len(child_spans) == 3
    assert (
        parent_span.attributes["lmnr.association.properties.session_id"]
        == "parent_session"
    )

    # Check all spans share the same trace ID
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 1, "All spans should share the same trace ID"

    # Check all child spans have the parent as their parent
    for child_span in child_spans:
        assert child_span.parent.span_id == parent_span.context.span_id


def test_threading_deeply_nested_with_parallelism(exporter: InMemorySpanExporter):
    """Test deeply nested thread spans with some parallelism."""

    leaf_results = []

    def leaf_worker(task_id: str):
        with Laminar.start_as_current_span("leaf_worker"):
            time.sleep(0.01)
            result = f"leaf_{task_id}"
            leaf_results.append(result)
            Laminar.set_span_output(result)
            return result

    def branch_worker(branch_id: str):
        with Laminar.start_as_current_span("branch_worker"):
            # Each branch creates leaf threads
            threads = []
            for i in range(2):
                thread = threading.Thread(
                    target=leaf_worker, args=(f"{branch_id}_{i}",)
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            result = f"branch_{branch_id}"
            Laminar.set_span_output(result)
            return result

    def root_task():
        with Laminar.start_as_current_span("root_task"):
            Laminar.set_trace_session_id("root_session")
            # Create branch threads
            threads = []
            for i in range(2):
                thread = threading.Thread(target=branch_worker, args=(str(i),))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            result = "root_done"
            Laminar.set_span_output(result)
            return result

    root_task()

    spans = exporter.get_finished_spans()
    assert len(spans) == 7  # 1 root + 2 branches + 4 leaves

    # Find spans by name
    root_span = [s for s in spans if s.name == "root_task"][0]
    branch_spans = [s for s in spans if s.name == "branch_worker"]
    leaf_spans = [s for s in spans if s.name == "leaf_worker"]

    assert len(branch_spans) == 2
    assert len(leaf_spans) == 4

    # All spans should be in the same trace
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 1

    # Check hierarchy
    for branch_span in branch_spans:
        assert branch_span.parent.span_id == root_span.context.span_id

    # Each leaf should have a branch as parent
    for leaf_span in leaf_spans:
        assert leaf_span.parent.span_id in [b.context.span_id for b in branch_spans]


# =============================================================================
# THREADPOOLEXECUTOR CONCURRENCY TESTS
# =============================================================================


def test_threadpool_parallel_spans_separate_traces(exporter: InMemorySpanExporter):
    """Test multiple parallel ThreadPoolExecutor spans live in separate traces."""

    def task_worker(task_id: str):
        with Laminar.start_as_current_span("task_worker"):
            time.sleep(0.01)
            result = f"task_{task_id}"
            Laminar.set_span_output(result)
            return result

    # Use ThreadPoolExecutor to run tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(task_worker, str(i)) for i in range(3)]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    assert len(results) == 3

    # Check all spans have different trace IDs (separate traces)
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 3, "All spans should have different trace IDs"

    # Check no span has a parent (all are root spans)
    for span in spans:
        assert span.parent is None or span.parent.span_id == 0


def test_threadpool_parallel_spans_same_parent(exporter: InMemorySpanExporter):
    """Test multiple parallel ThreadPoolExecutor spans within one parent share the same trace."""

    def child_worker(task_id: str):
        with Laminar.start_as_current_span("child_worker"):
            time.sleep(0.01)
            result = f"child_{task_id}"
            Laminar.set_span_output(result)
            return result

    def parent_task():
        with Laminar.start_as_current_span("parent_task"):
            Laminar.set_trace_session_id("parent_session")
            # Use ThreadPoolExecutor within parent context
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(child_worker, str(i)) for i in range(3)]
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
            Laminar.set_span_output(results)
            return results

    result = parent_task()

    spans = exporter.get_finished_spans()
    assert len(spans) == 4  # 1 parent + 3 children
    assert len(result) == 3

    # Find parent and child spans
    parent_span = [s for s in spans if s.name == "parent_task"][0]
    child_spans = [s for s in spans if s.name == "child_worker"]

    assert len(child_spans) == 3
    assert (
        parent_span.attributes["lmnr.association.properties.session_id"]
        == "parent_session"
    )

    # Check all spans share the same trace ID
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 1, "All spans should share the same trace ID"

    # Check all child spans have the parent as their parent
    for child_span in child_spans:
        assert child_span.parent.span_id == parent_span.context.span_id


def test_threadpool_deeply_nested_with_parallelism(exporter: InMemorySpanExporter):
    """Test deeply nested ThreadPoolExecutor spans with some parallelism."""

    def leaf_worker(task_id: str):
        with Laminar.start_as_current_span("leaf_worker"):
            time.sleep(0.01)
            result = f"leaf_{task_id}"
            Laminar.set_span_output(result)
            return result

    def branch_worker(branch_id: str):
        with Laminar.start_as_current_span("branch_worker"):
            # Each branch uses ThreadPoolExecutor for leaf tasks
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(leaf_worker, f"{branch_id}_{i}") for i in range(2)
                ]
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
            Laminar.set_span_output(results)
            return results

    def root_task():
        with Laminar.start_as_current_span("root_task"):
            Laminar.set_trace_session_id("root_session")
            # Use ThreadPoolExecutor for branch tasks
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(branch_worker, str(i)) for i in range(2)]
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
            Laminar.set_span_output(results)
            return results

    root_task()

    spans = exporter.get_finished_spans()
    assert len(spans) == 7  # 1 root + 2 branches + 4 leaves

    # Find spans by name
    root_span = [s for s in spans if s.name == "root_task"][0]
    branch_spans = [s for s in spans if s.name == "branch_worker"]
    leaf_spans = [s for s in spans if s.name == "leaf_worker"]

    assert len(branch_spans) == 2
    assert len(leaf_spans) == 4

    # All spans should be in the same trace
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 1

    # Check hierarchy
    for branch_span in branch_spans:
        assert branch_span.parent.span_id == root_span.context.span_id

    # Each leaf should have a branch as parent
    for leaf_span in leaf_spans:
        assert leaf_span.parent.span_id in [b.context.span_id for b in branch_spans]


# =============================================================================
# MIXED CONCURRENCY TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_mixed_concurrency_isolation(exporter: InMemorySpanExporter):
    """Test that different concurrency models don't interfere with each other."""

    async_result = []
    thread_result = []

    async def async_task():
        with Laminar.start_as_current_span("async_task"):
            await asyncio.sleep(0.01)
            async_result.append("async_done")
            Laminar.set_span_output("async_task")
            return "async_task"

    def thread_task():
        with Laminar.start_as_current_span("thread_task"):
            time.sleep(0.01)
            thread_result.append("thread_done")
            Laminar.set_span_output("thread_task")
            return "thread_task"

    def threadpool_task():
        with Laminar.start_as_current_span("threadpool_task"):
            time.sleep(0.01)
            Laminar.set_span_output("threadpool_task")
            return "threadpool_task"

    # Run different concurrency models simultaneously
    async_future = asyncio.create_task(async_task())

    thread = threading.Thread(target=thread_task)
    thread.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        threadpool_future = executor.submit(threadpool_task)
        threadpool_future.result()

    await async_future
    thread.join()

    spans = exporter.get_finished_spans()
    assert len(spans) == 3

    # Each should be in a separate trace
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 3, "All spans should have different trace IDs"

    # All should be root spans
    for span in spans:
        assert span.parent is None or span.parent.span_id == 0

    # Check span names
    span_names = {span.name for span in spans}
    assert span_names == {"async_task", "thread_task", "threadpool_task"}

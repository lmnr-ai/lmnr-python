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
async def test_asyncio_parallel_spans_separate_traces(
    span_exporter: InMemorySpanExporter,
):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    assert set(results) == {"task_a", "task_b", "task_c"}

    # Check all spans have different trace IDs (separate traces)
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 3, "All spans should have different trace IDs"

    # Check no span has a parent (all are root spans)
    for span in spans:
        assert span.parent is None or span.parent.span_id == 0


@pytest.mark.asyncio
async def test_asyncio_parallel_spans_same_parent(span_exporter: InMemorySpanExporter):
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

    spans = span_exporter.get_finished_spans()
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
async def test_asyncio_deeply_nested_with_parallelism(
    span_exporter: InMemorySpanExporter,
):
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

    spans = span_exporter.get_finished_spans()
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


def test_threading_parallel_spans_separate_traces(span_exporter: InMemorySpanExporter):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    assert len(results) == 3

    # Check all spans have different trace IDs (separate traces)
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 3, "All spans should have different trace IDs"

    # Check no span has a parent (all are root spans)
    for span in spans:
        assert span.parent is None or span.parent.span_id == 0


def test_threading_parallel_spans_same_parent(span_exporter: InMemorySpanExporter):
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

    spans = span_exporter.get_finished_spans()
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


def test_threading_deeply_nested_with_parallelism(span_exporter: InMemorySpanExporter):
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

    spans = span_exporter.get_finished_spans()
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


def test_threadpool_parallel_spans_separate_traces(span_exporter: InMemorySpanExporter):
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

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    assert len(results) == 3

    # Check all spans have different trace IDs (separate traces)
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 3, "All spans should have different trace IDs"

    # Check no span has a parent (all are root spans)
    for span in spans:
        assert span.parent is None or span.parent.span_id == 0


@pytest.mark.vcr(record_mode="once")
def test_threadpool_parallel_spans_with_openai(span_exporter: InMemorySpanExporter):
    """Test multiple parallel ThreadPoolExecutor spans live in separate traces
    including auto-instrumented OpenAI spans."""
    from openai import OpenAI

    openai_client = OpenAI()

    def task_worker(task_id: str):
        with Laminar.start_as_current_span("task_worker"):
            time.sleep(0.01)
            result = f"task_{task_id}"
            openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "what is the capital of France?"}
                ],
            )
            Laminar.set_span_output(result)
            return result

    # Use ThreadPoolExecutor to run tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(task_worker, str(i)) for i in range(3)]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 6
    assert len(results) == 3

    # There's one trace per thread
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 3, "Number of traces should match number of threads"

    for span in spans:
        if span.name == "task_worker":
            assert span.parent is None or span.parent.span_id == 0
        else:
            assert span.parent is not None


def test_threadpool_parallel_spans_same_parent(span_exporter: InMemorySpanExporter):
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

    spans = span_exporter.get_finished_spans()
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


def test_threadpool_deeply_nested_with_parallelism(span_exporter: InMemorySpanExporter):
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

    spans = span_exporter.get_finished_spans()
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


def test_threadpool_exception_handling(span_exporter: InMemorySpanExporter):
    """Test that exceptions in ThreadPoolExecutor tasks are properly handled and spans are still created."""

    def failing_worker(task_id: str):
        with Laminar.start_as_current_span("failing_worker"):
            time.sleep(0.01)
            Laminar.set_span_output(f"task_{task_id}_started")
            if task_id == "fail":
                raise ValueError(f"Task {task_id} failed")
            return f"task_{task_id}_success"

    def success_worker(task_id: str):
        with Laminar.start_as_current_span("success_worker"):
            time.sleep(0.01)
            result = f"task_{task_id}_success"
            Laminar.set_span_output(result)
            return result

    def parent_task():
        with Laminar.start_as_current_span("parent_task"):
            Laminar.set_trace_session_id("exception_session")
            results = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Submit both failing and successful tasks
                futures = [
                    executor.submit(failing_worker, "fail"),
                    executor.submit(success_worker, "success1"),
                    executor.submit(success_worker, "success2"),
                ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except ValueError as e:
                        results.append(f"error: {str(e)}")

            Laminar.set_span_output(results)
            return results

    result = parent_task()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4  # 1 parent + 1 failing + 2 success
    assert len(result) == 3

    # Find spans by name
    parent_span = [s for s in spans if s.name == "parent_task"][0]
    failing_spans = [s for s in spans if s.name == "failing_worker"]
    success_spans = [s for s in spans if s.name == "success_worker"]

    assert len(failing_spans) == 1
    assert len(success_spans) == 2

    # All spans should be in the same trace
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert failing_spans[0].parent.span_id == parent_span.context.span_id
    assert success_spans[0].parent.span_id == parent_span.context.span_id
    assert len(set(trace_ids)) == 1

    # Check that error is in results
    assert any("error:" in str(r) for r in result)
    assert any("success" in str(r) for r in result)


def test_threadpool_multiple_executors(span_exporter: InMemorySpanExporter):
    """Test using multiple ThreadPoolExecutor instances in parallel."""

    def worker_type_a(task_id: str):
        with Laminar.start_as_current_span("worker_type_a"):
            time.sleep(0.01)
            result = f"type_a_{task_id}"
            Laminar.set_span_output(result)
            return result

    def worker_type_b(task_id: str):
        with Laminar.start_as_current_span("worker_type_b"):
            time.sleep(0.01)
            result = f"type_b_{task_id}"
            Laminar.set_span_output(result)
            return result

    def parent_task():
        with Laminar.start_as_current_span("parent_task"):
            Laminar.set_trace_session_id("multi_executor_session")
            results = []

            # Use two separate executors
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor_a:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor_b:
                    # Submit tasks to both executors
                    futures_a = [
                        executor_a.submit(worker_type_a, str(i)) for i in range(2)
                    ]
                    futures_b = [
                        executor_b.submit(worker_type_b, str(i)) for i in range(2)
                    ]

                    # Collect results from both executors
                    all_futures = futures_a + futures_b
                    for future in concurrent.futures.as_completed(all_futures):
                        results.append(future.result())

            Laminar.set_span_output(results)
            return results

    result = parent_task()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 5  # 1 parent + 2 type_a + 2 type_b
    assert len(result) == 4

    # Find spans by name
    parent_span = [s for s in spans if s.name == "parent_task"][0]
    type_a_spans = [s for s in spans if s.name == "worker_type_a"]
    type_b_spans = [s for s in spans if s.name == "worker_type_b"]

    assert len(type_a_spans) == 2
    assert len(type_b_spans) == 2

    # All spans should be in the same trace
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 1

    # Check all worker spans have the parent as their parent
    for span in type_a_spans + type_b_spans:
        assert span.parent.span_id == parent_span.context.span_id


def test_threadpool_context_propagation_with_metadata(
    span_exporter: InMemorySpanExporter,
):
    """Test that complex context and metadata propagate correctly through ThreadPoolExecutor."""

    def metadata_worker(task_id: str, metadata: dict):
        with Laminar.start_as_current_span("metadata_worker") as span:
            time.sleep(0.01)
            # Set various span attributes
            for key, value in metadata.items():
                span.set_attribute(f"custom.{key}", value)

            result = f"task_{task_id}_with_metadata"
            Laminar.set_span_output(result)
            return result

    def parent_task():
        with Laminar.start_as_current_span("parent_task") as parent_span:
            Laminar.set_trace_session_id("metadata_session")
            parent_span.set_attribute("parent.type", "metadata_test")
            parent_span.set_attribute("parent.worker_count", 3)

            metadata_list = [
                {"worker_id": 0, "batch": "A", "priority": "high"},
                {"worker_id": 1, "batch": "B", "priority": "medium"},
                {"worker_id": 2, "batch": "A", "priority": "low"},
            ]

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(metadata_worker, str(i), metadata_list[i])
                    for i in range(3)
                ]
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]

            Laminar.set_span_output(results)
            return results

    result = parent_task()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4  # 1 parent + 3 workers
    assert len(result) == 3

    # Find spans by name
    parent_span = [s for s in spans if s.name == "parent_task"][0]
    worker_spans = [s for s in spans if s.name == "metadata_worker"]

    assert len(worker_spans) == 3

    # Check parent attributes
    assert parent_span.attributes["parent.type"] == "metadata_test"
    assert parent_span.attributes["parent.worker_count"] == 3

    # Check worker attributes
    for worker_span in worker_spans:
        assert "custom.worker_id" in worker_span.attributes
        assert "custom.batch" in worker_span.attributes
        assert "custom.priority" in worker_span.attributes
        assert worker_span.parent.span_id == parent_span.context.span_id

    # All spans should be in the same trace
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 1


def test_threadpool_executor_reuse_across_contexts(span_exporter: InMemorySpanExporter):
    """Test reusing the same ThreadPoolExecutor across different span contexts."""

    def reusable_worker(task_id: str, context_name: str):
        with Laminar.start_as_current_span("reusable_worker"):
            time.sleep(0.01)
            result = f"{context_name}_task_{task_id}"
            Laminar.set_span_output(result)
            return result

    # Create a shared executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as shared_executor:

        # First context
        def context_one():
            with Laminar.start_as_current_span("context_one"):
                Laminar.set_trace_session_id("reuse_session_1")
                futures = [
                    shared_executor.submit(reusable_worker, str(i), "context1")
                    for i in range(2)
                ]
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
                Laminar.set_span_output(results)
                return results

        # Second context (separate trace)
        def context_two():
            with Laminar.start_as_current_span("context_two"):
                Laminar.set_trace_session_id("reuse_session_2")
                futures = [
                    shared_executor.submit(reusable_worker, str(i), "context2")
                    for i in range(2)
                ]
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
                Laminar.set_span_output(results)
                return results

        result1 = context_one()
        result2 = context_two()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 6  # 2 contexts + 4 workers (2 per context)
    assert len(result1) == 2
    assert len(result2) == 2

    # Find spans by name
    context_one_spans = [s for s in spans if s.name == "context_one"]
    context_two_spans = [s for s in spans if s.name == "context_two"]
    worker_spans = [s for s in spans if s.name == "reusable_worker"]

    assert len(context_one_spans) == 1
    assert len(context_two_spans) == 1
    assert len(worker_spans) == 4

    # Should have 2 separate traces (one for each context)
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 2, "Should have 2 separate traces"

    # Check that each context has its workers
    context_one_trace_id = context_one_spans[0].get_span_context().trace_id
    context_two_trace_id = context_two_spans[0].get_span_context().trace_id

    context_one_workers = [
        s for s in worker_spans if s.get_span_context().trace_id == context_one_trace_id
    ]
    context_two_workers = [
        s for s in worker_spans if s.get_span_context().trace_id == context_two_trace_id
    ]

    assert len(context_one_workers) == 2
    assert len(context_two_workers) == 2


def test_threadpool_sequential_vs_parallel_execution(
    span_exporter: InMemorySpanExporter,
):
    """Test ThreadPoolExecutor with sequential execution vs parallel execution patterns."""

    def sequential_worker(task_id: str, batch_id: str):
        with Laminar.start_as_current_span("sequential_worker"):
            time.sleep(0.01)
            result = f"sequential_{batch_id}_{task_id}"
            Laminar.set_span_output(result)
            return result

    def parallel_worker(task_id: str):
        with Laminar.start_as_current_span("parallel_worker"):
            time.sleep(0.01)
            result = f"parallel_{task_id}"
            Laminar.set_span_output(result)
            return result

    def parent_task():
        with Laminar.start_as_current_span("parent_task"):
            Laminar.set_trace_session_id("execution_pattern_session")
            all_results = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Sequential batches
                for batch in ["batch_a", "batch_b"]:
                    futures = [
                        executor.submit(sequential_worker, str(i), batch)
                        for i in range(2)
                    ]
                    # Wait for this batch to complete before starting next
                    batch_results = [future.result() for future in futures]
                    all_results.extend(batch_results)

                # Parallel execution
                parallel_futures = [
                    executor.submit(parallel_worker, str(i)) for i in range(3)
                ]
                parallel_results = [
                    future.result()
                    for future in concurrent.futures.as_completed(parallel_futures)
                ]
                all_results.extend(parallel_results)

            Laminar.set_span_output(all_results)
            return all_results

    result = parent_task()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 8  # 1 parent + 4 sequential + 3 parallel
    assert len(result) == 7

    # Find spans by name
    parent_span = [s for s in spans if s.name == "parent_task"][0]
    sequential_spans = [s for s in spans if s.name == "sequential_worker"]
    parallel_spans = [s for s in spans if s.name == "parallel_worker"]

    assert len(sequential_spans) == 4
    assert len(parallel_spans) == 3

    # All spans should be in the same trace
    trace_ids = [span.get_span_context().trace_id for span in spans]
    assert len(set(trace_ids)) == 1

    # Check all worker spans have the parent as their parent
    for span in sequential_spans + parallel_spans:
        assert span.parent.span_id == parent_span.context.span_id

    # Verify we have the expected results
    sequential_results = [r for r in result if "sequential_" in r]
    parallel_results = [r for r in result if "parallel_" in r]
    assert len(sequential_results) == 4
    assert len(parallel_results) == 3


# =============================================================================
# MIXED CONCURRENCY TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_mixed_concurrency_isolation(span_exporter: InMemorySpanExporter):
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

    spans = span_exporter.get_finished_spans()
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

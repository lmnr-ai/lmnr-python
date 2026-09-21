import asyncio
import pytest
import threading
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor, as_completed

from lmnr import Laminar
from lmnr.opentelemetry_lib import tracing as tracing_mod
from lmnr.opentelemetry_lib.tracing import (
    flush_tracing,
    force_reinit_processor,
    get_tracer_wrapper,
    init_tracing,
    is_tracing_initialized,
    reset_tracing,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class TestTracerWrapperRaceCondition:
    """Test suite for reproducing and testing TracerWrapper race conditions.

    These tests are designed to FAIL when race conditions occur, so we can
    work backwards from the failures to fix the threading issues.
    """

    def setup_method(self):
        """Setup method to clear instances before each test."""
        self._clear_tracer_instances()

    def teardown_method(self):
        """Teardown method to clean up after each test."""
        self._clear_tracer_instances()

    def _clear_tracer_instances(self):
        """Drop the tracer singleton to ensure clean state."""
        reset_tracing()

    def test_concurrent_initialization_and_flush_must_succeed(self):
        """Test that concurrent initialization and flush operations succeed without race conditions."""
        results = []
        exceptions = []

        def initialize_and_flush(thread_id):
            """Simulate initialization and immediate flush."""
            try:
                exporter = InMemorySpanExporter()

                init_tracing(
                    project_api_key=f"race-test-key-{thread_id}",
                    disable_batch=True,
                    exporter=exporter,
                    timeout_seconds=1,
                )

                result = flush_tracing()
                results.append(f"Thread-{thread_id}: {result}")
                return result

            except AttributeError as e:
                exception_msg = f"Thread-{thread_id}: RACE CONDITION - {str(e)}"
                exceptions.append(exception_msg)
                raise AssertionError(exception_msg) from e
            except Exception as e:
                exception_msg = f"Thread-{thread_id}: {type(e).__name__} - {str(e)}"
                exceptions.append(exception_msg)
                raise AssertionError(exception_msg) from e

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(initialize_and_flush, i) for i in range(10)]

            for future in as_completed(futures):
                try:
                    future.result()
                except AssertionError:
                    # print debugging info before re-raising
                    print(f"Results so far: {results}")
                    print(f"Exceptions: {exceptions}")
                    raise

        assert (
            len(results) == 10
        ), f"Expected 10 successful results, got {len(results)}: {results}"
        assert (
            len(exceptions) == 0
        ), f"No exceptions should occur, but got: {exceptions}"

    def test_rapid_concurrent_laminar_operations(self):
        """Test rapid concurrent Laminar operations that commonly cause race conditions.

        This simulates the evaluation pattern where multiple evaluations run concurrently.
        """
        results = []
        exceptions = []

        def rapid_laminar_operations(thread_id):
            """Perform rapid Laminar operations."""
            try:
                # Rapid initialization and flush cycle
                Laminar.initialize(
                    project_api_key=f"rapid-test-{thread_id}",
                    disable_batch=True,
                )

                # Immediate flush without delay - this is where race conditions occur
                result = Laminar.flush()
                results.append(f"Rapid-{thread_id}: Success")
                return result

            except AttributeError as e:
                if "span_processor" in str(e):
                    exception_msg = (
                        f"Rapid-{thread_id}: RACE CONDITION DETECTED - {str(e)}"
                    )
                    exceptions.append(exception_msg)
                    raise AssertionError(exception_msg) from e
                raise
            except Exception as e:
                exception_msg = f"Rapid-{thread_id}: UNEXPECTED ERROR - {type(e).__name__}: {str(e)}"
                exceptions.append(exception_msg)
                raise AssertionError(exception_msg) from e

        # Use more threads and no delay to maximize race condition probability
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(rapid_laminar_operations, i)
                for i in range(15)  # More operations to increase race probability
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except AssertionError:
                    print(f"Rapid test results: {results}")
                    print(f"Rapid test exceptions: {exceptions}")
                    raise

        assert (
            len(results) == 15
        ), f"Expected 15 successful operations, got {len(results)}"
        assert len(exceptions) == 0, f"No race conditions should occur: {exceptions}"

    @pytest.mark.asyncio
    async def test_rapid_concurrent_laminar_operations_async(self):
        """Test rapid concurrent Laminar operations using asyncio that commonly cause race conditions.

        This simulates the async evaluation pattern where multiple evaluations run concurrently
        using asyncio instead of threading.
        """
        results = []
        exceptions = []

        async def rapid_laminar_operations_async(task_id):
            """Perform rapid Laminar operations asynchronously."""
            try:
                # Rapid initialization and flush cycle
                Laminar.initialize(
                    project_api_key=f"rapid-async-test-{task_id}",
                    disable_batch=True,
                )

                # Immediate flush without delay - this is where race conditions occur
                result = Laminar.flush()
                results.append(f"RapidAsync-{task_id}: Success")
                return result

            except AttributeError as e:
                if "span_processor" in str(e):
                    exception_msg = (
                        f"RapidAsync-{task_id}: RACE CONDITION DETECTED - {str(e)}"
                    )
                    exceptions.append(exception_msg)
                    raise AssertionError(exception_msg) from e
                raise
            except Exception as e:
                exception_msg = f"RapidAsync-{task_id}: UNEXPECTED ERROR - {type(e).__name__}: {str(e)}"
                exceptions.append(exception_msg)
                raise AssertionError(exception_msg) from e

        # Create many concurrent async tasks to maximize race condition probability
        tasks = [
            rapid_laminar_operations_async(i)
            for i in range(15)  # Same number as threaded version
        ]

        try:
            await asyncio.gather(*tasks)
        except AssertionError:
            print(f"Rapid async test results: {results}")
            print(f"Rapid async test exceptions: {exceptions}")
            raise

        assert (
            len(results) == 15
        ), f"Expected 15 successful async operations, got {len(results)}"
        assert (
            len(exceptions) == 0
        ), f"No race conditions should occur in async version: {exceptions}"

    @pytest.mark.asyncio
    async def test_async_threading_race_conditions(self):
        """Test async + threading combinations that can cause race conditions."""
        results = []
        exceptions = []

        def threaded_init_flush(worker_id):
            """Worker function that performs init/flush in a thread."""
            try:
                exporter = InMemorySpanExporter()
                init_tracing(
                    project_api_key=f"async-thread-{worker_id}",
                    disable_batch=True,
                    exporter=exporter,
                    timeout_seconds=1,
                )

                # Immediate flush to trigger race condition
                result = flush_tracing()
                results.append(f"AsyncThread-{worker_id}: {result}")
                return result

            except AttributeError as e:
                if "span_processor" in str(e):
                    exception_msg = (
                        f"AsyncThread-{worker_id}: ASYNC RACE CONDITION - {str(e)}"
                    )
                    exceptions.append(exception_msg)
                    raise AssertionError(exception_msg) from e
                raise
            except Exception as e:
                exception_msg = f"AsyncThread-{worker_id}: ASYNC ERROR - {str(e)}"
                exceptions.append(exception_msg)
                raise AssertionError(exception_msg) from e

        # Use asyncio with ThreadPoolExecutor to simulate real async behavior
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=6) as executor:
            # Create many concurrent async tasks
            tasks = [
                loop.run_in_executor(executor, threaded_init_flush, i)
                for i in range(12)
            ]

            try:
                await asyncio.gather(*tasks)
            except AssertionError:
                print(f"Async results: {results}")
                print(f"Async exceptions: {exceptions}")
                raise

        assert (
            len(results) == 12
        ), f"All async operations should succeed: {len(results)}/12"
        assert (
            len(exceptions) == 0
        ), f"No async race conditions should occur: {exceptions}"

    def test_stress_singleton_pattern(self):
        """Stress test the singleton pattern under extreme concurrent load."""
        results = []
        exceptions = []
        barrier = threading.Barrier(20)  # Synchronize thread starts

        def stress_singleton(thread_id):
            """Stress test singleton creation and usage."""
            try:
                # Wait for all threads to be ready
                barrier.wait()

                # All threads try to create/use singleton simultaneously
                exporter = InMemorySpanExporter()
                init_tracing(
                    project_api_key=f"stress-{thread_id}",
                    disable_batch=True,
                    exporter=exporter,
                    timeout_seconds=1,
                )

                # Immediate operations on the singleton
                initialized = is_tracing_initialized()
                flush_result = flush_tracing()

                results.append(
                    f"Stress-{thread_id}: init={initialized}, flush={flush_result}"
                )
                return flush_result

            except AttributeError as e:
                if "span_processor" in str(e):
                    exception_msg = f"Stress-{thread_id}: SINGLETON RACE - {str(e)}"
                    exceptions.append(exception_msg)
                    raise AssertionError(exception_msg) from e
                raise
            except Exception as e:
                exception_msg = f"Stress-{thread_id}: STRESS ERROR - {str(e)}"
                exceptions.append(exception_msg)
                raise AssertionError(exception_msg) from e

        # Create many threads that all start simultaneously
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(stress_singleton, i) for i in range(20)]

            for future in as_completed(futures):
                try:
                    future.result()
                except AssertionError:
                    print(f"Stress results: {results}")
                    print(f"Stress exceptions: {exceptions}")
                    raise

        assert (
            len(results) == 20
        ), f"All stress operations should succeed: {len(results)}/20"
        assert len(exceptions) == 0, f"No singleton race conditions: {exceptions}"

    def test_flush_before_init_returns_false(self):
        """Lifecycle calls on an uninitialized tracer must return False, never
        raise. This used to be reachable as a genuine race (the singleton was
        published mid-build, so a concurrent flush could hit a wrapper with no
        span processor); publishing only the finished wrapper makes that state
        unrepresentable, and this pins the remaining contract."""
        self._clear_tracer_instances()

        assert flush_tracing() is False
        assert force_reinit_processor() is False
        assert is_tracing_initialized() is False
        assert get_tracer_wrapper() is None

    def test_wrapper_is_not_published_during_init_instrumentations(self):
        """`init_instrumentations` runs while the init lock is held and can
        import modules that evaluate the `observe` decorator at import time, so
        the wrapper must still be invisible at that point. The Langfuse bridge
        depends on the same ordering."""
        self._clear_tracer_instances()

        observed = []
        original = tracing_mod.init_instrumentations

        def spy(*args, **kwargs):
            observed.append(
                (is_tracing_initialized(), get_tracer_wrapper() is None)
            )
            return original(*args, **kwargs)

        with patch.object(tracing_mod, "init_instrumentations", side_effect=spy):
            init_tracing(
                project_api_key="publish-order",
                disable_batch=True,
                exporter=InMemorySpanExporter(),
                timeout_seconds=1,
            )

        assert observed == [(False, True)]
        assert is_tracing_initialized() is True

    def test_initialization_atomicity(self):
        """Test that TracerWrapper initialization is atomic and thread-safe."""
        initialization_count = 0
        instances_created = []
        lock = threading.Lock()

        def try_initialize(thread_id):
            """Try to initialize and record what happens."""
            nonlocal initialization_count

            try:
                exporter = InMemorySpanExporter()
                init_tracing(
                    project_api_key=f"atomic-{thread_id}",
                    disable_batch=True,
                    exporter=exporter,
                    timeout_seconds=1,
                )

                with lock:
                    initialization_count += 1
                    instances_created.append(id(get_tracer_wrapper()))

                # Test immediate usage
                result = flush_tracing()
                return f"Thread-{thread_id}: {result}"

            except AttributeError as e:
                if "span_processor" in str(e):
                    raise AssertionError(
                        f"INITIALIZATION NOT ATOMIC - Thread {thread_id}: {str(e)}"
                    ) from e
                raise
            except Exception as e:
                raise AssertionError(
                    f"INITIALIZATION FAILED - Thread {thread_id}: {str(e)}"
                ) from e

        # Many threads trying to initialize simultaneously
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(try_initialize, i) for i in range(15)]

            results = []
            for future in as_completed(futures):
                results.append(future.result())

        # Verify singleton behavior
        unique_instances = set(instances_created)
        assert (
            len(unique_instances) == 1
        ), f"Should only create one instance, got {len(unique_instances)}: {unique_instances}"
        assert len(results) == 15, f"All threads should succeed: {len(results)}/15"

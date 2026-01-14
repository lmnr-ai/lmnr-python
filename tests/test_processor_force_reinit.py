"""
Unit tests for LaminarSpanProcessor.force_reinit() functionality.

These are low-level unit tests that directly test the processor's force_reinit behavior,
specifically for Lambda-like environments where BatchSpanProcessor needs proper shutdown.
"""

import time
import pytest
from unittest.mock import patch, MagicMock
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SpanExportResult

from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor


@pytest.fixture
def mock_otlp_exporter():
    """Create a mock OTLP exporter that tracks all operations."""
    exported_spans = []

    def create_mock_instance(*args, **kwargs):
        mock = MagicMock()
        mock.export = MagicMock(
            side_effect=lambda spans: (
                exported_spans.extend(spans),
                SpanExportResult.SUCCESS,
            )[1]
        )
        mock.shutdown = MagicMock()
        mock.force_flush = MagicMock(return_value=True)
        return mock

    with patch(
        "lmnr.opentelemetry_lib.tracing.exporter.OTLPSpanExporter",
        side_effect=create_mock_instance,
    ):
        yield exported_spans


def test_force_reinit_shuts_down_old_processor_before_reinit_exporter(
    mock_otlp_exporter,
):
    """
    Test that force_reinit() calls shutdown() on the old processor BEFORE
    reinitializing the exporter. This is the key fix for the Lambda issue.

    If the order is wrong (exporter reinit first, then processor shutdown),
    the new exporter gets shut down immediately, causing 'Exporter already shutdown' errors.
    """
    # Create processor with BatchSpanProcessor
    processor = LaminarSpanProcessor(
        base_url="http://test.local",
        api_key="test_key",
        disable_batch=False,
        force_http=False,
    )

    # Track the order of operations
    operation_order = []

    # Patch the exporter's _init_instance to track when it's called
    original_init = processor.exporter._init_instance

    def tracked_init():
        operation_order.append("exporter_reinit")
        return original_init()

    processor.exporter._init_instance = tracked_init

    # Patch the processor instance's shutdown to track when it's called
    original_shutdown = processor.instance.shutdown

    def tracked_shutdown():
        operation_order.append("processor_shutdown")
        return original_shutdown()

    processor.instance.shutdown = tracked_shutdown

    # Call force_reinit
    processor.force_reinit()

    # CRITICAL ASSERTION: processor_shutdown must happen BEFORE exporter_reinit
    assert operation_order == ["processor_shutdown", "exporter_reinit"], (
        f"Expected ['processor_shutdown', 'exporter_reinit'], got {operation_order}. "
        f"The old processor must be shut down BEFORE reinitializing the exporter, "
        f"otherwise the new exporter gets shut down immediately!"
    )

    processor.shutdown()


def test_force_reinit_creates_fresh_exporter_instance():
    """Test that force_reinit() creates a completely new exporter instance."""
    # Patch OTLP exporter to track instances
    with patch("lmnr.opentelemetry_lib.tracing.exporter.OTLPSpanExporter") as mock_otlp:
        mock_instance_1 = MagicMock()
        mock_instance_2 = MagicMock()
        mock_otlp.side_effect = [mock_instance_1, mock_instance_2]

        processor = LaminarSpanProcessor(
            base_url="http://test.local",
            api_key="test_key",
            disable_batch=False,
            force_http=False,
        )

        # Get the initial exporter instance
        initial_exporter_instance = processor.exporter.instance
        assert initial_exporter_instance is mock_instance_1

        # Call force_reinit
        processor.force_reinit()

        # Get the new exporter instance
        new_exporter_instance = processor.exporter.instance

        # Assert they are different objects
        assert (
            new_exporter_instance is mock_instance_2
        ), "force_reinit should create a new exporter instance"
        assert initial_exporter_instance is not new_exporter_instance

        # Verify the old instance was shut down
        mock_instance_1.shutdown.assert_called()

        processor.shutdown()


def test_force_reinit_with_batch_processor_exports_pending_spans(mock_otlp_exporter):
    """
    Test that force_reinit with BatchSpanProcessor properly exports all pending spans
    by shutting down the old processor (which joins the daemon thread).
    """
    # Create processor and tracer provider
    processor = LaminarSpanProcessor(
        base_url="http://test.local",
        api_key="test_key",
        disable_batch=False,  # BatchSpanProcessor
        force_http=False,
    )

    provider = TracerProvider(resource=Resource.create({}))
    provider.add_span_processor(processor)
    tracer = provider.get_tracer(__name__)

    # Create and end a span (will be buffered in BatchSpanProcessor)
    span = tracer.start_span("test_span_1")
    span.end()

    # At this point, the span is in the batch queue but not exported yet
    time.sleep(0.1)  # Give it a moment

    # Force reinit should flush and export the span
    processor.force_reinit()

    # Allow time for the export (shutdown waits for the daemon thread)
    time.sleep(0.2)

    # Verify the span was exported
    assert (
        len(mock_otlp_exporter) >= 1
    ), "force_reinit should export pending spans from the old BatchSpanProcessor"

    exported_names = [span.name for span in mock_otlp_exporter]
    assert (
        "test_span_1" in exported_names
    ), f"Expected 'test_span_1' in exported spans, got {exported_names}"

    processor.shutdown()


def test_force_reinit_multiple_times(mock_otlp_exporter):
    """Test that force_reinit can be called multiple times successfully."""
    processor = LaminarSpanProcessor(
        base_url="http://test.local",
        api_key="test_key",
        disable_batch=False,
        force_http=False,
    )

    provider = TracerProvider(resource=Resource.create({}))
    provider.add_span_processor(processor)
    tracer = provider.get_tracer(__name__)

    # Call force_reinit multiple times with spans in between
    for i in range(3):
        span = tracer.start_span(f"span_{i}")
        span.end()

        processor.force_reinit()
        time.sleep(0.2)

    # All spans should have been exported
    exported_names = [span.name for span in mock_otlp_exporter]
    assert "span_0" in exported_names
    assert "span_1" in exported_names
    assert "span_2" in exported_names

    processor.shutdown()


def test_force_reinit_preserves_disable_batch_setting():
    """Test that force_reinit preserves the disable_batch setting."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor

    # Patch OTLP exporter
    with patch("lmnr.opentelemetry_lib.tracing.exporter.OTLPSpanExporter") as mock_otlp:

        def create_mock(*args, **kwargs):
            mock = MagicMock()
            mock.shutdown = MagicMock()
            mock.force_flush = MagicMock(return_value=True)
            return mock

        mock_otlp.side_effect = create_mock

        # Test with disable_batch=True
        processor_simple = LaminarSpanProcessor(
            base_url="http://test.local",
            api_key="test_key",
            disable_batch=True,
            force_http=False,
        )

        assert isinstance(processor_simple.instance, SimpleSpanProcessor)
        processor_simple.force_reinit()
        assert isinstance(
            processor_simple.instance, SimpleSpanProcessor
        ), "force_reinit should preserve SimpleSpanProcessor when disable_batch=True"

        processor_simple.shutdown()

        # Test with disable_batch=False
        processor_batch = LaminarSpanProcessor(
            base_url="http://test.local",
            api_key="test_key",
            disable_batch=False,
            force_http=False,
        )

        assert isinstance(processor_batch.instance, BatchSpanProcessor)
        processor_batch.force_reinit()
        assert isinstance(
            processor_batch.instance, BatchSpanProcessor
        ), "force_reinit should preserve BatchSpanProcessor when disable_batch=False"

        processor_batch.shutdown()

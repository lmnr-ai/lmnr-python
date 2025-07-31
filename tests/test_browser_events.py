import pytest
import uuid
from unittest.mock import MagicMock, AsyncMock, patch

from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.browser.playwright_otel import PlaywrightInstrumentor


class TestBrowserEvents:
    @pytest.fixture
    def mock_sync_client(self):
        client = MagicMock(spec=LaminarClient)
        client._browser_events = MagicMock()
        client._browser_events.send = MagicMock()
        return client

    @pytest.fixture
    def mock_async_client(self):
        client = AsyncMock(spec=AsyncLaminarClient)
        client._browser_events = AsyncMock()
        client._browser_events.send = AsyncMock()
        return client

    @pytest.fixture
    def mock_sync_page(self):
        page = MagicMock(name="SyncPage")
        page.evaluate.return_value = True
        page.goto.return_value = None
        page.click.return_value = None
        page.is_closed.side_effect = [
            False,
            False,
            True,
        ]  # Return False twice, then True
        return page

    @pytest.fixture
    def mock_async_page(self):
        page = AsyncMock(name="AsyncPage")
        page.evaluate.return_value = True
        page.goto.return_value = None
        page.click.return_value = None
        page.is_closed.side_effect = [
            False,
            False,
            True,
        ]  # Return False twice, then True
        return page

    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_sync_browser_events(self, mock_sleep, mock_sync_client, mock_sync_page):
        """Test that browser events are captured and sent to Laminar in synchronous mode"""

        # Set up the evaluate method to return events
        def evaluate_side_effect(js_code):
            if "lmnrGetAndClearEvents" in js_code:
                # Return mock events on the second call
                return [{"data": [1, 2, 3, 4]}]
            return True

        mock_sync_page.evaluate.side_effect = evaluate_side_effect

        # Directly test the send_events_sync function
        from lmnr.sdk.browser.pw_utils import send_events_sync

        # Create test session_id and trace_id
        session_id = str(uuid.uuid4().hex)
        trace_id = format(12345, "032x")  # Simple trace ID for testing

        # Send events
        send_events_sync(mock_sync_page, session_id, trace_id, mock_sync_client)

        # Verify events were sent
        assert (
            mock_sync_client._browser_events.send.called
        ), "No events were sent to Laminar"

        # Verify call arguments
        args = mock_sync_client._browser_events.send.call_args[0]
        assert (
            len(args) == 3
        ), "send() should receive 3 arguments: session_id, trace_id, events"

        # Verify data format
        sent_session_id, sent_trace_id, events = args
        assert sent_session_id == session_id, "session_id mismatch"
        assert sent_trace_id == trace_id, "trace_id mismatch"
        assert isinstance(events, list), "events should be a list"
        assert len(events) > 0, "events list should not be empty"

    def test_instrumentor_structure(self, mock_sync_client):
        """Test the structure of the instrumentor and its wrapped methods"""
        # Import the wrapped methods data structures
        from lmnr.sdk.browser.playwright_otel import (
            WRAPPED_METHODS,
            WRAPPED_METHODS_ASYNC,
        )

        # Verify that the wrapped methods have the expected structure
        expected_methods = [
            "launch",
            "new_context",
            "connect",
            "connect_over_cdp",
            "launch_persistent_context",
            "bring_to_front",
        ]

        expected_objects = ["BrowserContext", "Browser", "BrowserType", "Page"]

        # Check sync methods
        sync_methods = set()
        for method in WRAPPED_METHODS:
            assert (
                method["package"] == "playwright.sync_api"
            ), f"Unexpected package for {method['method']}"
            assert (
                method["object"] in expected_objects
            ), f"Unexpected object for {method['method']}"
            assert (
                method["method"] in expected_methods
            ), f"Unexpected method: {method['method']}"
            assert callable(
                method["wrapper"]
            ), f"Wrapper for {method['method']} is not callable"
            sync_methods.add(method["method"])

        # Check async methods
        async_methods = set()
        for method in WRAPPED_METHODS_ASYNC:
            assert (
                method["package"] == "playwright.async_api"
            ), f"Unexpected package for {method['method']}"
            assert (
                method["object"] in expected_objects
            ), f"Unexpected object for {method['method']}"
            assert (
                method["method"] in expected_methods
            ), f"Unexpected method: {method['method']}"
            assert callable(
                method["wrapper"]
            ), f"Wrapper for {method['method']} is not callable"
            async_methods.add(method["method"])

        # Verify all expected methods are covered
        for method in expected_methods:
            assert method in sync_methods, f"Missing sync method: {method}"
            assert method in async_methods, f"Missing async method: {method}"

        # Verify the instrumentor is initialized correctly
        instrumentor = PlaywrightInstrumentor(
            client=mock_sync_client, async_client=AsyncMock()
        )

        # Check client references
        assert instrumentor.client == mock_sync_client, "Client reference mismatch"

        # Verify instrumentation dependencies
        assert instrumentor.instrumentation_dependencies() == (
            "playwright >= 1.9.0",
        ), "Unexpected instrumentation dependencies"

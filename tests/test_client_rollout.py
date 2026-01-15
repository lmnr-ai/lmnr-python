"""
Tests for rollout resources on LaminarClient and AsyncLaminarClient.
"""

import uuid
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock, Mock

from lmnr import LaminarClient, AsyncLaminarClient
from lmnr.sdk.types import RolloutParam


class TestAsyncRolloutResource:
    """Test rollout resource on AsyncLaminarClient."""

    @pytest_asyncio.fixture
    async def async_client(self):
        """Create an AsyncLaminarClient for testing."""
        client = AsyncLaminarClient(
            base_url="http://test-api.com", project_api_key="test-key"
        )
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_connect_returns_streaming_response(self, async_client):
        """Test that connect() returns a streaming response."""
        session_id = uuid.uuid4()
        params = [{"name": "arg1", "required": True}]

        mock_stream = Mock()

        with patch.object(async_client.rollout._client, "stream") as mock_stream_method:
            mock_stream_method.return_value = mock_stream

            result = async_client.rollout.connect(
                session_id=session_id,
                function_name="test_function",
                params=params,
            )

            assert result == mock_stream

            # Verify correct API call
            mock_stream_method.assert_called_once()
            call_args = mock_stream_method.call_args
            assert call_args[0][0] == "POST"
            assert f"/v1/rollouts/{session_id}" in call_args[0][1]
            assert call_args[1]["headers"]["Accept"] == "text/event-stream"
            assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"
            assert call_args[1]["json"]["name"] == "test_function"
            assert call_args[1]["json"]["params"] == params

    @pytest.mark.asyncio
    async def test_connect_with_no_params(self, async_client):
        """Test connect() with no parameters."""
        session_id = uuid.uuid4()

        mock_stream = Mock()

        with patch.object(async_client.rollout._client, "stream") as mock_stream_method:
            mock_stream_method.return_value = mock_stream

            result = async_client.rollout.connect(
                session_id=session_id,
                function_name="test_function",
            )

            call_args = mock_stream_method.call_args
            assert call_args[1]["json"]["params"] == []

    @pytest.mark.asyncio
    async def test_update_status_success(self, async_client):
        """Test updating rollout session status."""
        session_id = "test-session-id"

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(
            async_client.rollout._client, "patch", new_callable=AsyncMock
        ) as mock_patch:
            mock_patch.return_value = mock_response

            await async_client.rollout.update_status(session_id, "RUNNING")

            mock_patch.assert_called_once()
            call_args = mock_patch.call_args
            assert f"/v1/rollouts/{session_id}/status" in call_args[0][0]
            assert call_args[1]["json"]["status"] == "RUNNING"

    @pytest.mark.asyncio
    async def test_update_status_all_states(self, async_client):
        """Test all valid status states."""
        session_id = "test-session"

        mock_response = Mock()
        mock_response.status_code = 200

        statuses = ["PENDING", "RUNNING", "FINISHED", "STOPPED"]

        with patch.object(
            async_client.rollout._client, "patch", new_callable=AsyncMock
        ) as mock_patch:
            mock_patch.return_value = mock_response

            for status in statuses:
                await async_client.rollout.update_status(session_id, status)

                call_args = mock_patch.call_args
                assert call_args[1]["json"]["status"] == status

    @pytest.mark.asyncio
    async def test_delete_session_success(self, async_client):
        """Test deleting a rollout session."""
        session_id = "test-session-id"

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(
            async_client.rollout._client, "delete", new_callable=AsyncMock
        ) as mock_delete:
            mock_delete.return_value = mock_response

            await async_client.rollout.delete_session(session_id)

            mock_delete.assert_called_once()
            call_args = mock_delete.call_args
            assert f"/v1/rollouts/{session_id}" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_update_span_info_success(self, async_client):
        """Test updating span info during rollout."""
        from datetime import datetime

        session_id = "test-session"
        span_data = {
            "name": "test_span",
            "span_id": uuid.uuid4(),
            "parent_span_id": uuid.uuid4(),
            "trace_id": uuid.uuid4(),
            "start_time": datetime.now(),
            "attributes": {"key": "value"},
            "span_type": "LLM",
        }

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(
            async_client.rollout._client, "patch", new_callable=AsyncMock
        ) as mock_patch:
            mock_patch.return_value = mock_response

            await async_client.rollout.update_span_info(session_id, span_data)

            # Verify payload transformation
            call_args = mock_patch.call_args
            assert (
                call_args[0][0]
                == f"http://test-api.com:443/v1/rollouts/{session_id}/update"
            )
            payload = call_args[1]["json"]

            assert payload["type"] == "spanStart"
            assert payload["name"] == "test_span"
            assert payload["spanId"] == str(span_data["span_id"])
            assert payload["parentSpanId"] == str(span_data["parent_span_id"])
            assert payload["traceId"] == str(span_data["trace_id"])
            assert payload["spanType"] == "LLM"
            assert payload["attributes"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_update_span_info_with_no_parent(self, async_client):
        """Test update_span_info when parent_span_id is None."""
        from datetime import datetime

        session_id = "test-session"
        span_data = {
            "name": "root_span",
            "span_id": uuid.uuid4(),
            "parent_span_id": None,
            "trace_id": uuid.uuid4(),
            "start_time": datetime.now(),
            "attributes": {},
            "span_type": "DEFAULT",
        }

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(
            async_client.rollout._client, "patch", new_callable=AsyncMock
        ) as mock_patch:
            mock_patch.return_value = mock_response

            await async_client.rollout.update_span_info(session_id, span_data)

            payload = mock_patch.call_args[1]["json"]
            assert payload["parentSpanId"] is None

    @pytest.mark.asyncio
    async def test_update_span_info_logs_on_failure(self, async_client):
        """Test that update_span_info logs but doesn't raise on failure."""
        from datetime import datetime

        session_id = "test-session"
        span_data = {
            "name": "test",
            "span_id": uuid.uuid4(),
            "parent_span_id": None,
            "trace_id": uuid.uuid4(),
            "start_time": datetime.now(),
            "attributes": {},
            "span_type": "DEFAULT",
        }

        with patch.object(
            async_client.rollout._client, "patch", new_callable=AsyncMock
        ) as mock_patch:
            mock_patch.side_effect = Exception("Network error")

            # Should not raise - just log
            await async_client.rollout.update_span_info(session_id, span_data)


class TestSyncRolloutResource:
    """Test rollout resource on synchronous LaminarClient."""

    @pytest.fixture
    def sync_client(self):
        """Create a LaminarClient for testing."""
        client = LaminarClient(
            base_url="http://test-api.com", project_api_key="test-key"
        )
        yield client
        client.close()

    def test_connect_returns_streaming_response(self, sync_client):
        """Test that connect() returns a streaming context manager."""
        session_id = uuid.uuid4()
        params = [{"name": "input", "type": "str"}]

        mock_stream = Mock()

        with patch.object(sync_client.rollout._client, "stream") as mock_stream_method:
            mock_stream_method.return_value = mock_stream

            result = sync_client.rollout.connect(
                session_id=session_id,
                function_name="my_function",
                params=params,
            )

            assert result == mock_stream

            # Verify headers include text/event-stream
            call_args = mock_stream_method.call_args
            assert call_args[1]["headers"]["Accept"] == "text/event-stream"
            assert call_args[1]["json"]["name"] == "my_function"
            assert call_args[1]["json"]["params"] == params

    def test_update_status(self, sync_client):
        """Test updating status."""
        session_id = "test-session"

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(sync_client.rollout._client, "patch") as mock_patch:
            mock_patch.return_value = mock_response

            sync_client.rollout.update_status(session_id, "PENDING")

            call_args = mock_patch.call_args
            assert call_args[1]["json"]["status"] == "PENDING"

    def test_delete_session(self, sync_client):
        """Test deleting a session."""
        session_id = "test-session"

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(sync_client.rollout._client, "delete") as mock_delete:
            mock_delete.return_value = mock_response

            sync_client.rollout.delete_session(session_id)

            mock_delete.assert_called_once()

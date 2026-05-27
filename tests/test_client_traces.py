"""Tests for the traces resource on LaminarClient and AsyncLaminarClient."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from lmnr import AsyncLaminarClient, LaminarClient


def _ok_response():
    response = MagicMock()
    response.status_code = 200
    return response


def _not_found_response():
    response = MagicMock()
    response.status_code = 404
    response.text = "Trace not found"
    return response


def _error_response():
    response = MagicMock()
    response.status_code = 500
    response.text = "Internal server error"
    return response


class TestSyncLaminarClientTraces:
    """Test traces methods on LaminarClient."""

    @pytest.fixture
    def sync_client(self):
        client = LaminarClient(base_url="http://test-api.com", project_api_key="test-key")
        yield client
        client.close()

    def test_push_metadata_success(self, sync_client):
        trace_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        with patch.object(sync_client._LaminarClient__client, "post") as mock_post:
            mock_post.return_value = _ok_response()
            sync_client.traces.push_metadata(trace_id, {"score": 0.85, "reviewer": "alice"})

        mock_post.assert_called_once()
        url, *_ = mock_post.call_args.args
        assert url == "http://test-api.com:443/v1/traces/metadata"
        assert mock_post.call_args.kwargs["json"] == {
            "traceId": str(trace_id),
            "metadata": {"score": 0.85, "reviewer": "alice"},
        }

    def test_push_metadata_accepts_int_trace_id(self, sync_client):
        trace_id_int = 0x12345678123456789ABC123456789ABC
        with patch.object(sync_client._LaminarClient__client, "post") as mock_post:
            mock_post.return_value = _ok_response()
            sync_client.traces.push_metadata(trace_id_int, {"k": "v"})
        sent_id = mock_post.call_args.kwargs["json"]["traceId"]
        assert uuid.UUID(sent_id).int == trace_id_int

    def test_push_metadata_404_does_not_raise(self, sync_client):
        with patch.object(sync_client._LaminarClient__client, "post") as mock_post:
            mock_post.return_value = _not_found_response()
            sync_client.traces.push_metadata(uuid.uuid4(), {"k": "v"})
        mock_post.assert_called_once()

    def test_push_metadata_500_raises(self, sync_client):
        with patch.object(sync_client._LaminarClient__client, "post") as mock_post:
            mock_post.return_value = _error_response()
            with pytest.raises(ValueError, match="Failed to push trace metadata"):
                sync_client.traces.push_metadata(uuid.uuid4(), {"k": "v"})

    def test_push_metadata_empty_metadata_raises(self, sync_client):
        with pytest.raises(ValueError, match="non-empty"):
            sync_client.traces.push_metadata(uuid.uuid4(), {})


class TestAsyncLaminarClientTraces:
    """Test traces methods on AsyncLaminarClient."""

    @pytest_asyncio.fixture
    async def async_client(self):
        client = AsyncLaminarClient(base_url="http://test-api.com", project_api_key="test-key")
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_push_metadata_success(self, async_client):
        trace_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        with patch.object(
            async_client._AsyncLaminarClient__client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _ok_response()
            await async_client.traces.push_metadata(
                trace_id, {"score": 0.85, "reviewer": "alice"}
            )

        mock_post.assert_called_once()
        url, *_ = mock_post.call_args.args
        assert url == "http://test-api.com:443/v1/traces/metadata"
        assert mock_post.call_args.kwargs["json"] == {
            "traceId": str(trace_id),
            "metadata": {"score": 0.85, "reviewer": "alice"},
        }

    @pytest.mark.asyncio
    async def test_push_metadata_404_does_not_raise(self, async_client):
        with patch.object(
            async_client._AsyncLaminarClient__client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _not_found_response()
            await async_client.traces.push_metadata(uuid.uuid4(), {"k": "v"})
        mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_metadata_500_raises(self, async_client):
        with patch.object(
            async_client._AsyncLaminarClient__client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = _error_response()
            with pytest.raises(ValueError, match="Failed to push trace metadata"):
                await async_client.traces.push_metadata(uuid.uuid4(), {"k": "v"})

    @pytest.mark.asyncio
    async def test_push_metadata_empty_metadata_raises(self, async_client):
        with pytest.raises(ValueError, match="non-empty"):
            await async_client.traces.push_metadata(uuid.uuid4(), {})

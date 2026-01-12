"""
Tests for SQL resources on LaminarClient and AsyncLaminarClient.
"""

import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock, Mock

from lmnr import LaminarClient, AsyncLaminarClient


class TestAsyncSqlResource:
    """Test SQL resource on AsyncLaminarClient."""

    @pytest_asyncio.fixture
    async def async_client(self):
        """Create an AsyncLaminarClient for testing."""
        client = AsyncLaminarClient(
            base_url="http://test-api.com",
            project_api_key="test-key"
        )
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_query_success(self, async_client):
        """Test successful SQL query."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "1", "name": "test1"},
                {"id": "2", "name": "test2"},
            ]
        }
        
        with patch.object(
            async_client.sql._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response
            
            result = await async_client.sql.query(
                "SELECT * FROM spans WHERE id = {id:String}",
                {"id": "test-id"}
            )
            
            assert len(result) == 2
            assert result[0]["id"] == "1"
            assert result[1]["name"] == "test2"
            
            # Verify correct API call
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://test-api.com:443/v1/sql/query"
            assert call_args[1]["json"] == {
                "query": "SELECT * FROM spans WHERE id = {id:String}",
                "parameters": {"id": "test-id"},
            }

    @pytest.mark.asyncio
    async def test_query_with_no_parameters(self, async_client):
        """Test SQL query without parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        
        with patch.object(
            async_client.sql._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response
            
            result = await async_client.sql.query("SELECT * FROM spans")
            
            assert result == []
            call_args = mock_post.call_args
            assert call_args[1]["json"]["parameters"] == {}

    @pytest.mark.asyncio
    async def test_query_empty_data_field(self, async_client):
        """Test when response has no data field."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        
        with patch.object(
            async_client.sql._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response
            
            result = await async_client.sql.query("SELECT * FROM spans")
            
            assert result == []

    @pytest.mark.asyncio
    async def test_query_http_error(self, async_client):
        """Test HTTP error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server error")
        
        with patch.object(
            async_client.sql._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response
            
            with pytest.raises(Exception, match="Server error"):
                await async_client.sql.query("SELECT * FROM spans")


class TestSyncSqlResource:
    """Test SQL resource on synchronous LaminarClient."""

    @pytest.fixture
    def sync_client(self):
        """Create a LaminarClient for testing."""
        client = LaminarClient(
            base_url="http://test-api.com",
            project_api_key="test-key"
        )
        yield client
        client.close()

    def test_query_success(self, sync_client):
        """Test successful SQL query."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"trace_id": "abc123", "span_id": "def456"},
            ]
        }
        
        with patch.object(sync_client.sql._client, "post") as mock_post:
            mock_post.return_value = mock_response
            
            result = sync_client.sql.query(
                "SELECT trace_id, span_id FROM spans",
                {"limit": 10}
            )
            
            assert len(result) == 1
            assert result[0]["trace_id"] == "abc123"
            
            # Verify API call
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["json"] == {
                "query": "SELECT trace_id, span_id FROM spans",
                "parameters": {"limit": 10},
            }

    def test_query_with_none_parameters(self, sync_client):
        """Test query when parameters is None."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        
        with patch.object(sync_client.sql._client, "post") as mock_post:
            mock_post.return_value = mock_response
            
            result = sync_client.sql.query("SELECT * FROM spans", None)
            
            assert result == []
            call_args = mock_post.call_args
            assert call_args[1]["json"]["parameters"] == {}

    def test_query_complex_parameters(self, sync_client):
        """Test query with complex parameter types."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        
        with patch.object(sync_client.sql._client, "post") as mock_post:
            mock_post.return_value = mock_response
            
            params = {
                "traceId": "uuid-value",
                "paths": ["root.a", "root.b"],
                "limit": 100,
            }
            
            result = sync_client.sql.query(
                "SELECT * FROM spans WHERE trace_id = {traceId:UUID}",
                params
            )
            
            assert result == []
            call_args = mock_post.call_args
            assert call_args[1]["json"]["parameters"] == params

    def test_query_raises_on_http_error(self, sync_client):
        """Test that HTTP errors are raised."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("Not found")
        
        with patch.object(sync_client.sql._client, "post") as mock_post:
            mock_post.return_value = mock_response
            
            with pytest.raises(Exception, match="Not found"):
                sync_client.sql.query("SELECT * FROM nonexistent")

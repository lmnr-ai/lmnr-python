"""
Tests for the evaluators resource on LaminarClient and AsyncLaminarClient.
"""

import pytest
import pytest_asyncio
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

from lmnr import LaminarClient, AsyncLaminarClient


class TestAsyncLaminarClientEvaluators:
    """Test evaluators methods on AsyncLaminarClient."""

    @pytest_asyncio.fixture
    async def async_client(self):
        """Create an AsyncLaminarClient for testing."""
        client = AsyncLaminarClient(
            base_url="http://test-api.com",
            project_api_key="test-key"
        )
        yield client
        await client.close()

    @pytest.fixture
    def mock_success_response(self):
        """Mock successful HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        return mock_response

    @pytest.fixture
    def mock_unauthorized_response(self):
        """Mock unauthorized HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        return mock_response

    @pytest.fixture
    def mock_error_response(self):
        """Mock error HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        return mock_response

    @pytest.mark.asyncio
    async def test_score_with_trace_id_success(self, async_client, mock_success_response):
        """Test successful score creation with trace_id."""
        trace_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        
        with patch.object(async_client._AsyncLaminarClient__client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_success_response
            
            await async_client.evaluators.score(
                name="quality",
                trace_id=trace_id,
                score=0.95,
                metadata={"model": "gpt-4"}
            )
            
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # Verify URL
            assert call_args[0][0] == "http://test-api.com:443/v1/evaluators/score"
            
            # Verify payload
            expected_payload = {
                "name": "quality",
                "traceId": str(trace_id),
                "metadata": {"model": "gpt-4"},
                "score": 0.95,
                "source": "SDK",
            }
            assert call_args[1]["json"] == expected_payload

    @pytest.mark.asyncio
    async def test_score_with_span_id_success(self, async_client, mock_success_response):
        """Test successful score creation with span_id."""
        span_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        
        with patch.object(async_client._AsyncLaminarClient__client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_success_response
            
            await async_client.evaluators.score(
                name="relevance",
                span_id=span_id,
                score=0.87
            )
            
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # Verify payload
            expected_payload = {
                "name": "relevance",
                "spanId": str(span_id),
                "metadata": None,
                "score": 0.87,
                "source": "SDK",
            }
            assert call_args[1]["json"] == expected_payload

    @pytest.mark.asyncio
    async def test_score_neither_trace_nor_span_id_raises_error(self, async_client):
        """Test that providing neither trace_id nor span_id raises ValueError."""
        with pytest.raises(ValueError, match="Either 'trace_id' or 'span_id' must be provided"):
            await async_client.evaluators.score(
                name="quality",
                score=0.95
            )

    @pytest.mark.asyncio
    async def test_score_both_trace_and_span_id_raises_error(self, async_client):
        """Test that providing both trace_id and span_id raises ValueError."""
        trace_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        span_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        
        with pytest.raises(ValueError, match="Cannot provide both trace_id and span_id"):
            await async_client.evaluators.score(
                name="quality",
                trace_id=trace_id,
                span_id=span_id,
                score=0.95
            )

class TestLaminarClientEvaluators:
    """Test evaluators methods on synchronous LaminarClient."""

    @pytest.fixture
    def sync_client(self):
        """Create a LaminarClient for testing."""
        client = LaminarClient(
            base_url="http://test-api.com",
            project_api_key="test-key"
        )
        yield client
        client.close()

    @pytest.fixture
    def mock_success_response(self):
        """Mock successful HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        return mock_response

    @pytest.fixture
    def mock_unauthorized_response(self):
        """Mock unauthorized HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        return mock_response

    @pytest.fixture
    def mock_error_response(self):
        """Mock error HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        return mock_response

    def test_score_with_trace_id_success(self, sync_client, mock_success_response):
        """Test successful score creation with trace_id."""
        trace_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        
        with patch.object(sync_client._LaminarClient__client, 'post') as mock_post:
            mock_post.return_value = mock_success_response
            
            sync_client.evaluators.score(
                name="quality",
                trace_id=trace_id,
                score=0.95,
                metadata={"model": "gpt-4"}
            )
            
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # Verify URL
            assert call_args[0][0] == "http://test-api.com:443/v1/evaluators/score"
            
            # Verify payload
            expected_payload = {
                "name": "quality",
                "traceId": str(trace_id),
                "metadata": {"model": "gpt-4"},
                "score": 0.95,
                "source": "SDK",
            }
            assert call_args[1]["json"] == expected_payload

    def test_score_with_span_id_success(self, sync_client, mock_success_response):
        """Test successful score creation with span_id."""
        span_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        
        with patch.object(sync_client._LaminarClient__client, 'post') as mock_post:
            mock_post.return_value = mock_success_response
            
            sync_client.evaluators.score(
                name="relevance",
                span_id=span_id,
                score=0.87
            )
            
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # Verify payload
            expected_payload = {
                "name": "relevance",
                "spanId": str(span_id),
                "metadata": None,
                "score": 0.87,
                "source": "SDK",
            }
            assert call_args[1]["json"] == expected_payload

    def test_score_neither_trace_nor_span_id_raises_error(self, sync_client):
        """Test that providing neither trace_id nor span_id raises ValueError."""
        with pytest.raises(ValueError, match="Either 'trace_id' or 'span_id' must be provided"):
            sync_client.evaluators.score(
                name="quality",
                score=0.95
            )

    def test_score_both_trace_and_span_id_raises_error(self, sync_client):
        """Test that providing both trace_id and span_id raises ValueError."""
        trace_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        span_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        
        with pytest.raises(ValueError, match="Cannot provide both trace_id and span_id"):
            sync_client.evaluators.score(
                name="quality",
                trace_id=trace_id,
                span_id=span_id,
                score=0.95
            )
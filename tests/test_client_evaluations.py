"""
Tests for the new evaluation methods on LaminarClient and AsyncLaminarClient.
"""

import pytest
import pytest_asyncio
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

from lmnr import LaminarClient, AsyncLaminarClient
from lmnr.sdk.types import InitEvaluationResponse


class TestAsyncLaminarClientEvaluations:
    """Test evaluation methods on AsyncLaminarClient."""

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
    def mock_eval_response(self):
        """Mock evaluation response."""
        mock_response = MagicMock()
        mock_response.id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        mock_response.projectId = "test-project-id"
        return mock_response

    @pytest.mark.asyncio
    async def test_create_evaluation_success(self, async_client, mock_eval_response):
        """Test successful evaluation creation."""
        with patch.object(async_client.evals, 'create_evaluation', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_eval_response.id
            
            eval_id = await async_client.evals.create_evaluation(
                name="Test Evaluation",
                group_name="test_group"
            )
            
            assert eval_id == mock_eval_response.id
            mock_create.assert_called_once_with(name="Test Evaluation", group_name="test_group")

    @pytest.mark.asyncio
    async def test_create_evaluation_with_defaults(self, async_client, mock_eval_response):
        """Test evaluation creation with default parameters."""
        with patch.object(async_client.evals, 'create_evaluation', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_eval_response.id
            
            eval_id = await async_client.evals.create_evaluation()
            
            assert eval_id == mock_eval_response.id
            mock_create.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_create_datapoint_success(self, async_client):
        """Test successful datapoint creation."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        test_data = {"input": "test input"}
        test_target = {"expected": "test output"}
        test_metadata = {"custom": "value"}
        custom_trace_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        expected_datapoint_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
        
        with patch.object(async_client.evals, 'create_datapoint', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = expected_datapoint_id
            
            datapoint_id = await async_client.evals.create_datapoint(
                eval_id=eval_id,
                data=test_data,
                target=test_target,
                metadata=test_metadata,
                index=1,
                trace_id=custom_trace_id
            )
            
            # Verify datapoint_id is correct
            assert datapoint_id == expected_datapoint_id
            
            # Verify create_datapoint was called correctly
            mock_create.assert_called_once_with(
                eval_id=eval_id,
                data=test_data,
                target=test_target,
                metadata=test_metadata,
                index=1,
                trace_id=custom_trace_id
            )

    @pytest.mark.asyncio
    async def test_create_datapoint_with_defaults(self, async_client):
        """Test datapoint creation with minimal parameters."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        test_data = {"input": "test input"}
        expected_datapoint_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
        
        with patch.object(async_client.evals, 'create_datapoint', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = expected_datapoint_id
            
            datapoint_id = await async_client.evals.create_datapoint(
                eval_id=eval_id,
                data=test_data
            )
            
            # Verify datapoint_id is correct
            assert datapoint_id == expected_datapoint_id
            
            # Verify create_datapoint was called correctly
            mock_create.assert_called_once_with(
                eval_id=eval_id,
                data=test_data
            )

    @pytest.mark.asyncio
    async def test_update_datapoint_success(self, async_client):
        """Test successful datapoint update."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        datapoint_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        executor_output = {"result": "processed output"}
        scores = {"accuracy": 0.95, "relevance": 0.88}
        
        with patch.object(async_client.evals, 'update_datapoint', new_callable=AsyncMock) as mock_update:
            await async_client.evals.update_datapoint(
                eval_id=eval_id,
                datapoint_id=datapoint_id,
                scores=scores,
                executor_output=executor_output
            )
            
            mock_update.assert_called_once_with(
                eval_id=eval_id,
                datapoint_id=datapoint_id,
                scores=scores,
                executor_output=executor_output
            )

    @pytest.mark.asyncio
    async def test_update_datapoint_with_minimal_params(self, async_client):
        """Test datapoint update with minimal parameters."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        datapoint_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        scores = {"accuracy": 0.95}
        
        with patch.object(async_client.evals, 'update_datapoint', new_callable=AsyncMock) as mock_update:
            await async_client.evals.update_datapoint(
                eval_id=eval_id,
                datapoint_id=datapoint_id,
                scores=scores
            )
            
            mock_update.assert_called_once_with(
                eval_id=eval_id,
                datapoint_id=datapoint_id,
                scores=scores
            )


class TestLaminarClientEvaluations:
    """Test evaluation methods on synchronous LaminarClient."""

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
    def mock_eval_response(self):
        """Mock evaluation response."""
        mock_response = MagicMock()
        mock_response.id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        mock_response.projectId = "test-project-id"
        return mock_response

    def test_create_evaluation_success(self, sync_client, mock_eval_response):
        """Test successful evaluation creation."""
        with patch.object(sync_client.evals, 'create_evaluation') as mock_create:
            mock_create.return_value = mock_eval_response.id
            
            eval_id = sync_client.evals.create_evaluation(
                name="Test Evaluation",
                group_name="test_group"
            )
            
            assert eval_id == mock_eval_response.id
            mock_create.assert_called_once_with(name="Test Evaluation", group_name="test_group")

    def test_create_evaluation_with_defaults(self, sync_client, mock_eval_response):
        """Test evaluation creation with default parameters."""
        with patch.object(sync_client.evals, 'create_evaluation') as mock_create:
            mock_create.return_value = mock_eval_response.id
            
            eval_id = sync_client.evals.create_evaluation()
            
            assert eval_id == mock_eval_response.id
            mock_create.assert_called_once_with()

    def test_create_datapoint_success(self, sync_client):
        """Test successful datapoint creation."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        test_data = {"input": "test input"}
        test_target = {"expected": "test output"}
        test_metadata = {"custom": "value"}
        custom_trace_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        expected_datapoint_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
        
        with patch.object(sync_client.evals, 'create_datapoint') as mock_create:
            mock_create.return_value = expected_datapoint_id
            
            datapoint_id = sync_client.evals.create_datapoint(
                eval_id=eval_id,
                data=test_data,
                target=test_target,
                metadata=test_metadata,
                index=1,
                trace_id=custom_trace_id
            )
            
            # Verify datapoint_id is correct
            assert datapoint_id == expected_datapoint_id
            
            # Verify create_datapoint was called correctly
            mock_create.assert_called_once_with(
                eval_id=eval_id,
                data=test_data,
                target=test_target,
                metadata=test_metadata,
                index=1,
                trace_id=custom_trace_id
            )

    def test_create_datapoint_with_defaults(self, sync_client):
        """Test datapoint creation with minimal parameters."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        test_data = {"input": "test input"}
        expected_datapoint_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
        
        with patch.object(sync_client.evals, 'create_datapoint') as mock_create:
            mock_create.return_value = expected_datapoint_id
            
            datapoint_id = sync_client.evals.create_datapoint(
                eval_id=eval_id,
                data=test_data
            )
            
            # Verify datapoint_id is correct
            assert datapoint_id == expected_datapoint_id
            
            # Verify create_datapoint was called correctly
            mock_create.assert_called_once_with(
                eval_id=eval_id,
                data=test_data
            )

    def test_update_datapoint_success(self, sync_client):
        """Test successful datapoint update."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        datapoint_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        executor_output = {"result": "processed output"}
        scores = {"accuracy": 0.95, "relevance": 0.88}
        
        with patch.object(sync_client.evals, 'update_datapoint') as mock_update:
            sync_client.evals.update_datapoint(
                eval_id=eval_id,
                datapoint_id=datapoint_id,
                scores=scores,
                executor_output=executor_output
            )
            
            mock_update.assert_called_once_with(
                eval_id=eval_id,
                datapoint_id=datapoint_id,
                scores=scores,
                executor_output=executor_output
            )

    def test_update_datapoint_with_minimal_params(self, sync_client):
        """Test datapoint update with minimal parameters."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        datapoint_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        scores = {"accuracy": 0.95}
        
        with patch.object(sync_client.evals, 'update_datapoint') as mock_update:
            sync_client.evals.update_datapoint(
                eval_id=eval_id,
                datapoint_id=datapoint_id,
                scores=scores
            )
            
            mock_update.assert_called_once_with(
                eval_id=eval_id,
                datapoint_id=datapoint_id,
                scores=scores
            )


class TestClientEvaluationIntegration:
    """Integration tests for the full evaluation workflow."""

    @pytest.mark.asyncio
    async def test_async_client_full_workflow(self):
        """Test the complete evaluation workflow with AsyncLaminarClient."""
        client = AsyncLaminarClient(
            base_url="http://test-api.com",
            project_api_key="test-key"
        )
        
        mock_eval_response = MagicMock()
        mock_eval_response.id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        mock_eval_response.projectId = "test-project-id"
        
        try:
            with patch.object(client.evals, 'create_evaluation', new_callable=AsyncMock) as mock_create_eval, \
                 patch.object(client.evals, 'create_datapoint', new_callable=AsyncMock) as mock_create_dp, \
                 patch.object(client.evals, 'update_datapoint', new_callable=AsyncMock) as mock_update:
                
                mock_create_eval.return_value = mock_eval_response.id
                mock_create_dp.return_value = uuid.UUID("22222222-2222-2222-2222-222222222222")
                
                # Create evaluation
                eval_id = await client.evals.create_evaluation(name="Integration Test")
                
                # Create datapoint
                datapoint_id = await client.evals.create_datapoint(
                    eval_id=eval_id,
                    data={"query": "What is AI?"},
                    target={"answer": "Artificial Intelligence"}
                )
                
                # Update datapoint
                await client.evals.update_datapoint(
                    eval_id=eval_id,
                    datapoint_id=datapoint_id,
                    scores={"accuracy": 0.9},
                    executor_output={"response": "AI is artificial intelligence"}
                )
                
                # Verify all calls were made
                mock_create_eval.assert_called_once()
                mock_create_dp.assert_called_once()
                mock_update.assert_called_once()
                
        finally:
            await client.close()

    def test_sync_client_full_workflow(self):
        """Test the complete evaluation workflow with LaminarClient."""
        client = LaminarClient(
            base_url="http://test-api.com",
            project_api_key="test-key"
        )
        
        mock_eval_response = MagicMock()
        mock_eval_response.id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        mock_eval_response.projectId = "test-project-id"
        
        try:
            with patch.object(client.evals, 'create_evaluation') as mock_create_eval, \
                 patch.object(client.evals, 'create_datapoint') as mock_create_dp, \
                 patch.object(client.evals, 'update_datapoint') as mock_update:
                
                mock_create_eval.return_value = mock_eval_response.id
                mock_create_dp.return_value = uuid.UUID("22222222-2222-2222-2222-222222222222")
                
                # Create evaluation
                eval_id = client.evals.create_evaluation(name="Integration Test")
                
                # Create datapoint
                datapoint_id = client.evals.create_datapoint(
                    eval_id=eval_id,
                    data={"query": "What is AI?"},
                    target={"answer": "Artificial Intelligence"}
                )
                
                # Update datapoint
                client.evals.update_datapoint(
                    eval_id=eval_id,
                    datapoint_id=datapoint_id,
                    scores={"accuracy": 0.9},
                    executor_output={"response": "AI is artificial intelligence"}
                )
                
                # Verify all calls were made
                mock_create_eval.assert_called_once()
                mock_create_dp.assert_called_once()
                mock_update.assert_called_once()
                
        finally:
            client.close()


class TestClientEvaluationErrorHandling:
    """Test error handling in client evaluation methods."""

    @pytest.mark.asyncio
    async def test_async_client_create_evaluation_error(self):
        """Test error handling in async create_evaluation."""
        client = AsyncLaminarClient(
            base_url="http://test-api.com",
            project_api_key="test-key"
        )
        
        try:
            with patch.object(client.evals, 'create_evaluation', new_callable=AsyncMock) as mock_create:
                mock_create.side_effect = ValueError("API Error")
                
                with pytest.raises(ValueError, match="API Error"):
                    await client.evals.create_evaluation()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_async_client_update_datapoint_error(self):
        """Test error handling in async update_datapoint."""
        client = AsyncLaminarClient(
            base_url="http://test-api.com",
            project_api_key="test-key"
        )
        
        try:
            with patch.object(client.evals, 'update_datapoint', new_callable=AsyncMock) as mock_update:
                mock_update.side_effect = ValueError("Update failed")
                
                with pytest.raises(ValueError, match="Update failed"):
                    await client.evals.update_datapoint(
                        eval_id=uuid.uuid4(),
                        datapoint_id=uuid.uuid4(),
                        scores={"test": 1.0}
                    )
        finally:
            await client.close()

    def test_sync_client_create_evaluation_error(self):
        """Test error handling in sync create_evaluation."""
        client = LaminarClient(
            base_url="http://test-api.com",
            project_api_key="test-key"
        )
        
        try:
            with patch.object(client.evals, 'create_evaluation') as mock_create:
                mock_create.side_effect = ValueError("API Error")
                
                with pytest.raises(ValueError, match="API Error"):
                    client.evals.create_evaluation()
        finally:
            client.close()

    def test_sync_client_update_datapoint_error(self):
        """Test error handling in sync update_datapoint."""
        client = LaminarClient(
            base_url="http://test-api.com",
            project_api_key="test-key"
        )
        
        try:
            with patch.object(client.evals, 'update_datapoint') as mock_update:
                mock_update.side_effect = ValueError("Update failed")
                
                with pytest.raises(ValueError, match="Update failed"):
                    client.evals.update_datapoint(
                        eval_id=uuid.uuid4(),
                        datapoint_id=uuid.uuid4(),
                        scores={"test": 1.0}
                    )
        finally:
            client.close() 
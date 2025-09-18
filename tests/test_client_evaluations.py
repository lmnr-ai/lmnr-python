"""
Tests for the new evaluation methods on LaminarClient and AsyncLaminarClient.
"""

import pytest
import pytest_asyncio
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

from lmnr import LaminarClient, AsyncLaminarClient
from lmnr.sdk.types import PartialEvaluationDatapoint


class TestAsyncLaminarClientEvaluations:
    """Test evaluation methods on AsyncLaminarClient."""

    @pytest_asyncio.fixture
    async def async_client(self):
        """Create an AsyncLaminarClient for testing."""
        client = AsyncLaminarClient(
            base_url="http://test-api.com", project_api_key="test-key"
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
        with patch.object(
            async_client.evals, "create_evaluation", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_eval_response.id

            eval_id = await async_client.evals.create_evaluation(
                name="Test Evaluation",
                group_name="test_group",
                metadata={"metadata": "test metadata"},
            )

            assert eval_id == mock_eval_response.id
            mock_create.assert_called_once_with(
                name="Test Evaluation",
                group_name="test_group",
                metadata={"metadata": "test metadata"},
            )

    @pytest.mark.asyncio
    async def test_create_evaluation_with_defaults(
        self, async_client, mock_eval_response
    ):
        """Test evaluation creation with default parameters."""
        with patch.object(
            async_client.evals, "create_evaluation", new_callable=AsyncMock
        ) as mock_create:
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

        with patch.object(
            async_client.evals, "create_datapoint", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = expected_datapoint_id

            datapoint_id = await async_client.evals.create_datapoint(
                eval_id=eval_id,
                data=test_data,
                target=test_target,
                metadata=test_metadata,
                index=1,
                trace_id=custom_trace_id,
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
                trace_id=custom_trace_id,
            )

    @pytest.mark.asyncio
    async def test_create_datapoint_with_defaults(self, async_client):
        """Test datapoint creation with minimal parameters."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        test_data = {"input": "test input"}
        expected_datapoint_id = uuid.UUID("11111111-1111-1111-1111-111111111111")

        with patch.object(
            async_client.evals, "create_datapoint", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = expected_datapoint_id

            datapoint_id = await async_client.evals.create_datapoint(
                eval_id=eval_id, data=test_data
            )

            # Verify datapoint_id is correct
            assert datapoint_id == expected_datapoint_id

            # Verify create_datapoint was called correctly
            mock_create.assert_called_once_with(eval_id=eval_id, data=test_data)

    @pytest.mark.asyncio
    async def test_update_datapoint_success(self, async_client):
        """Test successful datapoint update."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        datapoint_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        executor_output = {"result": "processed output"}
        scores = {"accuracy": 0.95, "relevance": 0.88}

        with patch.object(
            async_client.evals, "update_datapoint", new_callable=AsyncMock
        ) as mock_update:
            await async_client.evals.update_datapoint(
                eval_id=eval_id,
                datapoint_id=datapoint_id,
                scores=scores,
                executor_output=executor_output,
            )

            mock_update.assert_called_once_with(
                eval_id=eval_id,
                datapoint_id=datapoint_id,
                scores=scores,
                executor_output=executor_output,
            )

    @pytest.mark.asyncio
    async def test_update_datapoint_with_minimal_params(self, async_client):
        """Test datapoint update with minimal parameters."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        datapoint_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        scores = {"accuracy": 0.95}

        with patch.object(
            async_client.evals, "update_datapoint", new_callable=AsyncMock
        ) as mock_update:
            await async_client.evals.update_datapoint(
                eval_id=eval_id, datapoint_id=datapoint_id, scores=scores
            )

            mock_update.assert_called_once_with(
                eval_id=eval_id, datapoint_id=datapoint_id, scores=scores
            )


class TestLaminarClientEvaluations:
    """Test evaluation methods on synchronous LaminarClient."""

    @pytest.fixture
    def sync_client(self):
        """Create a LaminarClient for testing."""
        client = LaminarClient(
            base_url="http://test-api.com", project_api_key="test-key"
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
        with patch.object(sync_client.evals, "create_evaluation") as mock_create:
            mock_create.return_value = mock_eval_response.id

            eval_id = sync_client.evals.create_evaluation(
                name="Test Evaluation",
                group_name="test_group",
                metadata={"metadata": "test metadata"},
            )

            assert eval_id == mock_eval_response.id
            mock_create.assert_called_once_with(
                name="Test Evaluation",
                group_name="test_group",
                metadata={"metadata": "test metadata"},
            )

    def test_create_evaluation_with_defaults(self, sync_client, mock_eval_response):
        """Test evaluation creation with default parameters."""
        with patch.object(sync_client.evals, "create_evaluation") as mock_create:
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

        with patch.object(sync_client.evals, "create_datapoint") as mock_create:
            mock_create.return_value = expected_datapoint_id

            datapoint_id = sync_client.evals.create_datapoint(
                eval_id=eval_id,
                data=test_data,
                target=test_target,
                metadata=test_metadata,
                index=1,
                trace_id=custom_trace_id,
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
                trace_id=custom_trace_id,
            )

    def test_create_datapoint_with_defaults(self, sync_client):
        """Test datapoint creation with minimal parameters."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        test_data = {"input": "test input"}
        expected_datapoint_id = uuid.UUID("11111111-1111-1111-1111-111111111111")

        with patch.object(sync_client.evals, "create_datapoint") as mock_create:
            mock_create.return_value = expected_datapoint_id

            datapoint_id = sync_client.evals.create_datapoint(
                eval_id=eval_id, data=test_data
            )

            # Verify datapoint_id is correct
            assert datapoint_id == expected_datapoint_id

            # Verify create_datapoint was called correctly
            mock_create.assert_called_once_with(eval_id=eval_id, data=test_data)

    def test_update_datapoint_success(self, sync_client):
        """Test successful datapoint update."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        datapoint_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        executor_output = {"result": "processed output"}
        scores = {"accuracy": 0.95, "relevance": 0.88}

        with patch.object(sync_client.evals, "update_datapoint") as mock_update:
            sync_client.evals.update_datapoint(
                eval_id=eval_id,
                datapoint_id=datapoint_id,
                scores=scores,
                executor_output=executor_output,
            )

            mock_update.assert_called_once_with(
                eval_id=eval_id,
                datapoint_id=datapoint_id,
                scores=scores,
                executor_output=executor_output,
            )

    def test_update_datapoint_with_minimal_params(self, sync_client):
        """Test datapoint update with minimal parameters."""
        eval_id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        datapoint_id = uuid.UUID("87654321-4321-8765-cba9-987654321abc")
        scores = {"accuracy": 0.95}

        with patch.object(sync_client.evals, "update_datapoint") as mock_update:
            sync_client.evals.update_datapoint(
                eval_id=eval_id, datapoint_id=datapoint_id, scores=scores
            )

            mock_update.assert_called_once_with(
                eval_id=eval_id, datapoint_id=datapoint_id, scores=scores
            )


class TestClientEvaluationIntegration:
    """Integration tests for the full evaluation workflow."""

    @pytest.mark.asyncio
    async def test_async_client_full_workflow(self):
        """Test the complete evaluation workflow with AsyncLaminarClient."""
        client = AsyncLaminarClient(
            base_url="http://test-api.com", project_api_key="test-key"
        )

        mock_eval_response = MagicMock()
        mock_eval_response.id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        mock_eval_response.projectId = "test-project-id"

        try:
            with (
                patch.object(
                    client.evals, "create_evaluation", new_callable=AsyncMock
                ) as mock_create_eval,
                patch.object(
                    client.evals, "create_datapoint", new_callable=AsyncMock
                ) as mock_create_dp,
                patch.object(
                    client.evals, "update_datapoint", new_callable=AsyncMock
                ) as mock_update,
            ):

                mock_create_eval.return_value = mock_eval_response.id
                mock_create_dp.return_value = uuid.UUID(
                    "22222222-2222-2222-2222-222222222222"
                )

                # Create evaluation
                eval_id = await client.evals.create_evaluation(name="Integration Test")

                # Create datapoint
                datapoint_id = await client.evals.create_datapoint(
                    eval_id=eval_id,
                    data={"query": "What is AI?"},
                    target={"answer": "Artificial Intelligence"},
                )

                # Update datapoint
                await client.evals.update_datapoint(
                    eval_id=eval_id,
                    datapoint_id=datapoint_id,
                    scores={"accuracy": 0.9},
                    executor_output={"response": "AI is artificial intelligence"},
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
            base_url="http://test-api.com", project_api_key="test-key"
        )

        mock_eval_response = MagicMock()
        mock_eval_response.id = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
        mock_eval_response.projectId = "test-project-id"

        try:
            with (
                patch.object(client.evals, "create_evaluation") as mock_create_eval,
                patch.object(client.evals, "create_datapoint") as mock_create_dp,
                patch.object(client.evals, "update_datapoint") as mock_update,
            ):

                mock_create_eval.return_value = mock_eval_response.id
                mock_create_dp.return_value = uuid.UUID(
                    "22222222-2222-2222-2222-222222222222"
                )

                # Create evaluation
                eval_id = client.evals.create_evaluation(name="Integration Test")

                # Create datapoint
                datapoint_id = client.evals.create_datapoint(
                    eval_id=eval_id,
                    data={"query": "What is AI?"},
                    target={"answer": "Artificial Intelligence"},
                )

                # Update datapoint
                client.evals.update_datapoint(
                    eval_id=eval_id,
                    datapoint_id=datapoint_id,
                    scores={"accuracy": 0.9},
                    executor_output={"response": "AI is artificial intelligence"},
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
            base_url="http://test-api.com", project_api_key="test-key"
        )

        try:
            with patch.object(
                client.evals, "create_evaluation", new_callable=AsyncMock
            ) as mock_create:
                mock_create.side_effect = ValueError("API Error")

                with pytest.raises(ValueError, match="API Error"):
                    await client.evals.create_evaluation()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_async_client_update_datapoint_error(self):
        """Test error handling in async update_datapoint."""
        client = AsyncLaminarClient(
            base_url="http://test-api.com", project_api_key="test-key"
        )

        try:
            with patch.object(
                client.evals, "update_datapoint", new_callable=AsyncMock
            ) as mock_update:
                mock_update.side_effect = ValueError("Update failed")

                with pytest.raises(ValueError, match="Update failed"):
                    await client.evals.update_datapoint(
                        eval_id=uuid.uuid4(),
                        datapoint_id=uuid.uuid4(),
                        scores={"test": 1.0},
                    )
        finally:
            await client.close()

    def test_sync_client_create_evaluation_error(self):
        """Test error handling in sync create_evaluation."""
        client = LaminarClient(
            base_url="http://test-api.com", project_api_key="test-key"
        )

        try:
            with patch.object(client.evals, "create_evaluation") as mock_create:
                mock_create.side_effect = ValueError("API Error")

                with pytest.raises(ValueError, match="API Error"):
                    client.evals.create_evaluation()
        finally:
            client.close()

    def test_sync_client_update_datapoint_error(self):
        """Test error handling in sync update_datapoint."""
        client = LaminarClient(
            base_url="http://test-api.com", project_api_key="test-key"
        )

        try:
            with patch.object(client.evals, "update_datapoint") as mock_update:
                mock_update.side_effect = ValueError("Update failed")

                with pytest.raises(ValueError, match="Update failed"):
                    client.evals.update_datapoint(
                        eval_id=uuid.uuid4(),
                        datapoint_id=uuid.uuid4(),
                        scores={"test": 1.0},
                    )
        finally:
            client.close()


class TestSyncClientRetryLogic:
    """Test retry logic for payload too large errors in sync client."""

    @pytest.fixture
    def sync_client(self):
        """Create a LaminarClient for testing."""
        client = LaminarClient(
            base_url="http://test-api.com", project_api_key="test-key"
        )
        yield client
        client.close()

    @pytest.fixture
    def sample_datapoints(self):
        """Create sample datapoints for testing."""
        return [
            PartialEvaluationDatapoint(
                id=uuid.uuid4(),
                data={"input": "large data " * 1000},  # Large data to trigger 413
                target={"expected": "output"},
                index=0,
                trace_id=uuid.uuid4(),
                executor_span_id=uuid.uuid4(),
                metadata={"test": "metadata"},
            ),
            PartialEvaluationDatapoint(
                id=uuid.uuid4(),
                data={"input": "more large data " * 1000},
                target={"expected": "output2"},
                index=1,
                trace_id=uuid.uuid4(),
                executor_span_id=uuid.uuid4(),
                metadata={"test": "metadata2"},
            ),
        ]

    def test_save_datapoints_success_on_first_try(self, sync_client, sample_datapoints):
        """Test successful save without needing retry."""
        eval_id = uuid.uuid4()

        # Mock successful response on first try
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(
            sync_client.evals._client, "post", return_value=mock_response
        ) as mock_post:
            sync_client.evals.save_datapoints(eval_id, sample_datapoints)

            # Should only be called once
            assert mock_post.call_count == 1
            # Verify the call was made with correct parameters
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert f"/v1/evals/{eval_id}/datapoints" in call_args[0][0]

    def test_save_datapoints_retry_on_413(self, sync_client, sample_datapoints):
        """Test retry logic when 413 Payload Too Large is returned."""
        eval_id = uuid.uuid4()

        # Mock 413 response first, then 200 on retry
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        # Patch the constant for testing to avoid large memory usage
        with patch(
            "lmnr.sdk.client.synchronous.resources.evals.INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH",
            1000,
        ):
            with patch.object(
                sync_client.evals._client,
                "post",
                side_effect=[mock_response_413, mock_response_200],
            ) as mock_post:
                sync_client.evals.save_datapoints(eval_id, sample_datapoints)

                # Should be called twice (initial + 1 retry)
                assert mock_post.call_count == 2
                # Verify both calls were made to the correct endpoint
                for call in mock_post.call_args_list:
                    assert f"/v1/evals/{eval_id}/datapoints" in call[0][0]

    def test_save_datapoints_multiple_retries(self, sync_client, sample_datapoints):
        """Test multiple retries with exponentially decreasing payload size."""
        eval_id = uuid.uuid4()

        # Mock multiple 413 responses, then success
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        # Patch the constant for testing
        with patch(
            "lmnr.sdk.client.synchronous.resources.evals.INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH",
            64,
        ):
            with patch.object(
                sync_client.evals._client,
                "post",
                side_effect=[
                    mock_response_413,
                    mock_response_413,
                    mock_response_413,
                    mock_response_200,
                ],
            ) as mock_post:
                sync_client.evals.save_datapoints(eval_id, sample_datapoints)

                # Should be called 4 times (initial + 3 retries)
                assert mock_post.call_count == 4

    def test_save_datapoints_max_retries_exceeded(self, sync_client, sample_datapoints):
        """Test that ValueError is raised when max retries is exceeded."""
        eval_id = uuid.uuid4()

        # Mock 413 response every time
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413
        mock_response_413.text = "Payload too large"

        # Patch the constant and max_retries for testing
        with patch(
            "lmnr.sdk.client.synchronous.resources.evals.INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH",
            64,
        ):
            with patch.object(
                sync_client.evals._client, "post", return_value=mock_response_413
            ) as mock_post:
                with pytest.raises(
                    ValueError, match="Error saving evaluation datapoints"
                ):
                    sync_client.evals.save_datapoints(eval_id, sample_datapoints)

                # Should be called initial + max_retries times (21 total with default max_retries=20)
                assert mock_post.call_count == 21

    def test_save_datapoints_length_becomes_zero(self, sync_client, sample_datapoints):
        """Test that ValueError is raised when data length becomes 0."""
        eval_id = uuid.uuid4()

        # Mock 413 response every time
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413

        # Start with very small initial length so it quickly becomes 0
        with patch(
            "lmnr.sdk.client.synchronous.resources.evals.INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH",
            2,
        ):
            # Override the _retry_save_datapoints method to use the patched constant
            with patch.object(
                sync_client.evals, "_retry_save_datapoints"
            ) as mock_retry:
                mock_retry.side_effect = ValueError(
                    "Error saving evaluation datapoints"
                )

                with patch.object(
                    sync_client.evals._client, "post", return_value=mock_response_413
                ) as mock_post:
                    with pytest.raises(
                        ValueError, match="Error saving evaluation datapoints"
                    ):
                        sync_client.evals.save_datapoints(eval_id, sample_datapoints)

                    # Should be called once (initial call gets 413, then retry method is called)
                    assert mock_post.call_count == 1
                    mock_retry.assert_called_once_with(eval_id, sample_datapoints, None)

    def test_save_datapoints_other_error_codes(self, sync_client, sample_datapoints):
        """Test that other error codes (not 413) are handled properly."""
        eval_id = uuid.uuid4()

        # Mock 500 response (server error)
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500
        mock_response_500.text = "Internal server error"

        with patch.object(
            sync_client.evals._client, "post", return_value=mock_response_500
        ) as mock_post:
            with pytest.raises(
                ValueError,
                match="Error saving evaluation datapoints.*500.*Internal server error",
            ):
                sync_client.evals.save_datapoints(eval_id, sample_datapoints)

            # Should only be called once (no retry for non-413 errors)
            assert mock_post.call_count == 1

    def test_save_datapoints_with_group_name(self, sync_client, sample_datapoints):
        """Test that group_name is properly passed through retries."""
        eval_id = uuid.uuid4()
        group_name = "test_group"

        # Mock 413 response first, then 200 on retry
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        with patch(
            "lmnr.sdk.client.synchronous.resources.evals.INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH",
            1000,
        ):
            with patch.object(
                sync_client.evals._client,
                "post",
                side_effect=[mock_response_413, mock_response_200],
            ) as mock_post:
                sync_client.evals.save_datapoints(
                    eval_id, sample_datapoints, group_name=group_name
                )

                # Verify group_name was passed in both calls
                for call in mock_post.call_args_list:
                    call_json = call[1]["json"]
                    assert call_json["groupName"] == group_name

    def test_retry_save_datapoints_data_length_halving(
        self, sync_client, sample_datapoints
    ):
        """Test that data length is halved with each retry."""
        eval_id = uuid.uuid4()
        initial_length = 1000

        # Mock the _retry_save_datapoints method to capture the length parameter
        original_to_dict = PartialEvaluationDatapoint.to_dict
        captured_lengths = []

        def mock_to_dict(self, max_data_length=None):
            captured_lengths.append(max_data_length)
            return original_to_dict(self, max_data_length)

        # Mock responses: multiple 413s then success
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        with patch(
            "lmnr.sdk.client.synchronous.resources.evals.INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH",
            initial_length,
        ):
            with patch.object(PartialEvaluationDatapoint, "to_dict", mock_to_dict):
                # Mock the retry method to pass the correct initial_length
                original_retry = sync_client.evals._retry_save_datapoints

                def mock_retry_with_length(eval_id, datapoints, group_name):
                    return original_retry(
                        eval_id, datapoints, group_name, initial_length=initial_length
                    )

                with patch.object(
                    sync_client.evals,
                    "_retry_save_datapoints",
                    side_effect=mock_retry_with_length,
                ):
                    with patch.object(
                        sync_client.evals._client,
                        "post",
                        side_effect=[
                            mock_response_413,
                            mock_response_413,
                            mock_response_200,
                        ],
                    ):
                        sync_client.evals.save_datapoints(eval_id, sample_datapoints)

        # Each datapoint calls to_dict, so we get multiple calls per request
        # Just check that we have the expected pattern of halving
        unique_lengths = []
        for length in captured_lengths:
            if length not in unique_lengths:
                unique_lengths.append(length)

        assert len(unique_lengths) == 3
        assert unique_lengths[0] == initial_length
        assert unique_lengths[1] == initial_length // 2
        assert unique_lengths[2] == initial_length // 4

    def test_retry_save_datapoints_direct_length_test(
        self, sync_client, sample_datapoints
    ):
        """Test the retry method directly with a small initial length to verify length becomes 0."""
        eval_id = uuid.uuid4()

        # Mock 413 response every time
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413

        with patch.object(
            sync_client.evals._client, "post", return_value=mock_response_413
        ) as mock_post:
            with pytest.raises(ValueError, match="Error saving evaluation datapoints"):
                # Call the retry method directly with a small initial length
                sync_client.evals._retry_save_datapoints(
                    eval_id, sample_datapoints, None, initial_length=2
                )

            # Should be called only once:
            # - retry iteration 1: length = 2 // 2 = 1, makes HTTP call
            # - retry iteration 2: length = 1 // 2 = 0, raises error immediately (no HTTP call)
            assert mock_post.call_count == 1


class TestAsyncClientRetryLogic:
    """Test retry logic for payload too large errors in async client."""

    @pytest_asyncio.fixture
    async def async_client(self):
        """Create an AsyncLaminarClient for testing."""
        client = AsyncLaminarClient(
            base_url="http://test-api.com", project_api_key="test-key"
        )
        yield client
        await client.close()

    @pytest.fixture
    def sample_datapoints(self):
        """Create sample datapoints for testing."""
        return [
            PartialEvaluationDatapoint(
                id=uuid.uuid4(),
                data={"input": "large data " * 1000},  # Large data to trigger 413
                target={"expected": "output"},
                index=0,
                trace_id=uuid.uuid4(),
                executor_span_id=uuid.uuid4(),
                metadata={"test": "metadata"},
            ),
            PartialEvaluationDatapoint(
                id=uuid.uuid4(),
                data={"input": "more large data " * 1000},
                target={"expected": "output2"},
                index=1,
                trace_id=uuid.uuid4(),
                executor_span_id=uuid.uuid4(),
                metadata={"test": "metadata2"},
            ),
        ]

    @pytest.mark.asyncio
    async def test_save_datapoints_success_on_first_try(
        self, async_client, sample_datapoints
    ):
        """Test successful save without needing retry."""
        eval_id = uuid.uuid4()

        # Mock successful response on first try
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(
            async_client.evals._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            await async_client.evals.save_datapoints(eval_id, sample_datapoints)

            # Should only be called once
            assert mock_post.call_count == 1
            # Verify the call was made with correct parameters
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert f"/v1/evals/{eval_id}/datapoints" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_save_datapoints_retry_on_413(self, async_client, sample_datapoints):
        """Test retry logic when 413 Payload Too Large is returned."""
        eval_id = uuid.uuid4()

        # Mock 413 response first, then 200 on retry
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        # Patch the constant for testing to avoid large memory usage
        with patch(
            "lmnr.sdk.client.asynchronous.resources.evals.INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH",
            1000,
        ):
            with patch.object(
                async_client.evals._client,
                "post",
                new_callable=AsyncMock,
                side_effect=[mock_response_413, mock_response_200],
            ) as mock_post:
                await async_client.evals.save_datapoints(eval_id, sample_datapoints)

                # Should be called twice (initial + 1 retry)
                assert mock_post.call_count == 2
                # Verify both calls were made to the correct endpoint
                for call in mock_post.call_args_list:
                    assert f"/v1/evals/{eval_id}/datapoints" in call[0][0]

    @pytest.mark.asyncio
    async def test_save_datapoints_multiple_retries(
        self, async_client, sample_datapoints
    ):
        """Test multiple retries with exponentially decreasing payload size."""
        eval_id = uuid.uuid4()

        # Mock multiple 413 responses, then success
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        # Patch the constant for testing
        with patch(
            "lmnr.sdk.client.asynchronous.resources.evals.INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH",
            64,
        ):
            with patch.object(
                async_client.evals._client,
                "post",
                new_callable=AsyncMock,
                side_effect=[
                    mock_response_413,
                    mock_response_413,
                    mock_response_413,
                    mock_response_200,
                ],
            ) as mock_post:
                await async_client.evals.save_datapoints(eval_id, sample_datapoints)

                # Should be called 4 times (initial + 3 retries)
                assert mock_post.call_count == 4

    @pytest.mark.asyncio
    async def test_save_datapoints_max_retries_exceeded(
        self, async_client, sample_datapoints
    ):
        """Test that ValueError is raised when max retries is exceeded."""
        eval_id = uuid.uuid4()

        # Mock 413 response every time
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413
        mock_response_413.text = "Payload too large"

        # Patch the constant and max_retries for testing
        with patch(
            "lmnr.sdk.client.asynchronous.resources.evals.INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH",
            64,
        ):
            with patch.object(
                async_client.evals._client,
                "post",
                new_callable=AsyncMock,
                return_value=mock_response_413,
            ) as mock_post:
                with pytest.raises(
                    ValueError, match="Error saving evaluation datapoints"
                ):
                    await async_client.evals.save_datapoints(eval_id, sample_datapoints)

                # Should be called initial + max_retries times (21 total with default max_retries=20)
                assert mock_post.call_count == 21

    @pytest.mark.asyncio
    async def test_save_datapoints_length_becomes_zero(
        self, async_client, sample_datapoints
    ):
        """Test that ValueError is raised when data length becomes 0."""
        eval_id = uuid.uuid4()

        # Mock 413 response every time
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413

        # Start with very small initial length so it quickly becomes 0
        with patch(
            "lmnr.sdk.client.asynchronous.resources.evals.INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH",
            2,
        ):
            # Override the _retry_save_datapoints method to use the patched constant
            with patch.object(
                async_client.evals, "_retry_save_datapoints", new_callable=AsyncMock
            ) as mock_retry:
                mock_retry.side_effect = ValueError(
                    "Error saving evaluation datapoints"
                )

                with patch.object(
                    async_client.evals._client,
                    "post",
                    new_callable=AsyncMock,
                    return_value=mock_response_413,
                ) as mock_post:
                    with pytest.raises(
                        ValueError, match="Error saving evaluation datapoints"
                    ):
                        await async_client.evals.save_datapoints(
                            eval_id, sample_datapoints
                        )

                    # Should be called once (initial call gets 413, then retry method is called)
                    assert mock_post.call_count == 1
                    mock_retry.assert_called_once_with(eval_id, sample_datapoints, None)

    @pytest.mark.asyncio
    async def test_save_datapoints_other_error_codes(
        self, async_client, sample_datapoints
    ):
        """Test that other error codes (not 413) are handled properly."""
        eval_id = uuid.uuid4()

        # Mock 500 response (server error)
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500
        mock_response_500.text = "Internal server error"

        with patch.object(
            async_client.evals._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response_500,
        ) as mock_post:
            with pytest.raises(
                ValueError,
                match="Error saving evaluation datapoints.*500.*Internal server error",
            ):
                await async_client.evals.save_datapoints(eval_id, sample_datapoints)

            # Should only be called once (no retry for non-413 errors)
            assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_save_datapoints_with_group_name(
        self, async_client, sample_datapoints
    ):
        """Test that group_name is properly passed through retries."""
        eval_id = uuid.uuid4()
        group_name = "test_group"

        # Mock 413 response first, then 200 on retry
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        with patch(
            "lmnr.sdk.client.asynchronous.resources.evals.INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH",
            1000,
        ):
            with patch.object(
                async_client.evals._client,
                "post",
                new_callable=AsyncMock,
                side_effect=[mock_response_413, mock_response_200],
            ) as mock_post:
                await async_client.evals.save_datapoints(
                    eval_id, sample_datapoints, group_name=group_name
                )

                # Verify group_name was passed in both calls
                for call in mock_post.call_args_list:
                    call_json = call[1]["json"]
                    assert call_json["groupName"] == group_name

    @pytest.mark.asyncio
    async def test_retry_save_datapoints_data_length_halving(
        self, async_client, sample_datapoints
    ):
        """Test that data length is halved with each retry."""
        eval_id = uuid.uuid4()
        initial_length = 1000

        # Mock the _retry_save_datapoints method to capture the length parameter
        original_to_dict = PartialEvaluationDatapoint.to_dict
        captured_lengths = []

        def mock_to_dict(self, max_data_length=None):
            captured_lengths.append(max_data_length)
            return original_to_dict(self, max_data_length)

        # Mock responses: multiple 413s then success
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        with patch(
            "lmnr.sdk.client.asynchronous.resources.evals.INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH",
            initial_length,
        ):
            with patch.object(PartialEvaluationDatapoint, "to_dict", mock_to_dict):
                # Mock the retry method to pass the correct initial_length
                original_retry = async_client.evals._retry_save_datapoints

                async def mock_retry_with_length(eval_id, datapoints, group_name):
                    return await original_retry(
                        eval_id, datapoints, group_name, initial_length=initial_length
                    )

                with patch.object(
                    async_client.evals,
                    "_retry_save_datapoints",
                    side_effect=mock_retry_with_length,
                ):
                    with patch.object(
                        async_client.evals._client,
                        "post",
                        new_callable=AsyncMock,
                        side_effect=[
                            mock_response_413,
                            mock_response_413,
                            mock_response_200,
                        ],
                    ):
                        await async_client.evals.save_datapoints(
                            eval_id, sample_datapoints
                        )

        # Each datapoint calls to_dict, so we get multiple calls per request
        # Just check that we have the expected pattern of halving
        unique_lengths = []
        for length in captured_lengths:
            if length not in unique_lengths:
                unique_lengths.append(length)

        assert len(unique_lengths) == 3
        assert unique_lengths[0] == initial_length
        assert unique_lengths[1] == initial_length // 2
        assert unique_lengths[2] == initial_length // 4

    @pytest.mark.asyncio
    async def test_retry_save_datapoints_direct_length_test(
        self, async_client, sample_datapoints
    ):
        """Test the retry method directly with a small initial length to verify length becomes 0."""
        eval_id = uuid.uuid4()

        # Mock 413 response every time
        mock_response_413 = MagicMock()
        mock_response_413.status_code = 413

        with patch.object(
            async_client.evals._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response_413,
        ) as mock_post:
            with pytest.raises(ValueError, match="Error saving evaluation datapoints"):
                # Call the retry method directly with a small initial length
                await async_client.evals._retry_save_datapoints(
                    eval_id, sample_datapoints, None, initial_length=2
                )

            # Should be called only once:
            # - retry iteration 1: length = 2 // 2 = 1, makes HTTP call
            # - retry iteration 2: length = 1 // 2 = 0, raises error immediately (no HTTP call)
            assert mock_post.call_count == 1

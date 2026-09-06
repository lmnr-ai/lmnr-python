import json
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr import Laminar
from lmnr.sdk.evaluations import evaluate
from lmnr.sdk.types import Datapoint


# Fixtures for common mock objects
@pytest.fixture
def mock_eval_response():
    """Create a mock evaluation response."""
    return {
        "id": "00000000-0000-0000-0000-000000000000",
        "projectId": "mock-project-id",
    }


@pytest.fixture
def mock_datapoints_response():
    """Create a mock datapoints response."""
    return MagicMock()


@pytest.fixture
def mock_dataset_push_response():
    """Create a mock dataset push response."""
    return {"dataset_id": "00000000-0000-0000-0000-000000000001"}


@pytest.fixture
def mock_dataset_pull_response():
    """Create a mock dataset pull response."""
    return {
        "items": [
            Datapoint(
                id=uuid.uuid4(),
                data="test",
                target="test",
                createdAt=datetime.now(),
            )
        ],
        "total_count": 1,
    }


# Helper functions for common test logic
def verify_basic_api_calls(
    mock_init, mock_dataset_push, mock_dataset_pull, mock_datapoints
):
    """Verify the expected API calls were made."""
    mock_init.assert_called_once()
    # TODO: add tests with Laminar dataset and verify pull
    assert mock_datapoints.call_count == 2


def verify_basic_spans(spans, expected_evaluator_names):
    """Verify the basic span structure and return categorized spans."""
    evaluation_span = next(
        (
            span
            for span in spans
            if span.attributes.get("lmnr.span.type") == "EVALUATION"
        ),
        None,
    )
    executor_span = next(
        (span for span in spans if span.attributes.get("lmnr.span.type") == "EXECUTOR"),
        None,
    )
    evaluator_spans = [
        span for span in spans if span.attributes.get("lmnr.span.type") == "EVALUATOR"
    ]

    assert evaluation_span.name == "evaluation"
    assert executor_span.name == "executor"
    assert sorted([span.name for span in evaluator_spans]) == sorted(
        expected_evaluator_names
    )

    return evaluation_span, executor_span, evaluator_spans


def verify_human_evaluator_spans(spans, expected_human_evaluator_names):
    """Verify human evaluator spans and return them."""
    human_evaluator_spans = [
        span
        for span in spans
        if span.attributes.get("lmnr.span.type") == "HUMAN_EVALUATOR"
    ]
    assert sorted([span.name for span in human_evaluator_spans]) == sorted(
        expected_human_evaluator_names
    )

    for human_span in human_evaluator_spans:
        assert human_span.attributes.get("lmnr.span.type") == "HUMAN_EVALUATOR"

    return human_evaluator_spans


@pytest.mark.asyncio
@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.datasets.AsyncDatasets.push")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
async def test_evaluate_with_mocks_async(
    mock_init,
    mock_datapoints,
    mock_dataset_push,
    mock_dataset_pull,
    mock_eval_response,
    mock_datapoints_response,
    mock_dataset_push_response,
    mock_dataset_pull_response,
    span_exporter: InMemorySpanExporter,
):
    """Test the evaluate function with mocked API calls (async)."""
    # Set up mock return values
    mock_init.return_value = mock_eval_response
    mock_datapoints.return_value = mock_datapoints_response
    mock_dataset_push.return_value = mock_dataset_push_response
    mock_dataset_pull.return_value = mock_dataset_pull_response

    # Run the evaluate function
    await evaluate(
        data=[{"data": "test", "target": "test"}],
        executor=lambda data: data,
        evaluators={
            "test": lambda output, target: 1 if output == target else 0,
            "test2": lambda output, target: 1 if output == target else 0,
        },
        project_api_key="test",
    )

    # Flush the traces
    Laminar.flush()

    # Verify the API calls
    verify_basic_api_calls(
        mock_init, mock_dataset_push, mock_dataset_pull, mock_datapoints
    )

    # Get the finished spans and verify
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    verify_basic_spans(spans, ["test", "test2"])


@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.datasets.AsyncDatasets.push")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
def test_evaluate_with_mocks(
    mock_init,
    mock_datapoints,
    mock_dataset_push,
    mock_dataset_pull,
    mock_eval_response,
    mock_datapoints_response,
    mock_dataset_push_response,
    mock_dataset_pull_response,
    span_exporter: InMemorySpanExporter,
):
    """Test the evaluate function with mocked API calls (sync)."""
    # Set up mock return values
    mock_init.return_value = mock_eval_response
    mock_datapoints.return_value = mock_datapoints_response
    mock_dataset_push.return_value = mock_dataset_push_response
    mock_dataset_pull.return_value = mock_dataset_pull_response

    # Run the evaluate function
    evaluate(
        data=[{"data": "test", "target": "test"}],
        executor=lambda data: data,
        evaluators={
            "test": lambda output, target: 1 if output == target else 0,
            "test2": lambda output, target: 1 if output == target else 0,
        },
        project_api_key="test",
    )

    # Flush the traces
    Laminar.flush()

    # Verify the API calls
    verify_basic_api_calls(
        mock_init, mock_dataset_push, mock_dataset_pull, mock_datapoints
    )

    # Get the finished spans and verify
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    verify_basic_spans(spans, ["test", "test2"])


@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.datasets.AsyncDatasets.push")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
def test_evaluate_after_init(
    mock_init,
    mock_datapoints,
    mock_dataset_push,
    mock_dataset_pull,
    mock_eval_response,
    mock_datapoints_response,
    mock_dataset_push_response,
    mock_dataset_pull_response,
    span_exporter: InMemorySpanExporter,
):
    """Test the evaluate function after Laminar.initialize() has been called."""
    # Set up mock return values
    mock_init.return_value = mock_eval_response
    mock_datapoints.return_value = mock_datapoints_response
    mock_dataset_push.return_value = mock_dataset_push_response
    mock_dataset_pull.return_value = mock_dataset_pull_response

    Laminar.initialize(project_api_key="test")

    # Run the evaluate function
    evaluate(
        data=[{"data": "test", "target": "test"}],
        executor=lambda data: data,
        evaluators={
            "test": lambda output, target: 1 if output == target else 0,
            "test2": lambda output, target: 1 if output == target else 0,
        },
        project_api_key="test",
    )

    # Flush the traces
    Laminar.flush()

    # Verify the API calls
    verify_basic_api_calls(
        mock_init, mock_dataset_push, mock_dataset_pull, mock_datapoints
    )

    # Get the finished spans and verify
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    verify_basic_spans(spans, ["test", "test2"])


@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.datasets.AsyncDatasets.push")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
def test_evaluate_with_flush(
    mock_init,
    mock_datapoints,
    mock_dataset_push,
    mock_dataset_pull,
    mock_eval_response,
    mock_datapoints_response,
    mock_dataset_push_response,
    mock_dataset_pull_response,
    span_exporter: InMemorySpanExporter,
):
    """Test the evaluate function when flush is called within the executor."""
    # Set up mock return values
    mock_init.return_value = mock_eval_response
    mock_datapoints.return_value = mock_datapoints_response
    mock_dataset_push.return_value = mock_dataset_push_response
    mock_dataset_pull.return_value = mock_dataset_pull_response

    def mock_executor(data):
        Laminar.flush()
        return data

    # Run the evaluate function
    evaluate(
        data=[{"data": "test", "target": "test"}],
        executor=mock_executor,
        evaluators={
            "test": lambda output, target: 1 if output == target else 0,
            "test2": lambda output, target: 1 if output == target else 0,
        },
        project_api_key="test",
    )

    # Flush the traces
    Laminar.flush()

    # Verify the API calls
    verify_basic_api_calls(
        mock_init, mock_dataset_push, mock_dataset_pull, mock_datapoints
    )

    # Get the finished spans and verify
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    verify_basic_spans(spans, ["test", "test2"])


@pytest.mark.asyncio
@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.datasets.AsyncDatasets.push")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
async def test_evaluate_propagates_evaluation_id_to_all_spans(
    mock_init,
    mock_datapoints,
    mock_dataset_push,
    mock_dataset_pull,
    mock_eval_response,
    mock_datapoints_response,
    mock_dataset_push_response,
    mock_dataset_pull_response,
    span_exporter: InMemorySpanExporter,
):
    """Every span produced inside an evaluate() trace — including child spans
    created inside the user's executor — should carry
    `lmnr.association.properties.metadata.evaluation_id` equal to the eval id
    returned by `evals.init`."""
    mock_init.return_value = mock_eval_response
    mock_datapoints.return_value = mock_datapoints_response
    mock_dataset_push.return_value = mock_dataset_push_response
    mock_dataset_pull.return_value = mock_dataset_pull_response

    def executor(data):
        with Laminar.start_as_current_span("inner_user_span"):
            return data

    await evaluate(
        data=[
            {"data": "a", "target": "a"},
            {"data": "b", "target": "b"},
        ],
        executor=executor,
        evaluators={
            "exact": lambda output, target: 1 if output == target else 0,
        },
        project_api_key="test",
    )

    Laminar.flush()

    spans = span_exporter.get_finished_spans()
    # 2 datapoints * (evaluation + executor + inner_user_span + 1 evaluator) = 8
    assert len(spans) == 8

    expected_eval_id = mock_eval_response["id"]
    for span in spans:
        actual = span.attributes.get(
            "lmnr.association.properties.metadata.evaluation_id"
        )
        assert actual == expected_eval_id, (
            f"span {span.name} missing/incorrect evaluation_id: "
            f"got {actual!r}, expected {expected_eval_id!r}"
        )

    trace_ids = {span.context.trace_id for span in spans}
    assert len(trace_ids) == 2

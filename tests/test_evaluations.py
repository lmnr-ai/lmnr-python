import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from lmnr import Laminar
from lmnr.sdk.evaluations import evaluate, get_average_scores
from lmnr.sdk.types import HumanEvaluator, EvaluationResultDatapoint, Datapoint
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import uuid


# Fixtures for common mock objects
@pytest.fixture
def mock_eval_response():
    """Create a mock evaluation response."""
    response = MagicMock()
    response.id = "00000000-0000-0000-0000-000000000000"
    response.projectId = "mock-project-id"
    return response


@pytest.fixture
def mock_datapoints_response():
    """Create a mock datapoints response."""
    return MagicMock()


@pytest.fixture
def mock_dataset_push_response():
    """Create a mock dataset push response."""
    response = MagicMock()
    response.dataset_id = "00000000-0000-0000-0000-000000000001"
    return response


@pytest.fixture
def mock_dataset_pull_response():
    """Create a mock dataset pull response."""
    response = MagicMock()
    response.items = [
        Datapoint(
            id=uuid.uuid4(),
            data="test",
            target="test",
            createdAt=datetime.now(),
        )
    ]
    response.total_count = 1
    return response


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
async def test_evaluate_with_human_evaluator_async(
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
    """Test the evaluate function with HumanEvaluator instances (async)."""
    # Set up mock return values
    mock_init.return_value = mock_eval_response
    mock_datapoints.return_value = mock_datapoints_response
    mock_dataset_push.return_value = mock_dataset_push_response
    mock_dataset_pull.return_value = mock_dataset_pull_response

    options = [
        {"label": "relevant", "value": 1},
        {"label": "irrelevant", "value": 0},
    ]

    # Create HumanEvaluator instances
    human_evaluator1 = HumanEvaluator()
    human_evaluator2 = HumanEvaluator(options=options)

    # Run the evaluate function with mixed evaluators
    result = await evaluate(
        data=[{"data": "test", "target": "test"}],
        executor=lambda data: data,
        evaluators={
            "accuracy": lambda output, target: 1 if output == target else 0,
            "human_quality": human_evaluator1,
            "human_relevance": human_evaluator2,
        },
        project_api_key="test",
    )

    # Flush the traces
    Laminar.flush()

    # Verify the API calls
    verify_basic_api_calls(
        mock_init, mock_dataset_push, mock_dataset_pull, mock_datapoints
    )

    # Verify return values - human evaluator scores should be None
    scores = result["average_scores"]
    assert "accuracy" in scores
    assert scores["accuracy"] == 1  # Regular evaluator should have a score
    # Human evaluators should not contribute to average scores since they're None
    assert "human_quality" not in scores
    assert "human_relevance" not in scores

    # Get the finished spans and verify
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 5  # evaluation, executor, 1 evaluator, 2 human_evaluators

    # Verify basic spans
    evaluation_span, executor_span, evaluator_spans = verify_basic_spans(
        spans, ["accuracy"]
    )
    assert len(evaluator_spans) == 1

    # Verify human evaluator spans
    verify_human_evaluator_spans(spans, ["human_quality", "human_relevance"])

    # Verify options are correctly set on human_relevance span
    options_span = next(
        span
        for span in spans
        if span.attributes.get("lmnr.span.type") == "HUMAN_EVALUATOR"
        and span.name == "human_relevance"
    )
    assert (
        json.loads(options_span.attributes.get("lmnr.span.human_evaluator_options"))
        == options
    )


@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.datasets.AsyncDatasets.push")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
def test_evaluate_with_human_evaluator(
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
    """Test the evaluate function with HumanEvaluator instances (sync)."""
    # Set up mock return values
    mock_init.return_value = mock_eval_response
    mock_datapoints.return_value = mock_datapoints_response
    mock_dataset_push.return_value = mock_dataset_push_response
    mock_dataset_pull.return_value = mock_dataset_pull_response

    options = [
        {"label": "relevant", "value": 1},
        {"label": "irrelevant", "value": 0},
    ]

    # Create HumanEvaluator instances
    human_evaluator1 = HumanEvaluator()
    human_evaluator2 = HumanEvaluator(options=options)

    # Run the evaluate function with mixed evaluators
    result = evaluate(
        data=[{"data": "test", "target": "test"}],
        executor=lambda data: data,
        evaluators={
            "precision": lambda output, target: 0.9,
            "human_quality": human_evaluator1,
            "human_relevance": human_evaluator2,
        },
        project_api_key="test",
    )

    # Flush the traces
    Laminar.flush()

    # Verify the API calls
    verify_basic_api_calls(
        mock_init, mock_dataset_push, mock_dataset_pull, mock_datapoints
    )

    # Verify return values
    scores = result["average_scores"]
    assert "precision" in scores
    assert scores["precision"] == 0.9
    # Human evaluator should not be in results since score is None
    assert "human_evaluation" not in scores

    # Get the finished spans and verify
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 5  # evaluation, executor, evaluator, 2x human_evaluator

    # Verify basic spans
    verify_basic_spans(spans, ["precision"])

    # Verify human evaluator spans
    verify_human_evaluator_spans(spans, ["human_quality", "human_relevance"])

    # Verify options are correctly set on human_relevance span
    options_span = next(
        span
        for span in spans
        if span.attributes.get("lmnr.span.type") == "HUMAN_EVALUATOR"
        and span.name == "human_relevance"
    )
    assert (
        json.loads(options_span.attributes.get("lmnr.span.human_evaluator_options"))
        == options
    )


def test_get_average_scores_with_human_evaluators():
    """
    Test that get_average_scores correctly handles HumanEvaluator scores (None values).
    """
    # Create test datapoints with mixed scores (some None from HumanEvaluators)
    datapoints = [
        EvaluationResultDatapoint(
            id=uuid.uuid4(),
            index=0,
            data="test1",
            target="target1",
            executor_output="output1",
            scores={"accuracy": 0.8, "human_quality": None, "precision": 0.9},
            trace_id=uuid.uuid4(),
            executor_span_id=uuid.uuid4(),
        ),
        EvaluationResultDatapoint(
            id=uuid.uuid4(),
            index=1,
            data="test2",
            target="target2",
            executor_output="output2",
            scores={"accuracy": 0.6, "human_quality": None, "precision": 0.7},
            trace_id=uuid.uuid4(),
            executor_span_id=uuid.uuid4(),
        ),
        EvaluationResultDatapoint(
            id=uuid.uuid4(),
            index=2,
            data="test3",
            target="target3",
            executor_output="output3",
            scores={"accuracy": 1.0, "human_quality": None, "precision": 0.8},
            trace_id=uuid.uuid4(),
            executor_span_id=uuid.uuid4(),
        ),
    ]

    # Calculate average scores
    average_scores = get_average_scores(datapoints)

    # Verify that None scores (from HumanEvaluators) are excluded
    assert "accuracy" in average_scores
    assert "precision" in average_scores
    assert (
        "human_quality" not in average_scores
    )  # Should not be included since all values are None

    # Verify correct averages
    assert abs(average_scores["accuracy"] - (0.8 + 0.6 + 1.0) / 3) < 1e-10  # ~0.8
    assert abs(average_scores["precision"] - (0.9 + 0.7 + 0.8) / 3) < 1e-10  # ~0.8

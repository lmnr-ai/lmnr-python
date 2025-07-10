import pytest
from unittest.mock import patch, MagicMock

from lmnr import Laminar
from lmnr.sdk.evaluations import evaluate, get_average_scores
from lmnr.sdk.types import HumanEvaluator, EvaluationResultDatapoint
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import uuid


@pytest.mark.asyncio
async def test_evaluate_with_mocks_async(span_exporter: InMemorySpanExporter):
    """
    Test the evaluate function with mocked API calls.
    """
    mock_eval_id = "00000000-0000-0000-0000-000000000000"

    # Mock the init eval endpoint
    init_eval_response = MagicMock()
    init_eval_response.id = mock_eval_id
    init_eval_response.projectId = "mock-project-id"

    # Mock the datapoints endpoint
    datapoints_response = MagicMock()

    # Patch the AsyncLaminarClient._evals.init method
    with patch(
        "lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init",
        return_value=init_eval_response,
    ) as mock_init:
        # Patch the datapoints method
        with patch(
            "lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints",
            return_value=datapoints_response,
        ) as mock_datapoints:

            # Run the evaluate function
            await evaluate(
                data=[
                    {
                        "data": "test",
                        "target": "test",
                    },
                ],
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
            mock_init.assert_called_once()
            assert mock_datapoints.call_count == 2  # Called once for each evaluator

            # Get the finished spans
            spans = span_exporter.get_finished_spans()

            # Verify the spans
            assert len(spans) == 4

            # Find the specific span types
            evaluation_span = next(
                (
                    span
                    for span in spans
                    if span.attributes.get("lmnr.span.type") == "EVALUATION"
                ),
                None,
            )
            executor_span = next(
                (
                    span
                    for span in spans
                    if span.attributes.get("lmnr.span.type") == "EXECUTOR"
                ),
                None,
            )
            evaluator_spans = [
                span
                for span in spans
                if span.attributes.get("lmnr.span.type") == "EVALUATOR"
            ]

            # Verify the span names
            assert evaluation_span.name == "evaluation"
            assert executor_span.name == "executor"
            assert sorted([span.name for span in evaluator_spans]) == ["test", "test2"]


def test_evaluate_with_mocks(span_exporter: InMemorySpanExporter):
    """
    Test the evaluate function with mocked API calls.
    """
    mock_eval_id = "00000000-0000-0000-0000-000000000000"

    # Mock the init eval endpoint
    init_eval_response = MagicMock()
    init_eval_response.id = mock_eval_id
    init_eval_response.projectId = "mock-project-id"

    # Mock the datapoints endpoint
    datapoints_response = MagicMock()

    # Patch the AsyncLaminarClient._evals.init method
    with patch(
        "lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init",
        return_value=init_eval_response,
    ) as mock_init:
        # Patch the datapoints method
        with patch(
            "lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints",
            return_value=datapoints_response,
        ) as mock_datapoints:

            # Run the evaluate function
            evaluate(
                data=[
                    {
                        "data": "test",
                        "target": "test",
                    },
                ],
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
            mock_init.assert_called_once()
            assert mock_datapoints.call_count == 2  # Called once for each evaluator

            # Get the finished spans
            spans = span_exporter.get_finished_spans()

            # Verify the spans
            assert len(spans) == 4

            # Find the specific span types
            evaluation_span = next(
                (
                    span
                    for span in spans
                    if span.attributes.get("lmnr.span.type") == "EVALUATION"
                ),
                None,
            )
            executor_span = next(
                (
                    span
                    for span in spans
                    if span.attributes.get("lmnr.span.type") == "EXECUTOR"
                ),
                None,
            )
            evaluator_spans = [
                span
                for span in spans
                if span.attributes.get("lmnr.span.type") == "EVALUATOR"
            ]

            # Verify the span names
            assert evaluation_span.name == "evaluation"
            assert executor_span.name == "executor"
            assert sorted([span.name for span in evaluator_spans]) == ["test", "test2"]


def test_evaluate_after_init(span_exporter: InMemorySpanExporter):
    """
    Test the evaluate function with mocked API calls.
    """
    mock_eval_id = "00000000-0000-0000-0000-000000000000"

    # Mock the init eval endpoint
    init_eval_response = MagicMock()
    init_eval_response.id = mock_eval_id
    init_eval_response.projectId = "mock-project-id"

    # Mock the datapoints endpoint
    datapoints_response = MagicMock()

    # Patch the AsyncLaminarClient._evals.init method
    with patch(
        "lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init",
        return_value=init_eval_response,
    ) as mock_init:
        # Patch the datapoints method
        with patch(
            "lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints",
            return_value=datapoints_response,
        ) as mock_datapoints:

            Laminar.initialize(
                project_api_key="test",
            )

            # Run the evaluate function
            evaluate(
                data=[
                    {
                        "data": "test",
                        "target": "test",
                    },
                ],
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
            mock_init.assert_called_once()
            assert mock_datapoints.call_count == 2  # Called once for each evaluator

            # Get the finished spans
            spans = span_exporter.get_finished_spans()

            # Verify the spans
            assert len(spans) == 4

            # Find the specific span types
            evaluation_span = next(
                (
                    span
                    for span in spans
                    if span.attributes.get("lmnr.span.type") == "EVALUATION"
                ),
                None,
            )
            executor_span = next(
                (
                    span
                    for span in spans
                    if span.attributes.get("lmnr.span.type") == "EXECUTOR"
                ),
                None,
            )
            evaluator_spans = [
                span
                for span in spans
                if span.attributes.get("lmnr.span.type") == "EVALUATOR"
            ]

            # Verify the span names
            assert evaluation_span.name == "evaluation"
            assert executor_span.name == "executor"
            assert sorted([span.name for span in evaluator_spans]) == ["test", "test2"]


def test_evaluate_with_flush(span_exporter: InMemorySpanExporter):
    """
    Test the evaluate function with mocked API calls.
    """
    mock_eval_id = "00000000-0000-0000-0000-000000000000"

    # Mock the init eval endpoint
    init_eval_response = MagicMock()
    init_eval_response.id = mock_eval_id
    init_eval_response.projectId = "mock-project-id"

    # Mock the datapoints endpoint
    datapoints_response = MagicMock()

    def mock_executor(data):
        Laminar.flush()
        return data

    # Patch the AsyncLaminarClient._evals.init method
    with patch(
        "lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init",
        return_value=init_eval_response,
    ) as mock_init:
        # Patch the datapoints method
        with patch(
            "lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints",
            return_value=datapoints_response,
        ) as mock_datapoints:

            # Run the evaluate function
            evaluate(
                data=[
                    {
                        "data": "test",
                        "target": "test",
                    },
                ],
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
            mock_init.assert_called_once()
            assert mock_datapoints.call_count == 2  # Called once for each evaluator

            # Get the finished spans
            spans = span_exporter.get_finished_spans()

            # Verify the spans
            assert len(spans) == 4

            # Find the specific span types
            evaluation_span = next(
                (
                    span
                    for span in spans
                    if span.attributes.get("lmnr.span.type") == "EVALUATION"
                ),
                None,
            )
            executor_span = next(
                (
                    span
                    for span in spans
                    if span.attributes.get("lmnr.span.type") == "EXECUTOR"
                ),
                None,
            )
            evaluator_spans = [
                span
                for span in spans
                if span.attributes.get("lmnr.span.type") == "EVALUATOR"
            ]

            # Verify the span names
            assert evaluation_span.name == "evaluation"
            assert executor_span.name == "executor"
            assert sorted([span.name for span in evaluator_spans]) == ["test", "test2"]


@pytest.mark.asyncio
async def test_evaluate_with_human_evaluator_async(span_exporter: InMemorySpanExporter):
    """
    Test the evaluate function with HumanEvaluator instances (async version).
    """
    mock_eval_id = "00000000-0000-0000-0000-000000000000"

    # Mock the init eval endpoint
    init_eval_response = MagicMock()
    init_eval_response.id = mock_eval_id
    init_eval_response.projectId = "mock-project-id"

    # Mock the datapoints endpoint
    datapoints_response = MagicMock()

    # Create HumanEvaluator instances
    human_evaluator1 = HumanEvaluator()
    human_evaluator2 = HumanEvaluator()

    # Patch the AsyncLaminarClient._evals.init method
    with patch(
        "lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init",
        return_value=init_eval_response,
    ) as mock_init:
        # Patch the datapoints method
        with patch(
            "lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints",
            return_value=datapoints_response,
        ) as mock_datapoints:

            # Run the evaluate function with mixed evaluators
            result = await evaluate(
                data=[
                    {
                        "data": "test",
                        "target": "test",
                    },
                ],
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
            mock_init.assert_called_once()
            assert (
                mock_datapoints.call_count == 2
            )  # Called once for partial, once for final

            # Verify return values - human evaluator scores should be None
            assert "accuracy" in result
            assert result["accuracy"] == 1  # Regular evaluator should have a score
            # Human evaluators should not contribute to average scores since they're None
            assert "human_quality" not in result
            assert "human_relevance" not in result

            # Get the finished spans
            spans = span_exporter.get_finished_spans()

            # Verify the spans - should have evaluation, executor, regular evaluator, and 2 human evaluator spans
            assert len(spans) == 5

            # Find the specific span types
            evaluation_span = next(
                (
                    span
                    for span in spans
                    if span.attributes.get("lmnr.span.type") == "EVALUATION"
                ),
                None,
            )
            executor_span = next(
                (
                    span
                    for span in spans
                    if span.attributes.get("lmnr.span.type") == "EXECUTOR"
                ),
                None,
            )
            evaluator_spans = [
                span
                for span in spans
                if span.attributes.get("lmnr.span.type") == "EVALUATOR"
            ]
            human_evaluator_spans = [
                span
                for span in spans
                if span.attributes.get("lmnr.span.type") == "HUMAN_EVALUATOR"
            ]

            # Verify the span names and types
            assert evaluation_span.name == "evaluation"
            assert executor_span.name == "executor"
            assert len(evaluator_spans) == 1
            assert evaluator_spans[0].name == "accuracy"
            assert len(human_evaluator_spans) == 2
            assert sorted([span.name for span in human_evaluator_spans]) == [
                "human_quality",
                "human_relevance",
            ]

            # Verify human evaluator spans have correct attributes
            for human_span in human_evaluator_spans:
                assert human_span.attributes.get("lmnr.span.type") == "HUMAN_EVALUATOR"


def test_evaluate_with_human_evaluator(span_exporter: InMemorySpanExporter):
    """
    Test the evaluate function with HumanEvaluator instances (sync version).
    """
    mock_eval_id = "00000000-0000-0000-0000-000000000000"

    # Mock the init eval endpoint
    init_eval_response = MagicMock()
    init_eval_response.id = mock_eval_id
    init_eval_response.projectId = "mock-project-id"

    # Mock the datapoints endpoint
    datapoints_response = MagicMock()

    # Create HumanEvaluator instance
    human_evaluator = HumanEvaluator()

    # Patch the AsyncLaminarClient._evals.init method
    with patch(
        "lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init",
        return_value=init_eval_response,
    ) as mock_init:
        # Patch the datapoints method
        with patch(
            "lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints",
            return_value=datapoints_response,
        ) as mock_datapoints:

            # Run the evaluate function with mixed evaluators
            result = evaluate(
                data=[
                    {
                        "data": "test",
                        "target": "test",
                    },
                ],
                executor=lambda data: data,
                evaluators={
                    "precision": lambda output, target: 0.9,
                    "human_evaluation": human_evaluator,
                },
                project_api_key="test",
            )

            # Flush the traces
            Laminar.flush()

            # Verify the API calls
            mock_init.assert_called_once()
            assert mock_datapoints.call_count == 2

            # Verify return values
            assert "precision" in result
            assert result["precision"] == 0.9
            # Human evaluator should not be in results since score is None
            assert "human_evaluation" not in result

            # Get the finished spans
            spans = span_exporter.get_finished_spans()

            # Verify the spans
            assert len(spans) == 4  # evaluation, executor, evaluator, human_evaluator

            # Find the human evaluator span
            human_evaluator_span = next(
                (
                    span
                    for span in spans
                    if span.attributes.get("lmnr.span.type") == "HUMAN_EVALUATOR"
                ),
                None,
            )

            assert human_evaluator_span is not None
            assert human_evaluator_span.name == "human_evaluation"
            assert (
                human_evaluator_span.attributes.get("lmnr.span.type")
                == "HUMAN_EVALUATOR"
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

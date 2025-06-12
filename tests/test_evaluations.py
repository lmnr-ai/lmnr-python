import pytest
from unittest.mock import patch, MagicMock

from lmnr import Laminar
from lmnr.sdk.evaluations import evaluate
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.asyncio
async def test_evaluate_with_mocks_async(exporter: InMemorySpanExporter):
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
            spans = exporter.get_finished_spans()

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


def test_evaluate_with_mocks(exporter: InMemorySpanExporter):
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
            spans = exporter.get_finished_spans()

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


def test_evaluate_after_init(exporter: InMemorySpanExporter):
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
            spans = exporter.get_finished_spans()

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


def test_evaluate_with_flush(exporter: InMemorySpanExporter):
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
            spans = exporter.get_finished_spans()

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

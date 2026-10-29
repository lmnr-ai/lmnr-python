import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typing_extensions import TypeVar

from lmnr import Laminar
from lmnr.sdk.datasets import LaminarDataset
from lmnr.sdk.evaluations import evaluate
from lmnr.sdk.evaluations.evaluation import _TaskIdResolver
from lmnr.sdk.evaluations.models import EvaluationResultDatapoint, EvaluationRunResult
from lmnr.sdk.types import Datapoint, Dataset

T = TypeVar("T")


# Fixtures for common mock objects
@pytest.fixture
def mock_eval_response() -> dict[str, str]:
    """Create a mock evaluation response."""
    return {
        "id": "00000000-0000-0000-0000-000000000000",
        "projectId": "mock-project-id",
    }


@pytest.fixture
def mock_datapoints_response() -> MagicMock:
    """Create a mock datapoints response."""
    return MagicMock()


@pytest.fixture
def mock_dataset_push_response() -> dict[str, str]:
    """Create a mock dataset push response."""
    return {"dataset_id": "00000000-0000-0000-0000-000000000001"}


@pytest.fixture
def mock_dataset_pull_response() -> dict[str, list[Datapoint] | int]:
    """Create a mock dataset pull response."""
    return {
        "items": [
            Datapoint(
                id=uuid.uuid4(),
                data="test",
                target="test",
                createdAt=datetime.now(tz=UTC),
            )
        ],
        "total_count": 1,
    }


def paged_pull(total: int, fetch_size: int) -> Callable[..., dict[str, list[Datapoint] | int]]:
    """Build a `Datasets.pull` side effect that serves one page per call."""
    all_items = [
        Datapoint(
            id=uuid.uuid4(),
            data=f"test-{i}",
            target=f"test-{i}",
            createdAt=datetime.now(tz=UTC),
        )
        for i in range(total)
    ]

    def pull(
        name: str | None = None,
        id: uuid.UUID | None = None,
        offset: int = 0,
        limit: int = fetch_size,
    ) -> dict[str, list[Datapoint] | int]:
        return {
            "items": all_items[offset : offset + limit],
            "total_count": total,
        }

    return pull


# Helper functions for common test logic
def verify_basic_api_calls(
    mock_init: MagicMock,
    mock_dataset_push: MagicMock,
    mock_dataset_pull: MagicMock,
    mock_datapoints: MagicMock,
):
    """Verify the expected API calls were made."""
    mock_init.assert_called_once()
    # TODO: add tests with Laminar dataset and verify pull
    assert mock_datapoints.call_count == 2


def verify_basic_spans(
    spans: Sequence[ReadableSpan],
    expected_evaluator_names: list[str],
) -> tuple[ReadableSpan, ReadableSpan, list[ReadableSpan]]:
    """Verify the basic span structure and return categorized spans."""
    evaluation_span = next(
        (span for span in spans if (span.attributes or {}).get("lmnr.span.type") == "EVALUATION"),
        None,
    )
    executor_span = next(
        (span for span in spans if (span.attributes or {}).get("lmnr.span.type") == "EXECUTOR"),
        None,
    )
    evaluator_spans = [
        span for span in spans if (span.attributes or {}).get("lmnr.span.type") == "EVALUATOR"
    ]

    assert evaluation_span is not None
    assert executor_span is not None
    assert evaluation_span.name == "evaluation"
    assert executor_span.name == "executor"
    assert sorted([span.name for span in evaluator_spans]) == sorted(expected_evaluator_names)

    return evaluation_span, executor_span, evaluator_spans


@pytest.mark.asyncio
@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.datasets.AsyncDatasets.push")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
async def test_evaluate_with_mocks_async(
    mock_init: MagicMock,
    mock_datapoints: MagicMock,
    mock_dataset_push: MagicMock,
    mock_dataset_pull: MagicMock,
    mock_eval_response: MagicMock,
    mock_datapoints_response: MagicMock,
    mock_dataset_push_response: MagicMock,
    mock_dataset_pull_response: MagicMock,
    span_exporter: InMemorySpanExporter,
):
    """Test the evaluate function with mocked API calls (async)."""
    # Set up mock return values
    mock_init.return_value = mock_eval_response
    mock_datapoints.return_value = mock_datapoints_response
    mock_dataset_push.return_value = mock_dataset_push_response
    mock_dataset_pull.return_value = mock_dataset_pull_response

    # Run the evaluate function
    _result = await cast(
        Awaitable[EvaluationRunResult],
        evaluate(
            data=[{"data": "test", "target": "test"}],
            executor=lambda data, _t=None: data,
            evaluators={
                "test": lambda output, target: 1 if output == target else 0,
                "test2": lambda output, target: 1 if output == target else 0,
            },
            project_api_key="test",
        ),
    )

    # Flush the traces
    _flush_success = Laminar.flush()

    # Verify the API calls
    verify_basic_api_calls(mock_init, mock_dataset_push, mock_dataset_pull, mock_datapoints)

    # Get the finished spans and verify
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    _grouped = verify_basic_spans(spans, ["test", "test2"])


@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.datasets.AsyncDatasets.push")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
def test_evaluate_with_mocks(
    mock_init: MagicMock,
    mock_datapoints: MagicMock,
    mock_dataset_push: MagicMock,
    mock_dataset_pull: MagicMock,
    mock_eval_response: MagicMock,
    mock_datapoints_response: MagicMock,
    mock_dataset_push_response: MagicMock,
    mock_dataset_pull_response: MagicMock,
    span_exporter: InMemorySpanExporter,
):
    """Test the evaluate function with mocked API calls (sync)."""
    # Set up mock return values
    mock_init.return_value = mock_eval_response
    mock_datapoints.return_value = mock_datapoints_response
    mock_dataset_push.return_value = mock_dataset_push_response
    mock_dataset_pull.return_value = mock_dataset_pull_response

    # Run the evaluate function
    _result = cast(
        EvaluationRunResult,
        evaluate(
            data=[{"data": "test", "target": "test"}],
            executor=lambda data, _t=None: data,
            evaluators={
                "test": lambda output, target: 1 if output == target else 0,
                "test2": lambda output, target: 1 if output == target else 0,
            },
            project_api_key="test",
        ),
    )

    # Flush the traces
    _flush_success = Laminar.flush()

    # Verify the API calls
    verify_basic_api_calls(mock_init, mock_dataset_push, mock_dataset_pull, mock_datapoints)

    # Get the finished spans and verify
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    _grouped = verify_basic_spans(spans, ["test", "test2"])


@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.datasets.AsyncDatasets.push")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
def test_evaluate_after_init(
    mock_init: MagicMock,
    mock_datapoints: MagicMock,
    mock_dataset_push: MagicMock,
    mock_dataset_pull: MagicMock,
    mock_eval_response: MagicMock,
    mock_datapoints_response: MagicMock,
    mock_dataset_push_response: MagicMock,
    mock_dataset_pull_response: MagicMock,
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
    _result = cast(
        EvaluationRunResult,
        evaluate(
            data=[{"data": "test", "target": "test"}],
            executor=lambda data, _t=None: data,
            evaluators={
                "test": lambda output, target: 1 if output == target else 0,
                "test2": lambda output, target: 1 if output == target else 0,
            },
            project_api_key="test",
        ),
    )

    # Flush the traces
    _flush_success = Laminar.flush()

    # Verify the API calls
    verify_basic_api_calls(mock_init, mock_dataset_push, mock_dataset_pull, mock_datapoints)

    # Get the finished spans and verify
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    _grouped = verify_basic_spans(spans, ["test", "test2"])


@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.datasets.AsyncDatasets.push")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
def test_evaluate_with_flush(
    mock_init: MagicMock,
    mock_datapoints: MagicMock,
    mock_dataset_push: MagicMock,
    mock_dataset_pull: MagicMock,
    mock_eval_response: MagicMock,
    mock_datapoints_response: MagicMock,
    mock_dataset_push_response: MagicMock,
    mock_dataset_pull_response: MagicMock,
    span_exporter: InMemorySpanExporter,
):
    """Test the evaluate function when flush is called within the executor."""
    # Set up mock return values
    mock_init.return_value = mock_eval_response
    mock_datapoints.return_value = mock_datapoints_response
    mock_dataset_push.return_value = mock_dataset_push_response
    mock_dataset_pull.return_value = mock_dataset_pull_response

    def mock_executor(data: T, _target: Any = None) -> T:
        _flush_success = Laminar.flush()
        return data

    # Run the evaluate function
    _result = cast(
        EvaluationRunResult,
        evaluate(
            data=[{"data": "test", "target": "test"}],
            executor=mock_executor,
            evaluators={
                "test": lambda output, target: 1 if output == target else 0,
                "test2": lambda output, target: 1 if output == target else 0,
            },
            project_api_key="test",
        ),
    )

    # Flush the traces
    _flush_success = Laminar.flush()

    # Verify the API calls
    verify_basic_api_calls(mock_init, mock_dataset_push, mock_dataset_pull, mock_datapoints)

    # Get the finished spans and verify
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    _grouped = verify_basic_spans(spans, ["test", "test2"])


@pytest.mark.asyncio
@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.datasets.AsyncDatasets.push")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
async def test_evaluate_propagates_evaluation_id_to_all_spans(
    mock_init: MagicMock,
    mock_datapoints: MagicMock,
    mock_dataset_push: MagicMock,
    mock_dataset_pull: MagicMock,
    mock_eval_response: MagicMock,
    mock_datapoints_response: MagicMock,
    mock_dataset_push_response: MagicMock,
    mock_dataset_pull_response: MagicMock,
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

    def executor(data: T, _target: Any = None) -> T:
        with Laminar.start_as_current_span("inner_user_span"):
            return data

    _result = await cast(
        Awaitable[EvaluationRunResult],
        evaluate(
            data=[
                {"data": "a", "target": "a"},
                {"data": "b", "target": "b"},
            ],
            executor=executor,
            evaluators={
                "exact": lambda output, target: 1 if output == target else 0,
            },
            project_api_key="test",
        ),
    )

    _flush_success = Laminar.flush()

    spans = span_exporter.get_finished_spans()
    # 2 datapoints * (evaluation + executor + inner_user_span + 1 evaluator) = 8
    assert len(spans) == 8

    expected_eval_id = cast(uuid.UUID, mock_eval_response["id"])
    for span in spans:
        actual = (span.attributes or {}).get("lmnr.association.properties.metadata.evaluation_id")
        assert actual == expected_eval_id, (
            f"span {span.name} missing/incorrect evaluation_id: got {actual!r}, expected {expected_eval_id!r}"
        )

    trace_ids: set[int] = set()
    for span in spans:
        context = span.get_span_context()
        if context is not None:
            trace_ids.add(context.trace_id)
    assert len(trace_ids) == 2


@pytest.mark.asyncio
@patch("lmnr.sdk.client.asynchronous.resources.datasets.AsyncDatasets.get_dataset_by_name")
@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
async def test_evaluate_streams_laminar_dataset_in_pages(
    mock_init: MagicMock,
    mock_datapoints: MagicMock,
    mock_dataset_pull: MagicMock,
    mock_get_dataset_by_name: MagicMock,
    mock_eval_response: MagicMock,
    mock_datapoints_response: MagicMock,
    span_exporter: InMemorySpanExporter,
):
    """A LaminarDataset is pulled one page at a time while datapoints run, and
    streamed pages are not retained on the dataset."""
    mock_init.return_value = mock_eval_response
    mock_datapoints.return_value = mock_datapoints_response
    mock_dataset_pull.side_effect = paged_pull(total=7, fetch_size=3)
    mock_get_dataset_by_name.return_value = [
        Dataset(id=uuid.uuid4(), name="test_dataset", created_at=datetime.now(tz=UTC))
    ]

    executed: list[Any] = []

    def executor(data: T, _target: Any = None) -> T:
        executed.append(data)
        return data

    dataset = LaminarDataset(name="test_dataset", fetch_size=3)

    result = await cast(
        Awaitable[EvaluationRunResult],
        evaluate(
            data=dataset,
            executor=executor,
            evaluators={
                "exact": lambda output, target: 1 if output == target else 0,
            },
            project_api_key="test",
        ),
    )
    _flush_success = Laminar.flush()

    assert result["average_scores"]["exact"] == 1
    assert [str(datapoint) for datapoint in executed] == [f"test-{i}" for i in range(7)]
    # the reporter's len() caches page 0, then pages 1 and 2 are streamed
    assert mock_dataset_pull.call_count == 3
    # streaming must not accumulate pages beyond the one len() cached
    assert list(dataset._pages) == [0]
    # 7 datapoints * (evaluation + executor + evaluator) = 21 spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 21


@pytest.mark.asyncio
@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
async def test_evaluate_runs_trials_concurrently(
    mock_init: MagicMock,
    mock_datapoints: MagicMock,
    mock_dataset_pull: MagicMock,
    mock_eval_response: MagicMock,
    mock_datapoints_response: MagicMock,
    mock_dataset_pull_response: MagicMock,
    span_exporter: InMemorySpanExporter,
):
    """The semaphore sliding window keeps up to `concurrency_limit` Trials
    in flight at once."""
    mock_init.return_value = mock_eval_response
    mock_datapoints.return_value = mock_datapoints_response
    mock_dataset_pull.return_value = mock_dataset_pull_response

    in_flight = 0
    max_in_flight = 0

    async def executor(data: T, _target: Any = None) -> T:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return data

    result = await cast(
        Awaitable[EvaluationRunResult],
        evaluate(
            data=[{"data": f"test-{i}", "target": f"test-{i}"} for i in range(6)],
            executor=executor,
            evaluators={
                "exact": lambda output, target: 1 if output == target else 0,
            },
            concurrency_limit=3,
            project_api_key="test",
        ),
    )
    _flush_success = Laminar.flush()

    assert result["average_scores"]["exact"] == 1
    assert max_in_flight == 3


@pytest.mark.asyncio
@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
async def test_evaluate_trial_failure_does_not_abort_remaining(
    mock_init: MagicMock,
    mock_datapoints: MagicMock,
    mock_dataset_pull: MagicMock,
    mock_eval_response: MagicMock,
    mock_datapoints_response: MagicMock,
    mock_dataset_pull_response: MagicMock,
    span_exporter: InMemorySpanExporter,
):
    """A failing Trial must not stop the remaining Trials from being
    scheduled and run; its error still propagates once all have finished."""
    mock_init.return_value = mock_eval_response
    mock_datapoints.return_value = mock_datapoints_response
    mock_dataset_pull.return_value = mock_dataset_pull_response

    executed: list[Any] = []

    async def executor(data: T, _target: Any = None) -> T:
        executed.append(data)
        if data == "test-0":
            raise RuntimeError("executor failed")
        return data

    with pytest.raises(RuntimeError, match="executor failed"):
        _result = await cast(
            Awaitable[EvaluationRunResult],
            evaluate(
                data=[{"data": f"test-{i}", "target": f"test-{i}"} for i in range(6)],
                executor=executor,
                evaluators={
                    "exact": lambda output, target: 1 if output == target else 0,
                },
                concurrency_limit=1,
                project_api_key="test",
            ),
        )
    _flush_success = Laminar.flush()

    # With concurrency_limit=1 the failure on the first Trial completes
    # before the second is scheduled — every Trial must still run.
    assert executed == [f"test-{i}" for i in range(6)]


@pytest.mark.asyncio
@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
async def test_evaluate_upload_failure_does_not_skip_remaining_saves(
    mock_init: MagicMock,
    mock_datapoints: MagicMock,
    mock_dataset_pull: MagicMock,
    mock_eval_response: MagicMock,
    mock_datapoints_response: MagicMock,
    mock_dataset_pull_response: MagicMock,
):
    """One failing background upload must not abandon the result saves of the
    Trials after it; the upload error still propagates once they finish."""
    mock_init.return_value = mock_eval_response
    mock_dataset_pull.return_value = mock_dataset_pull_response

    started: list[int] = []
    completed: list[int] = []

    async def save_datapoints(
        _eval_id: uuid.UUID,
        datapoints: Sequence[Any],
        _group_name: str | None = None,
    ) -> MagicMock:
        datapoint = datapoints[0]
        if not isinstance(datapoint, EvaluationResultDatapoint):
            # The partial save that makes the Trial visible in the UI.
            return mock_datapoints_response
        started.append(datapoint.index)
        if datapoint.index == 0:
            raise ValueError("upload boom")
        # Keep the save in flight past the end of the Trial loop, so a
        # result that is only persisted because the run waits for it is
        # distinguishable from one that happened to land early.
        await asyncio.sleep(0.1)
        completed.append(datapoint.index)
        return mock_datapoints_response

    mock_datapoints.side_effect = save_datapoints

    async def executor(data: T, _target: Any = None) -> T:
        return data

    with pytest.raises(ValueError, match="upload boom"):
        _result = await cast(
            Awaitable[EvaluationRunResult],
            evaluate(
                data=[{"data": f"test-{i}", "target": f"test-{i}"} for i in range(6)],
                executor=executor,
                evaluators={
                    "exact": lambda output, target: 1 if output == target else 0,
                },
                concurrency_limit=1,
                project_api_key="test",
            ),
        )
    _flush_success = Laminar.flush()

    assert sorted(started) == list(range(6))
    # Every save after the failing one must have completed before `evaluate`
    # gave up and closed the client.
    assert sorted(completed) == list(range(1, 6))


def saved_results(mock_save: MagicMock) -> list[EvaluationResultDatapoint]:
    return [
        dp
        for call in mock_save.call_args_list
        for dp in call.args[1]
        if isinstance(dp, EvaluationResultDatapoint)
    ]


@pytest.mark.asyncio
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
async def test_evaluate_trials_share_task_id(
    mock_init: MagicMock,
    mock_save: MagicMock,
    mock_eval_response: MagicMock,
):
    """Each Task runs `trials` times; Trials get distinct ids but one task id."""
    mock_init.return_value = mock_eval_response

    _result = await cast(
        Awaitable[EvaluationRunResult],
        evaluate(
            data=[{"data": "a", "taskId": "alpha"}, {"data": "b"}],
            executor=lambda d, _t=None: d,
            evaluators={"one": lambda output, target: 1},
            trials=3,
            project_api_key="test",
        ),
    )
    _flush_success = Laminar.flush()

    results = saved_results(mock_save)
    assert len(results) == 6
    assert len({r.id for r in results}) == 6
    assert sorted(r.task_id or "" for r in results) == ["1"] * 3 + ["alpha"] * 3
    assert all(r.to_dict()["taskId"] == r.task_id for r in results)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_evaluate_rejects_invalid_trials(bad: Any):
    with pytest.raises(ValueError, match="trials"):
        _result = evaluate(
            data=[{"data": "a"}],
            executor=lambda d, _t=None: d,
            evaluators={"one": lambda output, target: 1},
            trials=bad,
            project_api_key="test",
        )


@pytest.mark.asyncio
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
async def test_evaluate_renames_duplicate_task_ids(
    mock_init: MagicMock,
    mock_save: MagicMock,
    mock_eval_response: MagicMock,
):
    """Different datapoints resolving to one task id are renamed `id#2`, ...,
    including a default (index) id colliding with an explicit one."""
    mock_init.return_value = mock_eval_response

    _result = await cast(
        Awaitable[EvaluationRunResult],
        evaluate(
            data=[
                {"data": 0, "task_id": "dup"},
                {"data": 1, "taskId": "dup"},
                {"data": 2, "taskId": "dup"},
                {"data": 3, "taskId": "4"},
                {"data": 4},
            ],
            executor=lambda d, _t=None: d,
            evaluators={"one": lambda output, target: 1},
            concurrency_limit=1,
            project_api_key="test",
        ),
    )
    _flush_success = Laminar.flush()

    by_data = {r.data: r.task_id for r in saved_results(mock_save)}
    assert by_data == {0: "dup", 1: "dup#2", 2: "dup#3", 3: "4", 4: "4#2"}


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (["foo", "foo", "foo#2", "foo"], ["foo", "foo#2", "foo#2#2", "foo#3"]),
        (["foo", "foo#2", "foo"], ["foo", "foo#2", "foo#3"]),
    ],
)
def test_task_id_resolver_skips_used_ids(requested: list[str], expected: list[str]):
    """A duplicate of base `b` gets the smallest unused `b#n`; generated ids
    count as used, so a later explicit id equal to one is renamed too."""
    resolver = _TaskIdResolver(MagicMock())
    resolved = [
        resolver.resolve(Datapoint(data=i, task_id=task_id), i)
        for i, task_id in enumerate(requested)
    ]
    assert resolved == expected


@pytest.mark.asyncio
@patch("lmnr.sdk.client.synchronous.resources.datasets.Datasets.pull")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.save_datapoints")
@patch("lmnr.sdk.client.asynchronous.resources.evals.AsyncEvals.init")
async def test_evaluate_dataset_link_requires_datapoint_id_and_created_at(
    mock_init: MagicMock,
    mock_save: MagicMock,
    mock_pull: MagicMock,
    mock_eval_response: MagicMock,
):
    mock_init.return_value = mock_eval_response
    linked = Datapoint(id=uuid.uuid4(), data="linked", createdAt=datetime.now(tz=UTC))
    mock_pull.return_value = {
        "items": [linked, Datapoint(data="no-id"), Datapoint(id=uuid.uuid4(), data="no-ts")],
        "total_count": 3,
    }

    _result = await cast(
        Awaitable[EvaluationRunResult],
        evaluate(
            data=LaminarDataset(id=uuid.uuid4()),
            executor=lambda d, _t=None: d,
            evaluators={"one": lambda output, target: 1},
            project_api_key="test",
        ),
    )
    _flush_success = Laminar.flush()

    links = {r.data: r.dataset_link for r in saved_results(mock_save)}
    assert links["linked"] is not None and links["linked"]["datapoint_id"] == linked.id
    assert links["no-id"] is None
    assert links["no-ts"] is None

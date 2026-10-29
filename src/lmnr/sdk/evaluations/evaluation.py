import asyncio
import re
import uuid
from asyncio.tasks import Task
from collections.abc import Awaitable, Callable
from logging import Logger
from typing import Any, cast

from lmnr.opentelemetry_lib.tracing.attributes import SPAN_TYPE
from lmnr.opentelemetry_lib.tracing.context import (
    attach_context,
    detach_context,
    set_association_prop_context,
)
from lmnr.opentelemetry_lib.tracing.instruments import Instruments
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.datasets import EvaluationDataset, LaminarDataset
from lmnr.sdk.evaluations.models import (
    DEFAULT_BATCH_SIZE,
    MAX_EXPORT_BATCH_SIZE,
    EvaluationDatapointDatasetLink,
    EvaluationResultDatapoint,
    EvaluationRunResult,
    EvaluatorFunction,
    EvaluatorFunctionReturnType,
    PartialEvaluationDatapoint,
)
from lmnr.sdk.evaluations.reporter import EvaluationReporter
from lmnr.sdk.evaluations.utils import get_evaluation_url
from lmnr.sdk.laminar import Laminar as L
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import (
    Datapoint,
    Numeric,
    NumericTypes,
    SpanType,
    TraceType,
)
from lmnr.sdk.utils import from_env, is_async

# Evaluation-metadata key that links an eval run to its debug session. Kept as
# `rollout.session_id` (matching the trace-metadata key the debugger stamps) so
# the eval and its debug session cross-reference under one identifier.
SESSION_METADATA_KEY = "rollout.session_id"


def _with_debugger_session_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Stamp the debug session id into eval metadata when running under debug.

    When this eval runs under a debug session, auto-stamp the session id into
    the evaluation metadata so the created evaluation links back to it with no
    extra step. The session id is resolved by the debug runtime EXACTLY like
    traces — `LMNR_DEBUG_SESSION_ID` env → `.lmnr/debug-session.json` → freshly
    minted — so a plain `LMNR_DEBUG=1 <run-your-eval>` groups the eval under the
    current debug session (no CLI wrapper needed). `get_runtime()` is None when
    debug mode is off, so the metadata is returned unchanged.

    The backend writes the `evaluation` block from this key at eval creation;
    notes are attached separately as `text` blocks keyed by the same session id
    (`lmnr-cli debug session add-note`). Called after `Laminar.initialize()`, so
    the runtime (and its resolved session id) already exist.
    """
    from lmnr.sdk.debug import get_runtime

    runtime = get_runtime()
    session_id = runtime.session_id if runtime is not None else None
    if session_id is None:
        return metadata
    return {**(metadata or {}), SESSION_METADATA_KEY: session_id}


class _TaskIdResolver:
    """Assigns each datapoint its task id for one eval run.

    The task id is the datapoint's own `task_id` if set (explicit, or pulled
    from a dataset), else its position in the source data. Two DIFFERENT
    source datapoints resolving to the same id would merge two Tasks, so later
    ones are renamed `id#2`, `id#3`, ...; the same source datapoint seen again
    (e.g. `select([0, 0])`) keeps its id, since those are Trials of one Task.
    """

    def __init__(self, logger: Logger):
        self._by_source_index: dict[int, str] = {}
        self._claimed: set[str] = set()
        self._warned: bool = False
        self._logger: Logger = logger

    def resolve(self, datapoint: Datapoint, source_index: int) -> str:
        existing = self._by_source_index.get(source_index)
        if existing is not None:
            return existing
        base = datapoint.task_id or str(source_index)
        task_id = base
        suffix = 2
        while task_id in self._claimed:
            task_id = f"{base}#{suffix}"
            suffix += 1
        if task_id != base and not self._warned:
            self._warned = True
            self._logger.warning(
                f'Duplicate task id "{base}" on different datapoints; renamed to '
                + f'"{task_id}". Further duplicates are renamed the same way.'
            )
        self._claimed.add(task_id)
        self._by_source_index[source_index] = task_id
        return task_id


class Evaluation:
    def __init__(
        self,
        data: EvaluationDataset | list[Datapoint | dict[Any, Any]],
        executor: Callable[..., Any],
        evaluators: dict[str, EvaluatorFunction],
        name: str | None = None,
        group_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        concurrency_limit: int = DEFAULT_BATCH_SIZE,
        project_api_key: str | None = None,
        base_url: str | None = None,
        base_http_url: str | None = None,
        http_port: int | None = None,
        grpc_port: int | None = None,
        frontend_port: int | None = None,
        instruments: (set[Instruments] | list[Instruments] | tuple[Instruments] | None) = None,
        disabled_instruments: (
            set[Instruments] | list[Instruments] | tuple[Instruments] | None
        ) = None,
        max_export_batch_size: int | None = MAX_EXPORT_BATCH_SIZE,
        trace_export_timeout_seconds: int | None = None,
        trials: int = 1,
    ):
        """
        Initializes an instance of the Evaluation class.

        Parameters:
            data (list[Datapoint|dict] | EvaluationDataset):\
                List of data points to evaluate or an evaluation dataset.
                    `data` is the input to the executor function.
                    `target` is the input to the evaluator function.
                    `metadata` is optional metadata to associate with the\
                        datapoint.
            executor (Callable[..., Any]): The executor function.\
                    Takes the data point + any additional arguments and returns\
                    the output to evaluate.
            evaluators (dict[str, Callable[..., Any] | HumanEvaluator]): Evaluator\
                functions and HumanEvaluator instances with names. Each evaluator\
                function takes the output of the executor _and_ the target data,\
                and returns a score. The score can be a single number or a dict\
                of string keys and number values. If the score is a single number,\
                it will be named after the evaluator function.\
                HumanEvaluator instances create empty spans for manual evaluation.\
                Evaluator names must contain only letters, digits, hyphens,\
                underscores, or spaces.
            name (str | None, optional): Optional name of the evaluation.\
                Used to identify the evaluation in the group.\
                If not provided, a random name will be generated.
                Defaults to None.
            group_name (str | None, optional): an identifier to group\
                evaluations. Only evaluations within the same group_name can be\
                visually compared. If not provided, "default" is assigned.
                Defaults to None
            metadata (dict[str, Any] | None): optional metadata to associate with\
            concurrency_limit (int, optional): The concurrency limit for\
                evaluation. This many data points will be evaluated in parallel\
                with a pool of workers.
                Defaults to DEFAULT_BATCH_SIZE.
            project_api_key (str | None, optional): The project API key.\
                If not provided, LMNR_PROJECT_API_KEY environment variable is\
                used.
                Defaults to an empty string.
            base_url (str | None, optional): The base URL for Laminar API.\
                Useful if self-hosted. Do NOT include the port, use `http_port`\
                and `grpc_port` instead.
                Defaults to "https://api.lmnr.ai".
            base_http_url (str | None, optional): The base HTTP URL for Laminar API.\
                Only set this if your Laminar backend HTTP is proxied\
                through a different host. If not specified, defaults\
                to https://api.lmnr.ai.
            http_port (int | None, optional): The port for Laminar API\
                HTTP service. Defaults to 443 if not specified.
            grpc_port (int | None, optional): The port for Laminar API\
                gRPC service. Defaults to 8443 if not specified.
            frontend_port (int | None, optional): The port for the Laminar frontend.\
                Defaults to 5667 if not specified.
            instruments (set[Instruments] | None, optional): Set of modules\
                to auto-instrument. If None, all available instruments will be\
                used.
                See https://docs.lmnr.ai/tracing/automatic-instrumentation
                Defaults to None.
            disabled_instruments (set[Instruments] | None, optional): Set of modules\
                to disable auto-instrumentations. If None, only modules passed\
                as `instruments` will be disabled.
                Defaults to None.
            trials (int, optional): How many times to run each Task; each run is a Trial.\
                Each run (Trial) gets its own trace and result row; Trials of\
                one Task share its task id.
                Defaults to 1.
        """

        if not evaluators:
            raise ValueError("No evaluators provided")
        if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
            raise ValueError(f"trials must be an integer >= 1, got {trials!r}")

        evaluator_name_regex = re.compile(r"^[\w\s-]+$")
        for evaluator_name in evaluators:
            if not evaluator_name_regex.match(evaluator_name):
                raise ValueError(
                    f'Invalid evaluator key: "{evaluator_name}". '
                    + "Keys must only contain letters, digits, hyphens,"
                    + "underscores, or spaces."
                )

        base_url = base_url or from_env("LMNR_BASE_URL") or "https://api.lmnr.ai"

        self.reporter: EvaluationReporter = EvaluationReporter(base_url, frontend_port)
        if isinstance(data, list):
            self.data: list[Datapoint] | EvaluationDataset = [
                (Datapoint.model_validate(point) if isinstance(point, dict) else point)
                for point in data
            ]
        else:
            self.data = data
        # Only eagerly-materialized lists can be length-checked here; any
        # EvaluationDataset (remote or subsampled) is resolved lazily in _run.
        if isinstance(self.data, list) and len(self.data) == 0:
            raise ValueError("No data provided. Skipping evaluation")
        # The underlying remote source (through any depth of subsampling
        # chaining), resolved in _run. None for in-memory data.
        self._source_dataset: LaminarDataset | None = None
        self.executor: Callable[..., Any] = executor
        self.evaluators: dict[str, EvaluatorFunction] = evaluators
        self.group_name: str | None = group_name
        self.name: str | None = name
        self.metadata: dict[str, Any] | None = metadata
        self.concurrency_limit: int = concurrency_limit
        self.trials: int = trials
        self.batch_size: int = concurrency_limit
        self._logger: Logger = get_default_logger(self.__class__.__name__)
        self.upload_tasks: list[Task[None]] = []
        self._upload_error: BaseException | None = None
        self.base_http_url: str = f"{base_http_url or base_url}:{http_port or 443}"

        api_key = project_api_key or from_env("LMNR_PROJECT_API_KEY")
        if not api_key and not L.is_initialized():
            raise ValueError(
                "Please pass the project API key to `evaluate`"
                + " or set the LMNR_PROJECT_API_KEY environment variable"
                + " in your environment or .env file"
            )
        self.project_api_key: str | None = api_key

        if L.is_initialized():
            self.client: AsyncLaminarClient = AsyncLaminarClient(
                base_url=L.get_base_http_url(),
                project_api_key=L.get_project_api_key(),
            )
        else:
            self.client = AsyncLaminarClient(
                base_url=self.base_http_url,
                project_api_key=self.project_api_key,
            )

        L.initialize(
            project_api_key=project_api_key,
            base_url=base_url,
            base_http_url=self.base_http_url,
            http_port=http_port,
            grpc_port=grpc_port,
            instruments=instruments,
            disabled_instruments=disabled_instruments,
            max_export_batch_size=max_export_batch_size,
            export_timeout_seconds=trace_export_timeout_seconds,
        )

    async def run(self) -> EvaluationRunResult:
        return await self._run()

    async def _run(self) -> EvaluationRunResult:
        source = None
        if isinstance(self.data, EvaluationDataset):
            source = self.data.source_dataset()
            if source is not None:
                # source_dataset() already resolved the remote source through
                # any subsampling wrappers, so inject the client into it
                # directly.
                source.set_client(
                    LaminarClient(
                        base_url=self.base_http_url,
                        project_api_key=self.project_api_key,
                    )
                )
                if not source.id:
                    try:
                        datasets = await self.client.datasets.get_dataset_by_name(source.name or "")
                        if len(datasets) == 0:
                            self._logger.warning(f"Dataset {source.name} not found")
                        else:
                            source.id = datasets[0]["id"]
                    except Exception:
                        # Backward compatibility with old Laminar API (self hosted)
                        self._logger.warning(
                            f"Error getting dataset {source.name}",
                            exc_info=True,
                        )
        self._source_dataset = source

        try:
            evaluation = await self.client.evals.init(
                name=self.name,
                group_name=self.group_name,
                metadata=_with_debugger_session_metadata(self.metadata),
            )
            evaluation_id = evaluation["id"]
            project_id = evaluation["projectId"]
            url = get_evaluation_url(project_id, evaluation_id, self.reporter.base_url)

            print(f"Check the results at {url}")

            self.reporter.start(len(self.data) * self.trials)
            average_scores = await self._evaluate_in_batches(evaluation["id"])
            # Wait for all background upload tasks to complete. Failures are
            # collected instead of raised eagerly: a plain `gather` returns as
            # soon as one task fails, leaving the other saves unfinished when
            # `_shutdown` closes the client below.
            if self.upload_tasks:
                self._logger.debug(f"Waiting for {len(self.upload_tasks)} upload tasks to complete")
                gathered_results = await asyncio.gather(*self.upload_tasks, return_exceptions=True)
                for gathered in gathered_results:
                    if isinstance(gathered, BaseException) and self._upload_error is None:
                        self._upload_error = gathered
                self._logger.debug("All upload tasks completed")
            if self._upload_error is not None:
                raise self._upload_error
        except Exception as e:
            await self._shutdown()
            self.reporter.stop_with_error(e)

        self.reporter.stop(average_scores, evaluation["projectId"], evaluation["id"])
        await self._shutdown()
        return {
            "average_scores": average_scores,
            "evaluation_id": evaluation_id,
            "project_id": project_id,
            "url": url,
            "error_message": None,
        }

    async def _shutdown(self):
        # We use flush() instead of shutdown() because multiple evaluations
        # can be run sequentially in the same process. `shutdown()` would
        # close the OTLP exporter and we wouldn't be able to export traces in
        # the next evaluation.
        _flush_success = L.flush()
        await self.client.close()
        source = self._source_dataset
        if source is not None and getattr(source, "client", None):
            source.client.close()

    def _prune_upload_tasks(self) -> None:
        # Drop references to finished background saves so their payloads are
        # not retained for the whole run. This is a memory measure only: a
        # failed save is latched on `self._upload_error` and re-raised by
        # `_run` *after* it has drained the remaining upload tasks. Folding it
        # into the Trial loop's `first_error` instead would abort
        # `_evaluate_in_batches` before that drain, so every still-pending save
        # would be abandoned when `_shutdown` closes the client.
        for task in self.upload_tasks:
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc is not None and self._upload_error is None:
                    self._upload_error = exc
        self.upload_tasks = [task for task in self.upload_tasks if not task.done()]

    async def _evaluate_in_batches(self, eval_id: uuid.UUID) -> dict[str, Numeric]:
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        tasks: list[Task[None]] = []
        # Aggregate scores incrementally so Trial results (with their full
        # data/target/output payloads) are not retained for the whole run.
        score_sums: dict[str, Numeric] = {}
        score_counts: dict[str, int] = {}

        def accumulate_scores(result: EvaluationResultDatapoint) -> None:
            for key, value in result.scores.items():
                if value is None:
                    continue
                score_sums[key] = score_sums.get(key, 0) + value
                score_counts[key] = score_counts.get(key, 0) + 1

        async def run_trial(datapoint: Datapoint, index: int, task_id: str) -> None:
            try:
                result = await self._evaluate_trial(eval_id, datapoint, index, task_id)
                accumulate_scores(result)
                self.reporter.update(1)
            finally:
                semaphore.release()

        # `LaminarDataset.__iter__` streams from the API one page at a time and
        # drops each page once consumed, so a huge dataset is never fully
        # resident. Pulling the next page is a blocking call on the sync
        # client, so the iterator is advanced in a thread pool to keep
        # in-flight Trials running while we wait.
        loop = asyncio.get_event_loop()
        data = self.data
        data_iter = enumerate(iter(data))
        task_ids = _TaskIdResolver(self._logger)

        def next_task() -> tuple[int, Datapoint, str] | None:
            item = next(data_iter, None)
            if item is None:
                return None
            idx, datapoint = item
            source_index = data.source_index(idx) if isinstance(data, EvaluationDataset) else idx
            return idx, datapoint, task_ids.resolve(datapoint, source_index)

        first_error: BaseException | None = None

        def prune_finished(task_list: list[Task[None]]) -> list[Task[None]]:
            # Drop references to completed Trial tasks as we go. A failed
            # task's exception is recorded, NOT raised here: every Trial
            # must still be scheduled and run (matching the previous behavior,
            # where all tasks were created before `gather` surfaced errors).
            nonlocal first_error
            for task in task_list:
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc is not None and first_error is None:
                        first_error = exc
            return [task for task in task_list if not task.done()]

        # Create tasks only after acquiring the semaphore, so at most
        # `concurrency_limit` Trials are in flight (and referenced) at once.
        # Trials of one Task are scheduled back to back.
        while (item := await loop.run_in_executor(None, next_task)) is not None:
            idx, datapoint, task_id = item
            for _ in range(self.trials):
                _success = await semaphore.acquire()
                task = asyncio.create_task(run_trial(datapoint, idx, task_id))
                tasks.append(task)
                tasks = prune_finished(tasks)
                self._prune_upload_tasks()

        # Wait for the remaining in-flight tasks to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException) and first_error is None:
                    first_error = result

        if first_error is not None:
            raise first_error

        return {
            key: score_sums[key] / score_counts[key] for key in score_sums if score_counts[key] > 0
        }

    def _dataset_link(self, datapoint: Datapoint) -> EvaluationDatapointDatasetLink | None:
        source = self._source_dataset
        if (
            source is None
            or source.id is None
            or datapoint.id is None
            or datapoint.created_at is None
        ):
            return None
        return EvaluationDatapointDatasetLink(
            dataset_id=source.id,
            datapoint_id=datapoint.id,
            created_at=datapoint.created_at,
        )

    async def _evaluate_trial(
        self, eval_id: uuid.UUID, datapoint: Datapoint, index: int, task_id: str
    ) -> EvaluationResultDatapoint:
        evaluation_id = uuid.uuid4()

        eval_id_ctx = set_association_prop_context(
            metadata={"evaluation_id": str(eval_id)},
            trace_type=TraceType.EVALUATION,
            attach=False,
        )
        eval_id_token = attach_context(eval_id_ctx)

        try:
            with L.start_as_current_span("evaluation") as evaluation_span:
                L._set_trace_type(trace_type=TraceType.EVALUATION)  # pyright: ignore[reportPrivateUsage]
                evaluation_span.set_attribute(SPAN_TYPE, SpanType.EVALUATION.value)
                with L.start_as_current_span(
                    "executor", input={"data": datapoint.data}
                ) as executor_span:
                    executor_span_id = uuid.UUID(int=executor_span.get_span_context().span_id)
                    trace_id = uuid.UUID(int=executor_span.get_span_context().trace_id)

                    partial_trial = PartialEvaluationDatapoint(
                        id=evaluation_id,
                        data=datapoint.data,
                        target=datapoint.target,
                        index=index,
                        task_id=task_id,
                        trace_id=trace_id,
                        executor_span_id=executor_span_id,
                        metadata=datapoint.metadata,
                        dataset_link=self._dataset_link(datapoint),
                    )
                    # First, create the Trial with trace_id so that the UI can show it while it runs
                    await self.client.evals.save_datapoints(
                        eval_id, [partial_trial], self.group_name
                    )
                    executor_span.set_attribute(SPAN_TYPE, SpanType.EXECUTOR.value)
                    # Run synchronous executors in a thread pool to avoid blocking
                    if not is_async(self.executor):
                        loop = asyncio.get_event_loop()
                        output = await loop.run_in_executor(None, self.executor, datapoint.data)
                    else:
                        output = await self.executor(datapoint.data)

                    L.set_span_output(output)
                target = datapoint.target

                # Iterate over evaluators
                scores: dict[str, Numeric | None] = {}
                for evaluator_name, evaluator in self.evaluators.items():
                    # Check if evaluator is a HumanEvaluator instance
                    # Regular evaluator function
                    with L.start_as_current_span(
                        evaluator_name,
                        input={"output": output, "target": target},
                    ) as evaluator_span:
                        evaluator_span.set_attribute(SPAN_TYPE, SpanType.EVALUATOR.value)
                        if is_async(evaluator):
                            value = await cast(
                                Awaitable[EvaluatorFunctionReturnType], evaluator(output, target)
                            )
                        else:
                            loop = asyncio.get_event_loop()
                            value = cast(
                                EvaluatorFunctionReturnType,
                                await loop.run_in_executor(None, evaluator, output, target),
                            )
                        L.set_span_output(value)

                    # If evaluator returns a single number, use evaluator name as key
                    if isinstance(value, NumericTypes):
                        scores[evaluator_name] = value
                    else:
                        scores.update(value)

                trace_id = uuid.UUID(int=evaluation_span.get_span_context().trace_id)
        finally:
            try:
                detach_context(eval_id_token)
            except Exception:
                self._logger.debug("Failed to detach evaluation trace context", exc_info=True)

        trial_result = EvaluationResultDatapoint(
            id=evaluation_id,
            data=datapoint.data,
            target=target,
            executor_output=output,
            scores=scores,
            trace_id=trace_id,
            executor_span_id=executor_span_id,
            index=index,
            task_id=task_id,
            metadata=datapoint.metadata,
            dataset_link=self._dataset_link(datapoint),
        )

        # Create background upload task without awaiting it
        upload_task = asyncio.create_task(
            self.client.evals.save_datapoints(eval_id, [trial_result], self.group_name)
        )
        self.upload_tasks.append(upload_task)

        return trial_result

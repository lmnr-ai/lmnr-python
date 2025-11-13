import asyncio
import re
import uuid

from typing import Any
from typing_extensions import TypedDict

from tqdm import tqdm

from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.opentelemetry_lib.tracing.instruments import Instruments
from lmnr.opentelemetry_lib.tracing.attributes import HUMAN_EVALUATOR_OPTIONS, SPAN_TYPE

from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.datasets import EvaluationDataset, LaminarDataset
from lmnr.sdk.eval_control import EVALUATION_INSTANCES, PREPARE_ONLY
from lmnr.sdk.laminar import Laminar as L
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import (
    Datapoint,
    EvaluationDatapointDatasetLink,
    EvaluationResultDatapoint,
    EvaluatorFunction,
    ExecutorFunction,
    HumanEvaluator,
    Numeric,
    NumericTypes,
    PartialEvaluationDatapoint,
    SpanType,
    TraceType,
)
from lmnr.sdk.utils import from_env, is_async

DEFAULT_BATCH_SIZE = 5
MAX_EXPORT_BATCH_SIZE = 64


class EvaluationRunResult(TypedDict):
    average_scores: dict[str, Numeric]
    evaluation_id: uuid.UUID
    project_id: uuid.UUID
    url: str
    error_message: str | None


def get_evaluation_url(
    project_id: str, evaluation_id: str, base_url: str | None = None
):
    if not base_url or base_url == "https://api.lmnr.ai":
        base_url = "https://www.lmnr.ai"

    url = base_url
    url = re.sub(r"\/$", "", url)
    if url.endswith("localhost") or url.endswith("127.0.0.1"):
        # We best effort assume that the frontend is running on port 5667
        url = url + ":5667"
    return f"{url}/project/{project_id}/evaluations/{evaluation_id}"


def get_average_scores(results: list[EvaluationResultDatapoint]) -> dict[str, Numeric]:
    per_score_values = {}
    for result in results:
        for key, value in result.scores.items():
            if key not in per_score_values:
                per_score_values[key] = []
            per_score_values[key].append(value)

    average_scores = {}
    for key, values in per_score_values.items():
        scores = [v for v in values if v is not None]

        # If there are no scores, we don't want to include the key in the average scores
        if len(scores) > 0:
            average_scores[key] = sum(scores) / len(scores)

    return average_scores


class EvaluationReporter:
    def __init__(self, base_url):
        self.base_url = base_url

    def start(self, length: int):
        self.cli_progress = tqdm(
            total=length,
            bar_format="{bar} {percentage:3.0f}% | ETA: {remaining}s | {n_fmt}/{total_fmt}",
            ncols=60,
        )

    def update(self, batch_length: int):
        self.cli_progress.update(batch_length)

    def stop_with_error(self, error: Exception):
        if hasattr(self, "cli_progress"):
            self.cli_progress.close()
        raise error

    def stop(
        self, average_scores: dict[str, Numeric], project_id: str, evaluation_id: str
    ):
        self.cli_progress.close()
        print("Average scores:")
        for name, score in average_scores.items():
            print(f"{name}: {score}")
        print(
            f"Check the results at {get_evaluation_url(project_id, evaluation_id, self.base_url)}\n"
        )


class Evaluation:
    def __init__(
        self,
        data: EvaluationDataset | list[Datapoint | dict],
        executor: Any,
        evaluators: dict[str, EvaluatorFunction | HumanEvaluator],
        name: str | None = None,
        group_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        concurrency_limit: int = DEFAULT_BATCH_SIZE,
        project_api_key: str | None = None,
        base_url: str | None = None,
        base_http_url: str | None = None,
        http_port: int | None = None,
        grpc_port: int | None = None,
        instruments: (
            set[Instruments] | list[Instruments] | tuple[Instruments] | None
        ) = None,
        disabled_instruments: (
            set[Instruments] | list[Instruments] | tuple[Instruments] | None
        ) = None,
        max_export_batch_size: int | None = MAX_EXPORT_BATCH_SIZE,
        trace_export_timeout_seconds: int | None = None,
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
            instruments (set[Instruments] | None, optional): Set of modules\
                to auto-instrument. If None, all available instruments will be\
                used.
                See https://docs.lmnr.ai/tracing/automatic-instrumentation
                Defaults to None.
            disabled_instruments (set[Instruments] | None, optional): Set of modules\
                to disable auto-instrumentations. If None, only modules passed\
                as `instruments` will be disabled.
                Defaults to None.
        """

        if not evaluators:
            raise ValueError("No evaluators provided")

        evaluator_name_regex = re.compile(r"^[\w\s-]+$")
        for evaluator_name in evaluators:
            if not evaluator_name_regex.match(evaluator_name):
                raise ValueError(
                    f'Invalid evaluator key: "{evaluator_name}". '
                    "Keys must only contain letters, digits, hyphens,"
                    "underscores, or spaces."
                )

        base_url = base_url or from_env("LMNR_BASE_URL") or "https://api.lmnr.ai"

        self.reporter = EvaluationReporter(base_url)
        if isinstance(data, list):
            self.data = [
                (Datapoint.model_validate(point) if isinstance(point, dict) else point)
                for point in data
            ]
        else:
            self.data = data
        if not isinstance(self.data, LaminarDataset) and len(self.data) == 0:
            raise ValueError("No data provided. Skipping evaluation")
        self.executor = executor
        self.evaluators = evaluators
        self.group_name = group_name
        self.name = name
        self.metadata = metadata
        self.concurrency_limit = concurrency_limit
        self.batch_size = concurrency_limit
        self._logger = get_default_logger(self.__class__.__name__)
        self.upload_tasks = []
        self.base_http_url = f"{base_http_url or base_url}:{http_port or 443}"

        api_key = project_api_key or from_env("LMNR_PROJECT_API_KEY")
        if not api_key and not L.is_initialized():
            raise ValueError(
                "Please pass the project API key to `evaluate`"
                " or set the LMNR_PROJECT_API_KEY environment variable"
                " in your environment or .env file"
            )
        self.project_api_key = api_key

        if L.is_initialized():
            self.client = AsyncLaminarClient(
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
        if isinstance(self.data, LaminarDataset):
            self.data.set_client(
                LaminarClient(
                    base_url=self.base_http_url,
                    project_api_key=self.project_api_key,
                )
            )
            if not self.data.id:
                try:
                    datasets = await self.client.datasets.get_dataset_by_name(
                        self.data.name
                    )
                    if len(datasets) == 0:
                        self._logger.warning(f"Dataset {self.data.name} not found")
                    else:
                        self.data.id = datasets[0].id
                except Exception as e:
                    # Backward compatibility with old Laminar API (self hosted)
                    self._logger.warning(f"Error getting dataset {self.data.name}: {e}")

        try:
            evaluation = await self.client.evals.init(
                name=self.name, group_name=self.group_name, metadata=self.metadata
            )
            evaluation_id = evaluation.id
            project_id = evaluation.projectId
            url = get_evaluation_url(project_id, evaluation_id, self.reporter.base_url)

            print(f"Check the results at {url}")

            self.reporter.start(len(self.data))
            result_datapoints = await self._evaluate_in_batches(evaluation.id)
            # Wait for all background upload tasks to complete
            if self.upload_tasks:
                self._logger.debug(
                    f"Waiting for {len(self.upload_tasks)} upload tasks to complete"
                )
                await asyncio.gather(*self.upload_tasks)
                self._logger.debug("All upload tasks completed")
        except Exception as e:
            await self._shutdown()
            self.reporter.stop_with_error(e)

        average_scores = get_average_scores(result_datapoints)
        self.reporter.stop(average_scores, evaluation.projectId, evaluation.id)
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
        L.flush()
        await self.client.close()
        if isinstance(self.data, LaminarDataset) and self.data.client:
            self.data.client.close()

    async def _evaluate_in_batches(
        self, eval_id: uuid.UUID
    ) -> list[EvaluationResultDatapoint]:

        semaphore = asyncio.Semaphore(self.concurrency_limit)
        tasks = []
        data_iter = self.data if isinstance(self.data, list) else range(len(self.data))

        async def evaluate_task(datapoint, index):
            try:
                result = await self._evaluate_datapoint(eval_id, datapoint, index)
                self.reporter.update(1)
                return index, result
            finally:
                semaphore.release()

        # Create tasks only after acquiring semaphore
        for idx, item in enumerate(data_iter):
            await semaphore.acquire()
            datapoint = item if isinstance(self.data, list) else self.data[item]
            task = asyncio.create_task(evaluate_task(datapoint, idx))
            tasks.append(task)

        # Wait for all tasks to complete and preserve order
        results = await asyncio.gather(*tasks)
        ordered_results = [result for _, result in sorted(results, key=lambda x: x[0])]

        return ordered_results

    async def _evaluate_datapoint(
        self, eval_id: uuid.UUID, datapoint: Datapoint, index: int
    ) -> EvaluationResultDatapoint:
        evaluation_id = uuid.uuid4()
        with L.start_as_current_span("evaluation") as evaluation_span:
            L._set_trace_type(trace_type=TraceType.EVALUATION)
            evaluation_span.set_attribute(SPAN_TYPE, SpanType.EVALUATION.value)
            with L.start_as_current_span(
                "executor", input={"data": datapoint.data}
            ) as executor_span:
                executor_span_id = uuid.UUID(
                    int=executor_span.get_span_context().span_id
                )
                trace_id = uuid.UUID(int=executor_span.get_span_context().trace_id)

                partial_datapoint = PartialEvaluationDatapoint(
                    id=evaluation_id,
                    data=datapoint.data,
                    target=datapoint.target,
                    index=index,
                    trace_id=trace_id,
                    executor_span_id=executor_span_id,
                    metadata=datapoint.metadata,
                )
                if isinstance(self.data, LaminarDataset):
                    partial_datapoint.dataset_link = EvaluationDatapointDatasetLink(
                        dataset_id=self.data.id,
                        datapoint_id=datapoint.id,
                        created_at=datapoint.created_at,
                    )
                # First, create datapoint with trace_id so that we can show the dp in the UI
                await self.client.evals.save_datapoints(
                    eval_id, [partial_datapoint], self.group_name
                )
                executor_span.set_attribute(SPAN_TYPE, SpanType.EXECUTOR.value)
                # Run synchronous executors in a thread pool to avoid blocking
                if not is_async(self.executor):
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        None, self.executor, datapoint.data
                    )
                else:
                    output = await self.executor(datapoint.data)

                L.set_span_output(output)
            target = datapoint.target

            # Iterate over evaluators
            scores: dict[str, Numeric] = {}
            for evaluator_name, evaluator in self.evaluators.items():
                # Check if evaluator is a HumanEvaluator instance
                if isinstance(evaluator, HumanEvaluator):
                    # Create an empty span for human evaluators
                    with L.start_as_current_span(
                        evaluator_name, input={"output": output, "target": target}
                    ) as human_evaluator_span:
                        human_evaluator_span.set_attribute(
                            SPAN_TYPE, SpanType.HUMAN_EVALUATOR.value
                        )
                        if evaluator.options:
                            human_evaluator_span.set_attribute(
                                HUMAN_EVALUATOR_OPTIONS, json_dumps(evaluator.options)
                            )
                        # Human evaluators don't execute automatically, just create the span
                        L.set_span_output(None)

                    # We don't want to save the score for human evaluators
                    scores[evaluator_name] = None
                else:
                    # Regular evaluator function
                    with L.start_as_current_span(
                        evaluator_name, input={"output": output, "target": target}
                    ) as evaluator_span:
                        evaluator_span.set_attribute(
                            SPAN_TYPE, SpanType.EVALUATOR.value
                        )
                        if is_async(evaluator):
                            value = await evaluator(output, target)
                        else:
                            loop = asyncio.get_event_loop()
                            value = await loop.run_in_executor(
                                None, evaluator, output, target
                            )
                        L.set_span_output(value)

                    # If evaluator returns a single number, use evaluator name as key
                    if isinstance(value, NumericTypes):
                        scores[evaluator_name] = value
                    else:
                        scores.update(value)

            trace_id = uuid.UUID(int=evaluation_span.get_span_context().trace_id)

        eval_datapoint = EvaluationResultDatapoint(
            id=evaluation_id,
            data=datapoint.data,
            target=target,
            executor_output=output,
            scores=scores,
            trace_id=trace_id,
            executor_span_id=executor_span_id,
            index=index,
            metadata=datapoint.metadata,
        )
        if isinstance(self.data, LaminarDataset):
            eval_datapoint.dataset_link = EvaluationDatapointDatasetLink(
                dataset_id=self.data.id,
                datapoint_id=datapoint.id,
                created_at=datapoint.created_at,
            )

        # Create background upload task without awaiting it
        upload_task = asyncio.create_task(
            self.client.evals.save_datapoints(
                eval_id, [eval_datapoint], self.group_name
            )
        )
        self.upload_tasks.append(upload_task)

        return eval_datapoint


def evaluate(
    data: EvaluationDataset | list[Datapoint | dict],
    executor: ExecutorFunction,
    evaluators: dict[str, EvaluatorFunction | HumanEvaluator],
    name: str | None = None,
    group_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    concurrency_limit: int = DEFAULT_BATCH_SIZE,
    project_api_key: str | None = None,
    base_url: str | None = None,
    base_http_url: str | None = None,
    http_port: int | None = None,
    grpc_port: int | None = None,
    instruments: (
        set[Instruments] | list[Instruments] | tuple[Instruments] | None
    ) = None,
    disabled_instruments: (
        set[Instruments] | list[Instruments] | tuple[Instruments] | None
    ) = None,
    max_export_batch_size: int | None = MAX_EXPORT_BATCH_SIZE,
    trace_export_timeout_seconds: int | None = None,
) -> EvaluationRunResult | None:
    """
    If added to the file which is called through `lmnr eval` command, then
    registers the evaluation; otherwise, runs the evaluation.

    If there is no event loop, creates it and runs the evaluation until
    completion.
    If there is an event loop, returns an awaitable handle immediately. IMPORTANT:
    You must await the call to `evaluate`.

    Parameters:
        data (list[EvaluationDatapoint|dict] | EvaluationDataset):\
            List of data points to evaluate or an evaluation dataset.
                `data` is the input to the executor function,
                `target` is the input to the evaluator function.
        executor (Callable[..., Any]): The executor function.\
            Takes the data point + any additional arguments\
            and returns the output to evaluate.
        evaluators (dict[str, Callable[..., Any] | HumanEvaluator]): Evaluator\
            functions and HumanEvaluator instances with names. Each evaluator\
            function takes the output of the executor _and_ the target data,\
            and returns a score. The score can be a single number or a dict\
            of string keys and number values. If the score is a single number,\
            it will be named after the evaluator function.\
            HumanEvaluator instances create empty spans for manual evaluation.\
            Evaluator function names must contain only letters, digits, hyphens,\
            underscores, or spaces.
        name (str | None, optional): Optional name of the evaluation.\
            Used to identify the evaluation in the group. If not provided, a\
            random name will be generated.
            Defaults to None.
        group_name (str | None, optional): An identifier to group evaluations.\
            Only evaluations within the same group_name can be visually compared.\
            If not provided, set to "default".
            Defaults to None
        metadata (dict[str, Any] | None, optional): Optional metadata to associate with\
        concurrency_limit (int, optional): The concurrency limit for evaluation.
                        Defaults to DEFAULT_BATCH_SIZE.
        project_api_key (str | None, optional): The project API key.
                        Defaults to None.
        base_url (str | None, optional): The base URL for Laminar API.\
                        Useful if self-hosted elsewhere. Do NOT include the\
                        port, use `http_port` and `grpc_port` instead.
                        Defaults to "https://api.lmnr.ai".
        base_http_url (str | None, optional): The base HTTP URL for Laminar API.\
                        Only set this if your Laminar backend HTTP is proxied\
                        through a different host. If not specified, defaults\
                        to https://api.lmnr.ai.
        http_port (int | None, optional): The port for Laminar API's HTTP\
                        service. 443 is used if not specified.
                        Defaults to None.
        grpc_port (int | None, optional): The port for Laminar API's gRPC\
                        service. 8443 is used if not specified.
                        Defaults to None.
        instruments (set[Instruments] | None, optional): Set of modules to\
                        auto-instrument. If None, all available instruments\
                        will be used.
                        Defaults to None.
        disabled_instruments (set[Instruments] | None, optional): Set of modules\
                        to disable auto-instrumentations. If None, no\
                        If None, only modules passed as `instruments` will be disabled.
                        Defaults to None.
        trace_export_timeout_seconds (int | None, optional): The timeout for\
                        trace export on OpenTelemetry exporter. Defaults to None.
    """
    evaluation = Evaluation(
        data=data,
        executor=executor,
        evaluators=evaluators,
        group_name=group_name,
        metadata=metadata,
        name=name,
        concurrency_limit=concurrency_limit,
        project_api_key=project_api_key,
        base_url=base_url,
        base_http_url=base_http_url,
        http_port=http_port,
        grpc_port=grpc_port,
        instruments=instruments,
        disabled_instruments=disabled_instruments,
        max_export_batch_size=max_export_batch_size,
        trace_export_timeout_seconds=trace_export_timeout_seconds,
    )

    if PREPARE_ONLY.get():
        existing_evaluations = EVALUATION_INSTANCES.get([])
        new_evaluations = (existing_evaluations or []) + [evaluation]
        EVALUATION_INSTANCES.set(new_evaluations)
        return None
    else:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return evaluation.run()
            else:
                return asyncio.run(evaluation.run())
        except RuntimeError:
            return asyncio.run(evaluation.run())

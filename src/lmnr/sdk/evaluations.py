import asyncio
import re
import sys
import uuid

from contextlib import contextmanager
from tqdm import tqdm
from typing import Any, Awaitable, Optional, Set, Union

from ..traceloop_sdk.instruments import Instruments
from ..traceloop_sdk.tracing.attributes import SPAN_TYPE

from .datasets import EvaluationDataset
from .laminar import Laminar as L
from .log import get_default_logger
from .types import (
    Datapoint,
    EvaluationResultDatapoint,
    EvaluatorFunction,
    ExecutorFunction,
    HumanEvaluator,
    Numeric,
    NumericTypes,
    SpanType,
    TraceType,
)
from .utils import is_async

DEFAULT_BATCH_SIZE = 5

_evaluation = None
_set_global_evaluation = False


@contextmanager
def set_global_evaluation(set_global_evaluation: bool):
    global _set_global_evaluation
    original = _set_global_evaluation
    try:
        _set_global_evaluation = set_global_evaluation
        yield
    finally:
        _set_global_evaluation = original
        pass


def get_evaluation_url(project_id: str, evaluation_id: str):
    return f"https://www.lmnr.ai/project/{project_id}/evaluations/{evaluation_id}"


def get_average_scores(results: list[EvaluationResultDatapoint]) -> dict[str, Numeric]:
    per_score_values = {}
    for result in results:
        for key, value in result.scores.items():
            if key not in per_score_values:
                per_score_values[key] = []
            per_score_values[key].append(value)

    average_scores = {}
    for key, values in per_score_values.items():
        average_scores[key] = sum(values) / len(values)

    return average_scores


class EvaluationReporter:
    def __init__(self):
        pass

    def start(self, length: int):
        self.cli_progress = tqdm(
            total=length,
            bar_format="{bar} {percentage:3.0f}% | ETA: {remaining}s | {n_fmt}/{total_fmt}",
            ncols=60,
        )

    def update(self, batch_length: int):
        self.cli_progress.update(batch_length)

    def stopWithError(self, error: Exception):
        self.cli_progress.close()
        sys.stderr.write(f"\nError: {error}\n")

    def stop(
        self, average_scores: dict[str, Numeric], project_id: str, evaluation_id: str
    ):
        self.cli_progress.close()
        print(
            f"\nCheck the results at {get_evaluation_url(project_id, evaluation_id)}\n"
        )
        print("Average scores:")
        for name, score in average_scores.items():
            print(f"{name}: {score}")
        print("\n")


class Evaluation:
    def __init__(
        self,
        data: Union[EvaluationDataset, list[Union[Datapoint, dict]]],
        executor: Any,
        evaluators: dict[str, EvaluatorFunction],
        human_evaluators: list[HumanEvaluator] = [],
        name: Optional[str] = None,
        group_id: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        project_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        http_port: Optional[int] = None,
        grpc_port: Optional[int] = None,
        instruments: Optional[Set[Instruments]] = None,
    ):
        """
        Initializes an instance of the Evaluations class.

        Parameters:
            data (Union[List[EvaluationDatapoint|dict], EvaluationDataset]):\
                List of data points to evaluate or an evaluation dataset.
                            `data` is the input to the executor function,
                            `target` is the input to the evaluator function.
            executor (Callable[..., Any]): The executor function.\
                            Takes the data point + any additional arguments\
                            and returns the output to evaluate.
            evaluators (dict[str, Callable[..., Any]]): Evaluator functions and\
                names. Each evaluator function takes the output of the executor\
                _and_ the target data, and returns a score. The score can be a\
                single number or a dict of string keys and number values.\
                If the score is a single number, it will be named after the\
                evaluator function. Evaluator function names must contain only\
                letters, digits, hyphens, underscores, or spaces.
            human_evaluators (list[HumanEvaluator], optional):\
                [Beta] List of instances of HumanEvaluator. For now, human\
                evaluator only holds the queue name.
                Defaults to an empty list.
            name (Optional[str], optional): Optional name of the evaluation.\
                Used to identify the evaluation in the group.\
                If not provided, a random name will be generated.
                Defaults to None.
            group_id (Optional[str], optional): an identifier to group\
                evaluations. Only evaluations within the same group_id can be\
                visually compared. If not provided, "default" is assigned.
                Defaults to None
            batch_size (int, optional): The batch size for evaluation. This many\
                data points will be evaluated in parallel.
                Defaults to DEFAULT_BATCH_SIZE.
            project_api_key (Optional[str], optional): The project API key.\
                If not provided, LMNR_PROJECT_API_KEY environment variable is\
                used.
                Defaults to an empty string.
            base_url (Optional[str], optional): The base URL for Laminar API.\
                Useful if self-hosted. Do NOT include the port, use `http_port`\
                and `grpc_port` instead.
                Defaults to "https://api.lmnr.ai".
            http_port (Optional[int], optional): The port for Laminar API\
                HTTP service. Defaults to 443 if not specified.
            grpc_port (Optional[int], optional): The port for Laminar API\
                gRPC service. Defaults to 8443 if not specified.
            instruments (Optional[Set[Instruments]], optional): Set of modules\
                to auto-instrument. If None, all available instruments will be\
                used.
                See https://docs.lmnr.ai/tracing/automatic-instrumentation
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

        self.is_finished = False
        self.reporter = EvaluationReporter()
        if isinstance(data, list):
            self.data = [
                (Datapoint.model_validate(point) if isinstance(point, dict) else point)
                for point in data
            ]
        else:
            self.data = data
        self.executor = executor
        self.evaluators = evaluators
        self.group_id = group_id
        self.name = name
        self.batch_size = batch_size
        self._logger = get_default_logger(self.__class__.__name__)
        self.human_evaluators = human_evaluators
        L.initialize(
            project_api_key=project_api_key,
            base_url=base_url,
            http_port=http_port,
            grpc_port=grpc_port,
            instruments=instruments,
        )

    def run(self) -> Union[None, Awaitable[None]]:
        if self.is_finished:
            raise Exception("Evaluation is already finished")

        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.create_task(self._run())
        else:
            return loop.run_until_complete(self._run())

    async def _run(self) -> None:
        self.reporter.start(len(self.data))

        try:
            result_datapoints = await self._evaluate_in_batches()
        except Exception as e:
            self.reporter.stopWithError(e)
            self.is_finished = True
            return

        # For now add all human evaluators to all result datapoints
        # In the future, we will add ways to specify which human evaluators
        # to add to which result datapoints, e.g. sample some randomly
        for result_datapoint in result_datapoints:
            result_datapoint.human_evaluators = self.human_evaluators or {}

        evaluation = L.create_evaluation(
            data=result_datapoints, group_id=self.group_id, name=self.name
        )
        average_scores = get_average_scores(result_datapoints)
        self.reporter.stop(average_scores, evaluation.projectId, evaluation.id)
        self.is_finished = True

    async def _evaluate_in_batches(self) -> list[EvaluationResultDatapoint]:
        result_datapoints = []
        for i in range(0, len(self.data), self.batch_size):
            batch = (
                self.data[i : i + self.batch_size]
                if isinstance(self.data, list)
                else self.data.slice(i, i + self.batch_size)
            )
            batch_datapoints = await self._evaluate_batch(batch)
            result_datapoints.extend(batch_datapoints)
            self.reporter.update(len(batch))
        return result_datapoints

    async def _evaluate_batch(
        self, batch: list[Datapoint]
    ) -> list[EvaluationResultDatapoint]:
        batch_promises = [self._evaluate_datapoint(datapoint) for datapoint in batch]
        results = await asyncio.gather(*batch_promises)
        return results

    async def _evaluate_datapoint(
        self, datapoint: Datapoint
    ) -> EvaluationResultDatapoint:
        with L.start_as_current_span("evaluation") as evaluation_span:
            L._set_trace_type(trace_type=TraceType.EVALUATION)
            evaluation_span.set_attribute(SPAN_TYPE, SpanType.EVALUATION.value)
            with L.start_as_current_span(
                "executor", input={"data": datapoint.data}
            ) as executor_span:
                executor_span.set_attribute(SPAN_TYPE, SpanType.EXECUTOR.value)
                output = (
                    await self.executor(datapoint.data)
                    if is_async(self.executor)
                    else self.executor(datapoint.data)
                )
                L.set_span_output(output)
                executor_span_id = uuid.UUID(
                    int=executor_span.get_span_context().span_id
                )
            target = datapoint.target

            # Iterate over evaluators
            scores: dict[str, Numeric] = {}
            for evaluator_name, evaluator in self.evaluators.items():
                with L.start_as_current_span(
                    evaluator_name, input={"output": output, "target": target}
                ) as evaluator_span:
                    evaluator_span.set_attribute(SPAN_TYPE, SpanType.EVALUATOR.value)
                    value = (
                        await evaluator(output, target)
                        if is_async(evaluator)
                        else evaluator(output, target)
                    )
                    L.set_span_output(value)

                # If evaluator returns a single number, use evaluator name as key
                if isinstance(value, NumericTypes):
                    scores[evaluator_name] = value
                else:
                    scores.update(value)

            trace_id = uuid.UUID(int=evaluation_span.get_span_context().trace_id)
            return EvaluationResultDatapoint(
                data=datapoint.data,
                target=target,
                executor_output=output,
                scores=scores,
                trace_id=trace_id,
                executor_span_id=executor_span_id,
            )


def evaluate(
    data: Union[EvaluationDataset, list[Union[Datapoint, dict]]],
    executor: ExecutorFunction,
    evaluators: dict[str, EvaluatorFunction],
    human_evaluators: list[HumanEvaluator] = [],
    name: Optional[str] = None,
    group_id: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    project_api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    http_port: Optional[int] = None,
    grpc_port: Optional[int] = None,
    instruments: Optional[Set[Instruments]] = None,
) -> Optional[Awaitable[None]]:
    """
    If added to the file which is called through `lmnr eval` command, then
    registers the evaluation; otherwise, runs the evaluation.

    If there is no event loop, creates it and runs the evaluation until
    completion.
    If there is an event loop, schedules the evaluation as a task in the
    event loop and returns an awaitable handle.

    Parameters:
        data (Union[list[EvaluationDatapoint|dict]], EvaluationDataset]):\
                    List of data points to evaluate or an evaluation dataset.
                        `data` is the input to the executor function,
                        `target` is the input to the evaluator function.
        executor (Callable[..., Any]): The executor function.\
                        Takes the data point + any additional arguments\
                        and returns the output to evaluate.
        evaluators (List[Callable[..., Any]]): 
            evaluators (dict[str, Callable[..., Any]]): Evaluator functions and\
                names. Each evaluator function takes the output of the executor\
                _and_ the target data, and returns a score. The score can be a\
                single number or a dict of string keys and number values.\
                If the score is a single number, it will be named after the\
                evaluator function. Evaluator function names must contain only\
                letters, digits, hyphens, underscores, or spaces.
        human_evaluators (list[HumanEvaluator], optional):\
            [Beta] List of instances of HumanEvaluator. For now, human\
            evaluator only holds the queue name.
            Defaults to an empty list.
        name (Optional[str], optional): Optional name of the evaluation.\
                        Used to identify the evaluation in the group.\
                        If not provided, a random name will be generated.
                        Defaults to None.
        group_id (Optional[str], optional): an identifier to group evaluations.\
                        Only evaluations within the same group_id can be\
                        visually compared. If not provided, set to "default".
                        Defaults to None
        batch_size (int, optional): The batch size for evaluation.
                        Defaults to DEFAULT_BATCH_SIZE.
        project_api_key (Optional[str], optional): The project API key.
                        Defaults to None.
        base_url (Optional[str], optional): The base URL for Laminar API.\
                        Useful if self-hosted elsewhere. Do NOT include the\
                        port, use `http_port` and `grpc_port` instead.
                        Defaults to "https://api.lmnr.ai".
        http_port (Optional[int], optional): The port for Laminar API's HTTP\
                        service. 443 is used if not specified.
                        Defaults to None.
        grpc_port (Optional[int], optional): The port for Laminar API's gRPC\
                        service. 8443 is used if not specified.
                        Defaults to None.
        instruments (Optional[Set[Instruments]], optional): Set of modules to\
                        auto-instrument. If None, all available instruments\
                        will be used.
                        Defaults to None.
    """

    evaluation = Evaluation(
        data=data,
        executor=executor,
        evaluators=evaluators,
        group_id=group_id,
        human_evaluators=human_evaluators,
        name=name,
        batch_size=batch_size,
        project_api_key=project_api_key,
        base_url=base_url,
        http_port=http_port,
        grpc_port=grpc_port,
        instruments=instruments,
    )

    global _evaluation
    if _set_global_evaluation:
        _evaluation = evaluation
    else:
        return evaluation.run()

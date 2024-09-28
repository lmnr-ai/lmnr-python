import asyncio
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Awaitable, Optional, Set, Union
import uuid

from tqdm import tqdm

from ..traceloop_sdk.instruments import Instruments
from ..traceloop_sdk.tracing.attributes import SPAN_TYPE

from .laminar import Laminar as L
from .types import (
    Datapoint,
    EvaluationResultDatapoint,
    EvaluatorFunction,
    ExecutorFunction,
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


class EvaluationReporter:
    def __init__(self):
        pass

    def start(self, name: str, project_id: str, id: str, length: int):
        print(f"Running evaluation {name}...\n")
        print(f"Check progress and results at {get_evaluation_url(project_id, id)}\n")
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

    def stop(self, average_scores: dict[str, Numeric]):
        self.cli_progress.close()
        print("\nAverage scores:")
        for name, score in average_scores.items():
            print(f"{name}: {score}")
        print("\n")


class EvaluationDataset(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Datapoint:
        pass

    def slice(self, start: int, end: int):
        return [self[i] for i in range(max(start, 0), min(end, len(self)))]


class Evaluation:
    def __init__(
        self,
        data: Union[EvaluationDataset, list[Union[Datapoint, dict]]],
        executor: Any,
        evaluators: dict[str, EvaluatorFunction],
        name: Optional[str] = None,
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
            data (Union[List[Union[EvaluationDatapoint, dict]], EvaluationDataset]): List of data points to evaluate or an evaluation dataset.
                            `data` is the input to the executor function,
                            `target` is the input to the evaluator function.
            executor (Callable[..., Any]): The executor function.
                            Takes the data point + any additional arguments
                            and returns the output to evaluate.
            evaluators (List[Callable[..., Any]]): List of evaluator functions.
                Each evaluator function takes the output of the executor _and_
                the target data, and returns a score. The score can be a
                single number or a record of string keys and number values.
                If the score is a single number, it will be named after the
                evaluator function. If the function is anonymous, it will be
                named `evaluator_${index}`, where index is the index of the
                evaluator function in the list starting from 1.
            name (Optional[str], optional): The name of the evaluation.
                            It will be auto-generated if not provided.
            batch_size (int, optional): The batch size for evaluation.
                            Defaults to DEFAULT_BATCH_SIZE.
            project_api_key (Optional[str], optional): The project API key.
                            Defaults to an empty string.
            base_url (Optional[str], optional): The base URL for the Laminar API.
                            Useful if self-hosted elsewhere.
                            Defaults to "https://api.lmnr.ai".
            http_port (Optional[int], optional): The port for the Laminar API HTTP service.
                            Defaults to 443.
            instruments (Optional[Set[Instruments]], optional): Set of modules to auto-instrument.
                            Defaults to None. If None, all available instruments will be used.
        """

        self.is_finished = False
        self.name = name
        self.reporter = EvaluationReporter()
        self.executor = executor
        self.evaluators = evaluators
        if isinstance(data, list):
            self.data = [
                (Datapoint.model_validate(point) if isinstance(point, dict) else point)
                for point in data
            ]
        else:
            self.data = data
        self.batch_size = batch_size
        L.initialize(
            project_api_key=project_api_key,
            base_url=base_url,
            http_port=http_port,
            grpc_port=grpc_port,
            instruments=instruments,
        )

    def run(self) -> Union[None, Awaitable[None]]:
        """Runs the evaluation.

        Creates a new evaluation if no evaluation with such name exists, or
        adds data to an existing one otherwise. Evaluates data points in
        batches of `self.batch_size`. The executor
        function is called on each data point to get the output,
        and then evaluate it by each evaluator function.

        Usage:
        ```python
        # in a synchronous context:
        e.run()
        # in an asynchronous context:
        await e.run()
        ```

        """
        if self.is_finished:
            raise Exception("Evaluation is already finished")

        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.create_task(self._run())
        else:
            return loop.run_until_complete(self._run())

    async def _run(self) -> None:
        evaluation = L.create_evaluation(self.name)
        self.reporter.start(
            evaluation.name,
            evaluation.projectId,
            evaluation.id,
            len(self.data),
        )

        try:
            await self.evaluate_in_batches(evaluation.id)
        except Exception as e:
            L.update_evaluation_status(evaluation.id, "Error")
            self.reporter.stopWithError(e)
            self.is_finished = True
            return

        update_evaluation_response = L.update_evaluation_status(evaluation.id, "Finished")
        average_scores = update_evaluation_response.stats.averageScores
        self.reporter.stop(average_scores)
        self.is_finished = True

    async def evaluate_in_batches(self, evaluation_id: uuid.UUID):
        for i in range(0, len(self.data), self.batch_size):
            batch = (
                self.data[i: i + self.batch_size]
                if isinstance(self.data, list)
                else self.data.slice(i, i + self.batch_size)
            )
            try:
                results = await self._evaluate_batch(batch)
                L.post_evaluation_results(evaluation_id, results)
            except Exception as e:
                print(f"Error evaluating batch: {e}")
            finally:
                self.reporter.update(len(batch))

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
            )


def evaluate(
    data: Union[EvaluationDataset, list[Union[Datapoint, dict]]],
    executor: ExecutorFunction,
    evaluators: dict[str, EvaluatorFunction],
    name: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    project_api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    http_port: Optional[int] = None,
    grpc_port: Optional[int] = None,
    instruments: Optional[Set[Instruments]] = None,
) -> Optional[Awaitable[None]]:
    """
    If added to the file which is called through lmnr eval command, then simply registers the evaluation.
    Otherwise, if there is no event loop, creates it and runs the evaluation until completion.
    If there is an event loop, schedules the evaluation as a task in the event loop and returns an awaitable handle.

    Parameters:
        data (Union[List[Union[EvaluationDatapoint, dict]], EvaluationDataset]): List of data points to evaluate or an evaluation dataset.
                        `data` is the input to the executor function,
                        `target` is the input to the evaluator function.
        executor (Callable[..., Any]): The executor function.
                        Takes the data point + any additional arguments
                        and returns the output to evaluate.
        evaluators (List[Callable[..., Any]]): List of evaluator functions.
            Each evaluator function takes the output of the executor _and_
            the target data, and returns a score. The score can be a
            single number or a record of string keys and number values.
            If the score is a single number, it will be named after the
            evaluator function. If the function is anonymous, it will be
            named `evaluator_${index}`, where index is the index of the
            evaluator function in the list starting from 1.
        name (Optional[str], optional): The name of the evaluation.
            It will be auto-generated if not provided.
        batch_size (int, optional): The batch size for evaluation.
                        Defaults to DEFAULT_BATCH_SIZE.
        project_api_key (Optional[str], optional): The project API key.
                        Defaults to an empty string.
        base_url (Optional[str], optional): The base URL for the Laminar API.
                        Useful if self-hosted elsewhere.
                        Defaults to "https://api.lmnr.ai".
        http_port (Optional[int], optional): The port for the Laminar API HTTP service.
                        Defaults to 443.
        grpc_port (Optional[int], optional): The port for the Laminar API gRPC service.
                        Defaults to 8443.
        instruments (Optional[Set[Instruments]], optional): Set of modules to auto-instrument.
                        Defaults to None. If None, all available instruments will be used.
    """

    evaluation = Evaluation(
        data=data,
        executor=executor,
        evaluators=evaluators,
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

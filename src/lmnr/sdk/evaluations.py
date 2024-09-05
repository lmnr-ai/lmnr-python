from typing import Union

from .utils import is_async
from .types import EvaluatorFunction, ExecutorFunction, EvaluationDatapoint, Numeric
from .laminar import Laminar as L
import asyncio

from abc import ABC, abstractmethod

DEFAULT_BATCH_SIZE = 5


class EvaluationDataset(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> EvaluationDatapoint:
        pass

    def slice(self, start: int, end: int):
        return [self[i] for i in range(max(start, 0), min(end, len(self)))]


class Evaluation:
    def __init__(
        self,
        name,
        data: Union[EvaluationDataset, list[Union[EvaluationDatapoint, dict]]],
        executor: ExecutorFunction,
        evaluators: list[EvaluatorFunction],
        batch_size: int = DEFAULT_BATCH_SIZE,
        project_api_key: str = "",
        base_url: str = "https://api.lmnr.ai",
    ):
        self.name = name
        self.executor = executor
        self.evaluators = dict(
            zip(
                [
                    (
                        e.__name__
                        if e.__name__ and e.__name__ != "<lambda>"
                        else f"evaluator_{i+1}"
                    )
                    for i, e in enumerate(evaluators)
                ],
                evaluators,
            )
        )
        self.evaluator_names = list(self.evaluators.keys())
        if isinstance(data, list):
            self.data = [
                (
                    EvaluationDatapoint.model_validate(point)
                    if isinstance(point, dict)
                    else point
                )
                for point in data
            ]
        else:
            self.data = data
        self.batch_size = batch_size
        L.initialize(project_api_key=project_api_key, base_url=base_url)

    async def run(self):
        response = L.create_evaluation(self.name)
        batch_promises = []

        for i in range(0, len(self.data), self.batch_size):
            batch = (
                self.data[i : i + self.batch_size]
                if isinstance(self.data, list)
                else self.data.slice(i, i + self.batch_size)
            )
            batch_promises.append(self.evaluate_batch(batch))

        try:
            await asyncio.gather(*batch_promises)
            L.update_evaluation_status(response.name, "Finished")
            print(f"Evaluation {response.id} complete")
        except Exception as e:
            print(f"Error evaluating batch: {e}")

    async def evaluate_batch(self, batch: list[EvaluationDatapoint]):
        results = []
        for datapoint in batch:
            output = (
                await self.executor(datapoint.data)
                if is_async(self.executor)
                else self.executor(datapoint.data)
            )
            target = datapoint.target

            # iterate in order of evaluators
            scores = {}
            for evaluator_name in self.evaluator_names:
                evaluator = self.evaluators[evaluator_name]
                value = (
                    await evaluator(output, target)
                    if is_async(evaluator)
                    else evaluator(output, target)
                )

                # if the evaluator returns a single number, use the evaluator name as the key
                if isinstance(value, Numeric):
                    scores[evaluator_name] = value
                else:
                    # if the evaluator returns an object, use the object keys as the keys
                    scores.update(value)

            results.append(
                {
                    "executorOutput": output,
                    "data": datapoint.data,
                    "target": target,
                    "scores": scores,
                }
            )

        return L.post_evaluation_results(self.name, results)

from typing import NoReturn
from uuid import UUID

from tqdm import tqdm

from lmnr.sdk.evaluations.utils import get_evaluation_url
from lmnr.sdk.types import Numeric


class EvaluationReporter:
    def __init__(self, base_url: str, frontend_port: int | None = None):
        self.base_url: str = base_url
        self.frontend_port: int | None = frontend_port
        self.cli_progress: tqdm[NoReturn] | None = None

    def start(self, length: int):
        self.cli_progress = tqdm(
            total=length,
            bar_format="{bar} {percentage:3.0f}% | ETA: {remaining}s | {n_fmt}/{total_fmt}",
            ncols=60,
        )

    def update(self, batch_length: int):
        assert self.cli_progress is not None
        _display_method_triggered = self.cli_progress.update(batch_length)

    def stop_with_error(self, error: Exception):
        if self.cli_progress is not None:
            self.cli_progress.close()
        raise error

    def stop(
        self, average_scores: dict[str, Numeric], project_id: UUID, evaluation_id: UUID
    ):
        assert self.cli_progress is not None
        self.cli_progress.close()
        print("Average scores:")
        for name, score in average_scores.items():
            print(f"{name}: {score}")
        print(
            f"Check the results at {get_evaluation_url(project_id, evaluation_id, self.base_url, self.frontend_port)}\n"
        )

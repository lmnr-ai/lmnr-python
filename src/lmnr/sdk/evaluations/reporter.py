from uuid import UUID

from tqdm import tqdm

from lmnr.sdk.evaluations.utils import get_evaluation_url
from lmnr.sdk.types import Numeric


class EvaluationReporter:
    def __init__(self, base_url, frontend_port: int | None = None):
        self.base_url = base_url
        self.frontend_port = frontend_port

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
        self, average_scores: dict[str, Numeric], project_id: UUID, evaluation_id: UUID
    ):
        self.cli_progress.close()
        print("Average scores:")
        for name, score in average_scores.items():
            print(f"{name}: {score}")
        print(
            f"Check the results at {get_evaluation_url(project_id, evaluation_id, self.base_url, self.frontend_port)}\n"
        )

from uuid import UUID

from lmnr.sdk.evaluations.models import EvaluationResultDatapoint
from lmnr.sdk.types import Numeric
from lmnr.sdk.utils import from_env, get_frontend_url


def get_evaluation_url(
    project_id: UUID,
    evaluation_id: UUID,
    base_url: str | None = None,
    frontend_port: int | None = None,
):
    """
    Get the frontend URL for an evaluation.

    Args:
        project_id: Project ID
        evaluation_id: Evaluation ID
        base_url: Base API URL
        frontend_port: Optional frontend port for localhost (defaults to 5667)

    Returns:
        Full URL to the evaluation in the frontend
    """

    # Check environment variable if frontend_port not explicitly provided
    if frontend_port is None:
        port_str = from_env("LMNR_FRONTEND_PORT")
        if port_str:
            try:
                frontend_port = int(port_str)
            except ValueError:
                pass

    frontend_url = get_frontend_url(base_url, frontend_port)
    return f"{frontend_url}/project/{project_id}/evaluations/{evaluation_id}"


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

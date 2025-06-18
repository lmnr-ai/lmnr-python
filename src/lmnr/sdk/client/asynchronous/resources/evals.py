"""Evals resource for interacting with Laminar evaluations API."""

from typing import Any
import uuid

from lmnr.sdk.client.asynchronous.resources.base import BaseAsyncResource
from lmnr.sdk.types import (
    InitEvaluationResponse,
    EvaluationResultDatapoint,
    PartialEvaluationDatapoint,
)


class AsyncEvals(BaseAsyncResource):
    """Resource for interacting with Laminar evaluations API."""

    async def init(
        self, name: str | None = None, group_name: str | None = None
    ) -> InitEvaluationResponse:
        """Initialize a new evaluation.

        Args:
            name (str | None, optional): Name of the evaluation. Defaults to None.
            group_name (str | None, optional): Group name for the evaluation. Defaults to None.

        Returns:
            InitEvaluationResponse: The response from the initialization request.
        """
        response = await self._client.post(
            self._base_url + "/v1/evals",
            json={
                "name": name,
                "groupName": group_name,
            },
            headers=self._headers(),
        )
        if response.status_code != 200:
            if response.status_code == 401:
                raise ValueError("Unauthorized. Please check your project API key.")
            raise ValueError(f"Error initializing evaluation: {response.text}")
        resp_json = response.json()
        return InitEvaluationResponse.model_validate(resp_json)

    async def save_datapoints(
        self,
        eval_id: uuid.UUID,
        datapoints: list[EvaluationResultDatapoint | PartialEvaluationDatapoint],
        group_name: str | None = None,
    ):
        """Save evaluation datapoints.

        Args:
            eval_id (uuid.UUID): The evaluation ID.
            datapoints (list[EvaluationResultDatapoint | PartialEvaluationDatapoint]): The datapoints to save.
            group_name (str | None, optional): Group name for the datapoints. Defaults to None.

        Raises:
            ValueError: If there's an error saving the datapoints.
        """
        response = await self._client.post(
            self._base_url + f"/v1/evals/{eval_id}/datapoints",
            json={
                "points": [datapoint.to_dict() for datapoint in datapoints],
                "groupName": group_name,
            },
            headers=self._headers(),
        )
        if response.status_code != 200:
            raise ValueError(f"Error saving evaluation datapoints: {response.text}")
        
    
    async def update_datapoint(
        self,
        eval_id: uuid.UUID,
        datapoint_id: uuid.UUID,
        scores: dict[str, float | int],
        executor_output: Any | None = None,
    ) -> None:
        """Update a datapoint with evaluation results.

        Args:
            eval_id (uuid.UUID): The evaluation ID.
            datapoint_id (uuid.UUID): The datapoint ID.
            executor_output (Any): The executor output.
            scores (dict[str, float | int] | None, optional): The scores. Defaults to None.
        """
        
        response = await self._client.post(
            self._base_url + f"/v1/evals/{eval_id}/datapoints/{datapoint_id}",
            json={
                "executorOutput": executor_output,
                "scores": scores,
            },
            headers=self._headers(),
        )

        if response.status_code != 200:
            raise ValueError(f"Error updating evaluation datapoint: {response.text}")
        
    
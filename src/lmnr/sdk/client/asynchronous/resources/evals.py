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
        self, name: str | None = None, group_name: str | None = None, metadata: dict[str, Any] | None = None
    ) -> InitEvaluationResponse:
        """Initialize a new evaluation.

        Args:
            name (str | None, optional): Name of the evaluation. Defaults to None.
            group_name (str | None, optional): Group name for the evaluation. Defaults to None.
            metadata (dict[str, Any] | None, optional): Metadata to associate with. Defaults to None.

        Returns:
            InitEvaluationResponse: The response from the initialization request.
        """
        response = await self._client.post(
            self._base_url + "/v1/evals",
            json={
                "name": name,
                "groupName": group_name,
                "metadata": metadata,
            },
            headers=self._headers(),
        )
        if response.status_code != 200:
            if response.status_code == 401:
                raise ValueError("Unauthorized. Please check your project API key.")
            raise ValueError(f"Error initializing evaluation: {response.text}")
        resp_json = response.json()
        return InitEvaluationResponse.model_validate(resp_json)

    async def create_evaluation(
        self,
        name: str | None = None,
        group_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """
        Create a new evaluation and return its ID.
        
        Parameters:
            name (str | None, optional): Optional name of the evaluation.
            group_name (str | None, optional): An identifier to group evaluations.
            metadata (dict[str, Any] | None, optional): Metadata to associate with. Defaults to None.

        Returns:
            uuid.UUID: The evaluation ID.
        """
        evaluation = await self.init(name=name, group_name=group_name, metadata=metadata)
        return evaluation.id

    async def create_datapoint(
        self,
        eval_id: uuid.UUID,
        data: Any,
        target: Any = None,
        metadata: dict[str, Any] | None = None,
        index: int | None = None,
        trace_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """
        Create a datapoint for an evaluation.
        
        Parameters:
            eval_id (uuid.UUID): The evaluation ID.
            data: The input data for the executor.
            target: The target/expected output for evaluators.
            metadata (dict[str, Any] | None, optional): Optional metadata.
            index (int | None, optional): Optional index of the datapoint.
            trace_id (uuid.UUID | None, optional): Optional trace ID.
        
        Returns:
            uuid.UUID: The datapoint ID.
        """
        
        datapoint_id = uuid.uuid4()
        
        # Create a minimal datapoint first
        partial_datapoint = PartialEvaluationDatapoint(
            id=datapoint_id,
            data=data,
            target=target,
            index=index or 0,
            trace_id=trace_id or uuid.uuid4(),
            executor_span_id=uuid.uuid4(),  # Will be updated when executor runs
            metadata=metadata,
        )
        
        await self.save_datapoints(eval_id, [partial_datapoint])
        return datapoint_id

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
        
    
"""Evals resource for interacting with Laminar evaluations API."""

import uuid
import warnings

from typing import Any

from lmnr.sdk.client.synchronous.resources.base import BaseResource
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import (
    GetDatapointsResponse,
    EvaluationResultDatapoint,
    InitEvaluationResponse,
    PartialEvaluationDatapoint,
)
from lmnr.sdk.utils import serialize

INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH = 16_000_000  # 16MB
logger = get_default_logger(__name__)


class Evals(BaseResource):
    """Resource for interacting with Laminar evaluations API."""

    def init(
        self,
        name: str | None = None,
        group_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InitEvaluationResponse:
        """Initialize a new evaluation.

        Args:
            name (str | None, optional): Name of the evaluation. Defaults to None.
            group_name (str | None, optional): Group name for the evaluation. Defaults to None.
            metadata (dict[str, Any] | None, optional): Metadata to associate with. Defaults to None.

        Returns:
            InitEvaluationResponse: The response from the initialization request.
        """
        response = self._client.post(
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

    def create_evaluation(
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
        evaluation = self.init(name=name, group_name=group_name, metadata=metadata)
        return evaluation.id

    def create_datapoint(
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

        self.save_datapoints(eval_id, [partial_datapoint])
        return datapoint_id

    def save_datapoints(
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
        length = INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH
        points = [datapoint.to_dict(max_data_length=length) for datapoint in datapoints]
        response = self._client.post(
            self._base_url + f"/v1/evals/{eval_id}/datapoints",
            json={
                "points": points,
                "groupName": group_name,
            },
            headers=self._headers(),
        )
        if response.status_code == 413:
            self._retry_save_datapoints(eval_id, datapoints, group_name)
            return

        if response.status_code != 200:
            raise ValueError(
                f"Error saving evaluation datapoints: [{response.status_code}] {response.text}"
            )

    def update_datapoint(
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

        response = self._client.post(
            self._base_url + f"/v1/evals/{eval_id}/datapoints/{datapoint_id}",
            json={
                "executorOutput": (
                    str(serialize(executor_output))[
                        :INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH
                    ]
                    if executor_output is not None
                    else None
                ),
                "scores": scores,
            },
            headers=self._headers(),
        )

        if response.status_code != 200:
            raise ValueError(f"Error updating evaluation datapoint: {response.text}")

    def get_datapoints(
        self,
        dataset_name: str,
        offset: int,
        limit: int,
    ) -> GetDatapointsResponse:
        """Get datapoints from a dataset.

        Args:
            dataset_name (str): The name of the dataset.
            offset (int): The offset to start from.
            limit (int): The maximum number of datapoints to return.

        Returns:
            GetDatapointsResponse: The response containing the datapoints.

        Raises:
            ValueError: If there's an error fetching the datapoints.
        """

        warnings.warn(
            "Use client.datasets.pull instead",
            DeprecationWarning,
        )

        params = {"name": dataset_name, "offset": offset, "limit": limit}
        response = self._client.get(
            self._base_url + "/v1/datasets/datapoints",
            params=params,
            headers=self._headers(),
        )
        if response.status_code != 200:
            try:
                resp_json = response.json()
                raise ValueError(
                    f"Error fetching datapoints: [{response.status_code}] {resp_json}"
                )
            except Exception:
                raise ValueError(
                    f"Error fetching datapoints: [{response.status_code}] {response.text}"
                )
        return GetDatapointsResponse.model_validate(response.json())

    def _retry_save_datapoints(
        self,
        eval_id: uuid.UUID,
        datapoints: list[EvaluationResultDatapoint | PartialEvaluationDatapoint],
        group_name: str | None = None,
        initial_length: int = INITIAL_EVALUATION_DATAPOINT_MAX_DATA_LENGTH,
        max_retries: int = 20,
    ):
        retry = 0
        length = initial_length
        while retry < max_retries:
            retry += 1
            length = length // 2
            logger.debug(
                f"Retrying save datapoints: {retry} of {max_retries}, length: {length}"
            )
            if length == 0:
                raise ValueError("Error saving evaluation datapoints")
            points = [
                datapoint.to_dict(max_data_length=length) for datapoint in datapoints
            ]
            response = self._client.post(
                self._base_url + f"/v1/evals/{eval_id}/datapoints",
                json={
                    "points": points,
                    "groupName": group_name,
                },
                headers=self._headers(),
            )
            if response.status_code != 413:
                break
        if response.status_code != 200:
            raise ValueError(
                f"Error saving evaluation datapoints: [{response.status_code}] {response.text}"
            )

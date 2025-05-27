"""Evals resource for interacting with Laminar evaluations API."""

import uuid
import urllib.parse

from lmnr.sdk.client.synchronous.resources.base import BaseResource
from lmnr.sdk.types import (
    InitEvaluationResponse,
    EvaluationResultDatapoint,
    PartialEvaluationDatapoint,
    GetDatapointsResponse,
)


class Evals(BaseResource):
    """Resource for interacting with Laminar evaluations API."""

    def init(
        self, name: str | None = None, group_name: str | None = None
    ) -> InitEvaluationResponse:
        """Initialize a new evaluation.

        Args:
            name (str | None, optional): Name of the evaluation. Defaults to None.
            group_name (str | None, optional): Group name for the evaluation. Defaults to None.

        Returns:
            InitEvaluationResponse: The response from the initialization request.
        """
        response = self._client.post(
            self._base_url + "/v1/evals",
            json={
                "name": name,
                "groupName": group_name,
            },
            headers=self._headers(),
        )
        resp_json = response.json()
        return InitEvaluationResponse.model_validate(resp_json)

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
        response = self._client.post(
            self._base_url + f"/v1/evals/{eval_id}/datapoints",
            json={
                "points": [datapoint.to_dict() for datapoint in datapoints],
                "groupName": group_name,
            },
            headers=self._headers(),
        )
        if response.status_code != 200:
            raise ValueError(f"Error saving evaluation datapoints: {response.text}")

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
        params = {"name": dataset_name, "offset": offset, "limit": limit}
        url = (
            self._base_url + "/v1/datasets/datapoints?" + urllib.parse.urlencode(params)
        )
        response = self._client.get(url, headers=self._headers())
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

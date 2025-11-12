"""Datasets resource for interacting with Laminar datasets API."""

import math
import uuid

from lmnr.sdk.client.synchronous.resources.base import BaseResource
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import (
    Datapoint,
    Dataset,
    GetDatapointsResponse,
    PushDatapointsResponse,
)
from lmnr.sdk.utils import serialize

logger = get_default_logger(__name__)

DEFAULT_DATASET_PULL_LIMIT = 100
DEFAULT_DATASET_PUSH_BATCH_SIZE = 100


class Datasets(BaseResource):
    """Resource for interacting with Laminar datasets API."""

    def list_datasets(self) -> list[Dataset]:
        """List all datasets."""
        response = self._client.get(
            f"{self._base_url}/v1/datasets",
            headers=self._headers(),
        )
        if response.status_code != 200:
            raise ValueError(
                f"Error listing datasets: [{response.status_code}] {response.text}"
            )
        return [Dataset.model_validate(dataset) for dataset in response.json()]

    def get_dataset_by_name(self, name: str) -> list[Dataset]:
        """Get a dataset by name."""
        response = self._client.get(
            f"{self._base_url}/v1/datasets",
            params={"name": name},
            headers=self._headers(),
        )
        if response.status_code != 200:
            raise ValueError(
                f"Error getting dataset: [{response.status_code}] {response.text}"
            )
        return [Dataset.model_validate(dataset) for dataset in response.json()]

    def push(
        self,
        points: list[Datapoint],
        name: str | None = None,
        id: uuid.UUID | None = None,
        batch_size: int = DEFAULT_DATASET_PUSH_BATCH_SIZE,
        create_dataset: bool = False,
    ) -> PushDatapointsResponse | None:
        """Push data to a dataset."""

        if name is None and id is None:
            raise ValueError("Either name or id must be provided")

        if name is not None and id is not None:
            raise ValueError("Only one of name or id must be provided")

        if create_dataset and name is None:
            raise ValueError("Name must be provided when creating a new dataset")

        identifier = {"name": name} if name is not None else {"datasetId": id}

        batch_num = 0
        total_batches = math.ceil(len(points) / batch_size)
        response = None
        for i in range(0, len(points), batch_size):
            batch_num += 1
            logger.debug(f"Pushing batch {batch_num} of {total_batches}")
            batch = points[i : i + batch_size]
            response = self._client.post(
                f"{self._base_url}/v1/datasets/datapoints",
                json={
                    **identifier,
                    "datapoints": [serialize(point) for point in batch],
                    "createDataset": create_dataset,
                },
                headers=self._headers(),
            )

            # 201 when creating a new dataset
            if response.status_code not in [200, 201]:
                raise ValueError(
                    f"Error pushing data to dataset: [{response.status_code}] {response.text}"
                )

            response = PushDatapointsResponse.model_validate(response.json())
        # Currently, the response only contains the dataset ID,
        # so it's safe to return the last response only.
        return response

    def pull(
        self,
        name: str | None = None,
        id: uuid.UUID | None = None,
        # TODO: move const to one file, import in CLI
        limit: int = DEFAULT_DATASET_PULL_LIMIT,
        offset: int = 0,
    ) -> GetDatapointsResponse:
        """Pull data from a dataset."""

        if name is None and id is None:
            raise ValueError("Either name or id must be provided")

        if name is not None and id is not None:
            raise ValueError("Only one of name or id must be provided")

        identifier = {"name": name} if name is not None else {"datasetId": id}

        params = {
            **identifier,
            "offset": offset,
            "limit": limit,
        }
        response = self._client.get(
            f"{self._base_url}/v1/datasets/datapoints",
            params=params,
            headers=self._headers(),
        )
        if response.status_code != 200:
            raise ValueError(
                f"Error pulling data from dataset: [{response.status_code}] {response.text}"
            )
        return GetDatapointsResponse.model_validate(response.json())

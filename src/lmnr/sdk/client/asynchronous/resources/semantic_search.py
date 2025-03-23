"""SemanticSearch resource for interacting with Laminar semantic search API."""

import uuid
from typing import Optional

from lmnr.sdk.client.asynchronous.resources.base import BaseAsyncResource
from lmnr.sdk.types import (
    SemanticSearchRequest,
    SemanticSearchResponse,
)


class SemanticSearch(BaseAsyncResource):
    """Resource for interacting with Laminar semantic search API."""

    async def search(
        self,
        query: str,
        dataset_id: uuid.UUID,
        limit: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> SemanticSearchResponse:
        """Perform a semantic search on the given dataset.

        Args:
            query (str): query to search for
            dataset_id (uuid.UUID): dataset ID created in the UI
            limit (Optional[int], optional): maximum number of results to return
            threshold (Optional[float], optional): lowest similarity score to return

        Raises:
            ValueError: if an error happens while performing the semantic search

        Returns:
            SemanticSearchResponse: response from the semantic search
        """
        request = SemanticSearchRequest(
            query=query,
            dataset_id=dataset_id,
            limit=limit,
            threshold=threshold,
        )
        response = await self._client.post(
            self._base_url + "/v1/semantic-search",
            json=request.to_dict(),
            headers=self._headers(),
        )
        if response.status_code != 200:
            raise ValueError(
                f"Error performing semantic search: [{response.status_code}] {response.text}"
            )
        try:
            resp_json = response.json()
            for result in resp_json["results"]:
                result["dataset_id"] = uuid.UUID(result["datasetId"])
            return SemanticSearchResponse(**resp_json)
        except Exception as e:
            raise ValueError(
                f"Error parsing semantic search response: status={response.status_code} error={e}"
            )

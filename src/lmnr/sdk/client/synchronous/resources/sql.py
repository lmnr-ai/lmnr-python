"""SQL query resource for synchronous client."""

from typing import Any

from lmnr.sdk.client.synchronous.resources.base import BaseResource
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class Sql(BaseResource):
    """
    SQL query resource.

    Provides methods for executing SQL queries against the Laminar backend.
    """

    def query(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a SQL query against the Laminar backend.

        Args:
            sql: SQL query string with parameter placeholders
            parameters: Optional parameters to substitute in the query

        Returns:
            List[dict[str, Any]]: Query results as list of dictionaries

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        payload = {
            "query": sql,
            "parameters": parameters or {},
        }

        response = self._client.post(
            f"{self._base_url}/v1/sql/query",
            headers=self._headers(),
            json=payload,
        )
        response.raise_for_status()

        result = response.json()
        return result.get("data", [])

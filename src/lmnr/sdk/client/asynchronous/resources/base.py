"""Base class for resource objects."""

import httpx


class BaseAsyncResource:
    """Base class for all API resources."""

    def __init__(self, client: httpx.AsyncClient, base_url: str, project_api_key: str):
        """Initialize the resource.

        Args:
            client (httpx.AsyncClient): HTTP client instance
            base_url (str): Base URL for the API
            project_api_key (str): Project API key
        """
        self._client = client
        self._base_url = base_url
        self._project_api_key = project_api_key

    def _headers(self) -> dict[str, str]:
        """Generate request headers with authentication.

        Returns:
            dict[str, str]: Headers dictionary
        """
        assert self._project_api_key is not None, "Project API key is not set"
        return {
            "Authorization": "Bearer " + self._project_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

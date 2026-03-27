"""MCP-specific client wrapper around AsyncLaminarClient."""

from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.utils import from_env


class LaminarMcpClient:
    """Manages an AsyncLaminarClient instance for the MCP server.

    This thin wrapper lazily creates and reuses an AsyncLaminarClient,
    handling default configuration from environment variables.
    """

    def __init__(
        self,
        project_api_key: str | None = None,
        base_url: str | None = None,
        port: int | None = None,
    ):
        self._project_api_key = project_api_key or from_env("LMNR_PROJECT_API_KEY")
        self._base_url = base_url or from_env("LMNR_BASE_URL") or "https://api.lmnr.ai"
        self._port = port
        self._client: AsyncLaminarClient | None = None

    async def get_client(self) -> AsyncLaminarClient:
        """Get or create the underlying AsyncLaminarClient.

        Returns:
            AsyncLaminarClient: The configured client instance.

        Raises:
            ValueError: If project API key is not set.
        """
        if self._client is None or self._client.is_closed:
            self._client = AsyncLaminarClient(
                project_api_key=self._project_api_key,
                base_url=self._base_url,
                port=self._port,
            )
        return self._client

    async def close(self):
        """Close the underlying client if it exists and is open."""
        if self._client is not None and not self._client.is_closed:
            await self._client.close()

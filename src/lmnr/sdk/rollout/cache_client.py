"""
HTTP client for communicating with the cache server.

The cache client provides methods to fetch cached spans, path-to-count mappings,
and overrides from the cache server.
"""

from typing import Any

import httpx

from lmnr.sdk.log import get_default_logger
from lmnr.sdk.rollout.types import (
    CacheServerResponse,
    CachedSpan,
)

logger = get_default_logger(__name__)


class CacheClient:
    """
    Client for communicating with the rollout cache server.

    This client is used by instrumentations to check for cached responses
    and fetch override parameters.
    """

    def __init__(self, cache_server_url: str, timeout: float = 5.0):
        """
        Initialize the cache client.

        Args:
            cache_server_url: Base URL of the cache server (e.g., "http://localhost:12345")
            timeout: Request timeout in seconds (default: 5.0)
        """
        self.base_url = cache_server_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.Client | None = None
        self._path_to_count_cache: dict[str, int] | None = None
        self._overrides_cache: dict[str, Any] | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def get_cached_span(self, path: str, index: int) -> CachedSpan | None:
        """
        Fetch a cached span by path and index.

        Args:
            path: Span path (e.g., "root.llm_call")
            index: Call index for this path (e.g., 0, 1, 2)

        Returns:
            Optional[Dict[str, Any]]: Cached span data if found, None otherwise
        """
        try:
            client = self._get_client()
            response = client.post(
                f"{self.base_url}/cached",
                json={"path": path, "index": index},
            )

            if response.status_code == 200:
                data: CacheServerResponse = response.json()

                # Update cached metadata
                if "pathToCount" in data:
                    self._path_to_count_cache = data["pathToCount"]
                if "overrides" in data:
                    self._overrides_cache = data["overrides"]

                # Return the span if it exists
                span = data.get("span")
                if span:
                    logger.debug(f"Cache hit for {path}:{index}")
                    return span
                else:
                    logger.debug(f"Cache miss for {path}:{index}")
                    return None
            else:
                logger.warning(
                    f"Cache server returned status {response.status_code} "
                    f"for {path}:{index}"
                )
                return None

        except httpx.TimeoutException:
            logger.warning(f"Timeout fetching cached span {path}:{index}")
            return None
        except Exception as e:
            logger.warning(
                f"Error fetching cached span {path}:{index}: {e}", exc_info=True
            )
            return None

    def get_path_to_count(self) -> dict[str, int]:
        """
        Fetch the path-to-count mapping from cache server.

        Returns:
            Dict[str, int]: Mapping of span paths to cache counts.
                          Returns cached value or empty dict on error.
        """
        # Return cached value if available
        if self._path_to_count_cache is not None:
            return self._path_to_count_cache

        # Fetch metadata by requesting any cached span (with dummy values)
        # The cache server returns metadata with every /cached request
        try:
            client = self._get_client()
            response = client.post(
                f"{self.base_url}/cached",
                json={"path": "__metadata__", "index": 0},
            )

            if response.status_code == 200:
                data: CacheServerResponse = response.json()
                # Extract pathToCount from response
                path_to_count = data.get("pathToCount", {})
                self._path_to_count_cache = path_to_count
                logger.debug(f"Fetched path_to_count: {path_to_count}")
                return path_to_count
            else:
                logger.debug(
                    f"Cache server returned status {response.status_code} "
                    "when fetching metadata"
                )
                return {}

        except httpx.TimeoutException:
            logger.debug("Timeout fetching path_to_count metadata")
            return {}
        except Exception as e:
            logger.debug(f"Error fetching path_to_count metadata: {e}")
            return {}

    def get_overrides(self) -> dict[str, Any]:
        """
        Fetch override parameters from cache server.

        Returns:
            Dict[str, Any]: Override parameters (e.g., model, temperature).
                           Returns cached value or empty dict on error.
        """
        # Return cached value if available
        if self._overrides_cache is not None:
            return self._overrides_cache

        # Fetch metadata by requesting any cached span (with dummy values)
        # The cache server returns metadata with every /cached request
        try:
            client = self._get_client()
            response = client.post(
                f"{self.base_url}/cached",
                json={"path": "__metadata__", "index": 0},
            )

            if response.status_code == 200:
                data: CacheServerResponse = response.json()
                # Extract overrides from response
                overrides = data.get("overrides", {})
                self._overrides_cache = overrides
                logger.debug(f"Fetched overrides: {overrides}")
                return overrides
            else:
                logger.debug(
                    f"Cache server returned status {response.status_code} "
                    "when fetching metadata"
                )
                return {}

        except httpx.TimeoutException:
            logger.debug("Timeout fetching overrides metadata")
            return {}
        except Exception as e:
            logger.debug(f"Error fetching overrides metadata: {e}")
            return {}

    def should_use_cache(self, path: str, current_index: int) -> bool:
        """
        Check if a span at given path and index should use cache.

        Args:
            path: Span path
            current_index: Current call index for this path

        Returns:
            bool: True if should use cache, False otherwise
        """
        path_to_count = self.get_path_to_count()
        cache_limit = path_to_count.get(path, 0)
        return current_index < cache_limit

    def invalidate_cache(self) -> None:
        """
        Invalidate cached metadata.

        This forces the next calls to fetch fresh data from the cache server.
        """
        self._path_to_count_cache = None
        self._overrides_cache = None
        logger.debug("Cache invalidated")

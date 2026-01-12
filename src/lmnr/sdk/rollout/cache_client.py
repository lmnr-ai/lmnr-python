"""
HTTP client for communicating with the cache server.

The cache client provides methods to fetch cached spans, path-to-count mappings,
and overrides from the cache server.
"""

from typing import Any, Dict, Optional

import httpx

from lmnr.sdk.log import get_default_logger

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
        self._client: Optional[httpx.Client] = None
        self._path_to_count_cache: Optional[Dict[str, int]] = None
        self._overrides_cache: Optional[Dict[str, Any]] = None

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

    def get_cached_span(self, path: str, index: int) -> Optional[Dict[str, Any]]:
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
                data = response.json()

                # Update cached metadata
                if "path_to_count" in data:
                    self._path_to_count_cache = data["path_to_count"]
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

    def get_path_to_count(self) -> Dict[str, int]:
        """
        Fetch the path-to-count mapping from cache server.

        Returns:
            Dict[str, int]: Mapping of span paths to cache counts.
                          Returns cached value or empty dict on error.
        """
        # Return cached value if available
        if self._path_to_count_cache is not None:
            return self._path_to_count_cache

        try:
            client = self._get_client()
            response = client.get(f"{self.base_url}/path_to_count")

            if response.status_code == 200:
                self._path_to_count_cache = response.json()
                logger.debug(f"Fetched path_to_count: {self._path_to_count_cache}")
                return self._path_to_count_cache
            else:
                logger.warning(
                    f"Cache server returned status {response.status_code} "
                    "for path_to_count"
                )
                return {}

        except httpx.TimeoutException:
            logger.warning("Timeout fetching path_to_count")
            return {}
        except Exception as e:
            logger.warning(f"Error fetching path_to_count: {e}", exc_info=True)
            return {}

    def get_overrides(self) -> Dict[str, Any]:
        """
        Fetch override parameters from cache server.

        Returns:
            Dict[str, Any]: Override parameters (e.g., model, temperature).
                           Returns cached value or empty dict on error.
        """
        # Return cached value if available
        if self._overrides_cache is not None:
            return self._overrides_cache

        try:
            client = self._get_client()
            response = client.get(f"{self.base_url}/overrides")

            if response.status_code == 200:
                self._overrides_cache = response.json()
                logger.debug(f"Fetched overrides: {self._overrides_cache}")
                return self._overrides_cache
            else:
                logger.warning(
                    f"Cache server returned status {response.status_code} "
                    "for overrides"
                )
                return {}

        except httpx.TimeoutException:
            logger.warning("Timeout fetching overrides")
            return {}
        except Exception as e:
            logger.warning(f"Error fetching overrides: {e}", exc_info=True)
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

    def update_path_to_count(self, path_to_count: Dict[str, int]) -> bool:
        """
        Update the path-to-count mapping on the cache server.

        Args:
            path_to_count: New path-to-count mapping

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            client = self._get_client()
            response = client.post(
                f"{self.base_url}/path_to_count",
                json=path_to_count,
            )

            if response.status_code == 200:
                self._path_to_count_cache = path_to_count
                logger.debug(f"Updated path_to_count: {path_to_count}")
                return True
            else:
                logger.warning(
                    f"Cache server returned status {response.status_code} "
                    "when updating path_to_count"
                )
                return False

        except Exception as e:
            logger.warning(f"Error updating path_to_count: {e}", exc_info=True)
            return False

    def update_overrides(self, overrides: Dict[str, Any]) -> bool:
        """
        Update override parameters on the cache server.

        Args:
            overrides: New override parameters

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            client = self._get_client()
            response = client.post(
                f"{self.base_url}/overrides",
                json=overrides,
            )

            if response.status_code == 200:
                self._overrides_cache = overrides
                logger.debug(f"Updated overrides: {overrides}")
                return True
            else:
                logger.warning(
                    f"Cache server returned status {response.status_code} "
                    "when updating overrides"
                )
                return False

        except Exception as e:
            logger.warning(f"Error updating overrides: {e}", exc_info=True)
            return False

    def update_spans(self, spans: Dict[str, Dict[str, Any]]) -> bool:
        """
        Update cached spans on the cache server.

        Args:
            spans: Dictionary mapping cache keys (e.g., "path:index") to span data

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            client = self._get_client()
            response = client.post(
                f"{self.base_url}/spans",
                json=spans,
            )

            if response.status_code == 200:
                logger.debug(f"Updated {len(spans)} cached spans")
                return True
            else:
                logger.warning(
                    f"Cache server returned status {response.status_code} "
                    "when updating spans"
                )
                return False

        except Exception as e:
            logger.warning(f"Error updating spans: {e}", exc_info=True)
            return False

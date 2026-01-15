"""
Base instrumentation wrapper for rollout mode.

This module provides a base class that instrumentations can extend to add
caching and override capabilities during rollout sessions.
"""

import os
from typing import Any

from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.rollout_control import is_rollout_mode
from lmnr.sdk.rollout.cache_client import CacheClient


logger = get_default_logger(__name__)


class RolloutInstrumentationWrapper:
    """
    Base class for instrumentation wrappers in rollout mode.

    Provides common functionality for:
    - Checking if rollout mode is active
    - Getting current span path
    - Tracking call indices per path
    - Fetching cached responses
    - Getting override parameters
    """

    def __init__(self):
        """Initialize the rollout wrapper with lazy-loaded cache client."""
        self._cache_client: Any | None = None
        self._initialized = False
        self._path_to_index: dict[str, int] = {}
        self._path_to_count: dict[str, int] = {}
        self._overrides: dict[str, Any] = {}

    def _get_cache_client(self) -> CacheClient | None:
        """Get or create the cache client."""
        if self._cache_client is None:
            cache_server_url = os.getenv("LMNR_ROLLOUT_STATE_SERVER_ADDRESS")
            if not cache_server_url:
                logger.warning(
                    "LMNR_ROLLOUT_STATE_SERVER_ADDRESS not set, cache will not work"
                )
                return None

            self._cache_client = CacheClient(cache_server_url)
            logger.debug(f"Created cache client for {cache_server_url}")

        return self._cache_client

    def _ensure_initialized(self) -> None:
        """
        Ensure metadata is initialized by fetching from cache server.
        Only runs once per instance.
        """
        if self._initialized:
            return

        client = self._get_cache_client()
        if client:
            client.invalidate_cache()
            # Fetch initial metadata by making a dummy request
            # This populates the cache client's internal cache from server response
            client.get_cached_span("", 0)

            # Now fetch the metadata (will use cache client's internal cache)
            self._path_to_count = client.get_path_to_count()
            self._overrides = client.get_overrides()
            logger.debug(f"Initialized with path_to_count: {self._path_to_count}")
            logger.debug(
                f"Initialized with overrides keys: {list(self._overrides.keys())}"
            )

        self._initialized = True

    def should_use_rollout(self) -> bool:
        """
        Check if rollout mode is active.

        Returns:
            bool: True if in rollout mode, False otherwise
        """
        return is_rollout_mode()

    def get_span_path(self) -> str | None:
        """
        Get the current span path from Laminar context.

        Returns:
            Optional[str]: Dot-separated span path (e.g., "root.llm_call") or None
        """
        try:
            current_span = Laminar.get_current_span()
            if not current_span:
                return None

            span_context = current_span.get_laminar_span_context()
            if not span_context or not span_context.span_path:
                return None

            # Convert list to dot-separated string
            path = ".".join(span_context.span_path)
            return path if path else None

        except Exception as e:
            logger.debug(f"Failed to get span path: {e}", exc_info=True)
            return None

    def get_current_index_for_path(self, path: str) -> int:
        """
        Get the current call index for a given path and increment it.

        Args:
            path: Span path

        Returns:
            int: Current index (0-based) before incrementing
        """
        current_index = self._path_to_index.get(path, 0)
        self._path_to_index[path] = current_index + 1
        return current_index

    def should_use_cache(self, path: str, index: int) -> bool:
        """
        Check if a span at given path and index should use cache.

        Args:
            path: Span path
            index: Call index for this path

        Returns:
            bool: True if should use cache, False otherwise
        """
        # Ensure initialized to get path_to_count
        self._ensure_initialized()

        cache_limit = self._path_to_count.get(path, 0)
        should_cache = index < cache_limit

        logger.debug(
            f"Cache check for {path}:{index} - limit={cache_limit}, should_cache={should_cache}"
        )
        return should_cache

    def get_cached_response(self, path: str, index: int) -> dict[str, Any] | None:
        """
        Fetch cached span data from cache server.

        Args:
            path: Span path
            index: Call index for this path

        Returns:
            Optional[Dict[str, Any]]: Cached span data if found, None otherwise
        """
        self._ensure_initialized()

        client = self._get_cache_client()
        if not client:
            return None

        cached_data = client.get_cached_span(path, index)
        if cached_data:
            # Update metadata in case it changed
            self._path_to_count = client.get_path_to_count()
            self._overrides = client.get_overrides()

        return cached_data

    def get_overrides(self, path: str | None = None) -> dict[str, Any]:
        """
        Get override parameters for a given path.

        Args:
            path: Span path. If None, returns all overrides.

        Returns:
            Dict[str, Any]: Override parameters
        """
        self._ensure_initialized()

        if path is None:
            return self._overrides

        # Overrides are keyed by path
        return self._overrides.get(path, {})

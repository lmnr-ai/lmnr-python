"""In-process replay cache + the replay decision.

Cache key is (span_path, occurrence_index) (§6). Only the spine path is
populated, only for the first N occurrences. Replaces the old dev-server
`_path_to_index` + HTTP `POST /cached` machinery.
"""

from typing import Any

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class ReplayCache:
    """Holds cached spine responses, indexed by occurrence on the spine path."""

    def __init__(
        self,
        spine_path: str,
        cache_until: int,
        payloads: list[dict[str, Any]],
    ):
        # spine_calls[:N] payloads, indexed by occurrence on the spine path.
        self._spine_path = spine_path
        self._cache_until = cache_until
        self._payloads = payloads[:cache_until]

    @property
    def spine_path(self) -> str:
        return self._spine_path

    @property
    def cache_until(self) -> int:
        return self._cache_until

    def get_cached(self, span_path: str, occurrence: int) -> dict[str, Any] | None:
        """Return the cached payload to replay, or None to run live (§8)."""
        if span_path != self._spine_path:
            return None
        if occurrence >= self._cache_until:
            return None
        if occurrence >= len(self._payloads):
            return None
        return self._payloads[occurrence]

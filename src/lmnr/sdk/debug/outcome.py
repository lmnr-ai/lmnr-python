"""The three outcomes of a v2 debugger cache lookup (shared spec §7.2).

A single small value the `RolloutSessions.cache` resource returns and the
provider wrappers branch on:

- ``hit``  — the warm cache had this input hash; ``cached`` carries the recorded
  output (CachedSpan-shaped) to serve in place of the live call.
- ``miss`` — the cache is warm but this hash is absent; the SDK latches the
  process-wide "run live" flag and stops calling the endpoint.
- ``live`` — warmup timed out (or a transport error); run THIS call live only,
  do NOT latch the flag, retry the endpoint next call.
"""

from dataclasses import dataclass
from typing import Any, Literal, cast

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


@dataclass(frozen=True)
class CacheOutcome:
    kind: Literal["hit", "miss", "live"]
    cached: dict[str, Any] | None = None  # pyright: ignore[reportExplicitAny]


def parse_cache_outcome(data: object) -> CacheOutcome:
    """Map app-server's `{outcome: hit|miss|live, response?}` body to CacheOutcome.

    Shared by the sync and async resources. Anything unrecognized degrades to `live` (safe).
    """
    outcome = data.get("outcome") if isinstance(data, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if outcome == "hit":
        # A HIT must carry a response envelope to be servable; the provider
        # wrappers call cached_response_to_*(cached), which does cached.get().
        # A response-less HIT (omitted/null `response`) is malformed — degrade
        # to `live` so the call runs live (no latch) instead of raising.
        data_dict = cast(dict[str, Any], data)  # pyright: ignore[reportExplicitAny]
        response = data_dict.get("response")
        if response is None:
            logger.debug("Cache HIT without response body; running call live")
            return CacheOutcome(kind="live")
        return CacheOutcome(kind="hit", cached=response) # pyright: ignore[reportAny])
    if outcome == "miss":
        return CacheOutcome(kind="miss")
    return CacheOutcome(kind="live")

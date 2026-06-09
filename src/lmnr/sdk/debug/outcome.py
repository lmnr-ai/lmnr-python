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
from typing import Any, Literal


@dataclass(frozen=True)
class CacheOutcome:
    kind: Literal["hit", "miss", "live"]
    cached: dict[str, Any] | None = None

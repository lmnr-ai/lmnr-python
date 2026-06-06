"""Debug (rollout) session registration resource for the synchronous client."""

import uuid

from lmnr.sdk.client.synchronous.resources.base import BaseResource
from lmnr.sdk.debug.outcome import CacheOutcome
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class RolloutSessions(BaseResource):
    """Register / delete debug sessions on the backend.

    A debug run owns its session id; `register` is an idempotent upsert that
    makes the session visible in the UI. This is what turns a bare
    `LMNR_DEBUG=true` run (no replay) into something useful — without it the
    backend never learns the session id the SDK minted.
    """

    def register(
        self, session_id: uuid.UUID | str, name: str | None = None
    ) -> str | None:
        """Idempotently register (upsert) a debug session.

        A null/omitted `name` never clobbers a name set elsewhere (e.g. the UI).

        Returns the backend-resolved `projectId` (derived from the API key) so
        the caller can build the debugger URL; None if the body can't be parsed.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = self._client.post(
            f"{self._base_url}/v1/rollouts/{session_id}",
            headers=self._headers(),
            json={"name": name},
        )
        response.raise_for_status()
        try:
            return response.json().get("projectId")
        except Exception:
            return None

    def delete(self, session_id: uuid.UUID | str) -> None:
        """Delete a debug session.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = self._client.delete(
            f"{self._base_url}/v1/rollouts/{session_id}",
            headers=self._headers(),
        )
        response.raise_for_status()

    def cache(
        self,
        session_id: uuid.UUID | str,
        replay_trace_id: uuid.UUID | str,
        cache_until: str,
        input_hash: str,
    ) -> CacheOutcome:
        """Look up one LLM call's input hash in the server-side replay cache.

        POSTs to `/v1/rollouts/{session_id}/cache` and maps app-server's
        discriminated `{outcome}` response to a `CacheOutcome` (shared spec §7).

        Never raises: a non-2xx or transport error degrades to `kind="live"`
        (run this call live, retry next call) so a replay miss can never take
        down the user's program. Only a real MISS latches the static run-live
        flag — that's the caller's job, not this method's.
        """
        try:
            response = self._client.post(
                f"{self._base_url}/v1/rollouts/{session_id}/cache",
                headers=self._headers(),
                json={
                    "replayTraceId": str(replay_trace_id),
                    "cacheUntil": cache_until,
                    "inputHash": input_hash,
                },
            )
            if response.status_code // 100 != 2:
                logger.debug(
                    "Cache lookup returned HTTP %s; running this call live",
                    response.status_code,
                )
                return CacheOutcome(kind="live")
            data = response.json()
        except Exception as exc:
            logger.debug("Cache lookup failed (%s); running this call live", exc)
            return CacheOutcome(kind="live")
        return _parse_cache_outcome(data)


def _parse_cache_outcome(data: object) -> CacheOutcome:
    """Map app-server's `{outcome: hit|miss|live, response?}` body to CacheOutcome.

    Shared by the sync and async resources (kept here to avoid duplicating the
    parse). Anything unrecognized degrades to `live` (safe).
    """
    outcome = data.get("outcome") if isinstance(data, dict) else None
    if outcome == "hit":
        return CacheOutcome(kind="hit", cached=data.get("response"))
    if outcome == "miss":
        return CacheOutcome(kind="miss")
    return CacheOutcome(kind="live")

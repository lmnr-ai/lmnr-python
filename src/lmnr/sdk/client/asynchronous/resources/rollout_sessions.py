"""Debug (rollout) session registration resource for the asynchronous client."""

import uuid

from lmnr.sdk.client.asynchronous.resources.base import BaseAsyncResource
from lmnr.sdk.client.synchronous.resources.rollout_sessions import (
    _parse_cache_outcome,
)
from lmnr.sdk.debug.outcome import CacheOutcome
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class AsyncRolloutSessions(BaseAsyncResource):
    """Register / delete debug sessions on the backend.

    A debug run owns its session id; `register` is an idempotent upsert that
    makes the session visible in the UI. This is what turns a bare
    `LMNR_DEBUG=true` run (no replay) into something useful — without it the
    backend never learns the session id the SDK minted.
    """

    async def register(
        self, session_id: uuid.UUID | str, name: str | None = None
    ) -> str | None:
        """Idempotently register (upsert) a debug session.

        A null/omitted `name` never clobbers a name set elsewhere (e.g. the UI).

        Returns the backend-resolved `projectId` (derived from the API key) so
        the caller can build the debugger URL; None if the body can't be parsed.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = await self._client.post(
            f"{self._base_url}/v1/rollouts/{session_id}",
            headers=self._headers(),
            json={"name": name},
        )
        response.raise_for_status()
        try:
            return response.json().get("projectId")
        except Exception:
            return None

    async def delete(self, session_id: uuid.UUID | str) -> None:
        """Delete a debug session.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = await self._client.delete(
            f"{self._base_url}/v1/rollouts/{session_id}",
            headers=self._headers(),
        )
        response.raise_for_status()

    async def cache(
        self,
        session_id: uuid.UUID | str,
        replay_trace_id: uuid.UUID | str,
        cache_until: str,
        input_hash: str,
    ) -> CacheOutcome:
        """Async variant of `RolloutSessions.cache` (shared spec §7).

        Same swallow-and-degrade posture: never raises, degrades to
        `kind="live"` on any non-2xx / transport error.
        """
        try:
            response = await self._client.post(
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

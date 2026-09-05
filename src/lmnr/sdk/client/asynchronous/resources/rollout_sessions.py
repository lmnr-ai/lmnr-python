"""Debug (rollout) session registration resource for the asynchronous client."""

import uuid
from typing import cast

from lmnr.sdk.client.asynchronous.resources.base import BaseAsyncResource
from lmnr.sdk.debug.outcome import CacheOutcome, parse_cache_outcome
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import SessionBlock, SessionBlockContent, SessionBlockType

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
        response = response.raise_for_status()
        try:
            response_dict = cast(dict[str, str | None], response.json())
            return response_dict.get("projectId")
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
        response = response.raise_for_status()

    async def add_block(
        self,
        session_id: uuid.UUID | str,
        type: SessionBlockType,
        content: SessionBlockContent,
        fail_on_not_found: bool = False,
    ) -> str | None:
        """Async variant of `RolloutSessions.add_block`.

        Same 404 posture: logged and swallowed unless `fail_on_not_found` is
        set, in which case it raises. Any other non-OK status raises.

        Raises:
            httpx.HTTPStatusError: If the request fails (non-404).
        """
        response = await self._client.post(
            f"{self._base_url}/v1/rollouts/{session_id}/blocks",
            headers=self._headers(),
            json={"type": type, "content": content},
        )
        if response.status_code == 404:
            message = (
                f"Could not add a note: HTTP 404 for session {session_id}. "
                "Either the session isn't registered in this project (mint one "
                "with `lmnr-cli debug session new`, or run under LMNR_DEBUG=1), "
                "or this Laminar server doesn't expose the session-blocks write "
                "endpoint (POST /v1/rollouts/{sessionId}/blocks) yet."
            )
            if fail_on_not_found:
                raise RuntimeError(message)
            logger.warning(message)
            return None
        response = response.raise_for_status()
        try:
            response_dict = cast(dict[str, str], response.json())
            return response_dict.get("id")
        except Exception as e:
            logger.warning(f"Failed to parse add-block response: {e}")
            return None

    async def list_blocks(self, session_id: uuid.UUID | str) -> list[SessionBlock]:
        """Async variant of `RolloutSessions.list_blocks`.

        Returns every block on the session in creation order, or an empty list
        when there are none / the body can't be parsed.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = await self._client.get(
            f"{self._base_url}/v1/rollouts/{session_id}/blocks",
            headers=self._headers(),
        )
        response = response.raise_for_status()
        try:
            body = cast(list[SessionBlock] | dict[str, list[SessionBlock]], response.json())
            blocks = body if isinstance(body, list) else body.get("blocks")
            return blocks or []
        except Exception as e:
            logger.warning(f"Failed to parse list-blocks response: {e}")
            return []

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
            data = cast(object, response.json())
        except Exception as exc:
            logger.debug("Cache lookup failed (%s); running this call live", exc)
            return CacheOutcome(kind="live")
        return parse_cache_outcome(data)

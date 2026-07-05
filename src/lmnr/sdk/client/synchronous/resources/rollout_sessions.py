"""Debug (rollout) session registration resource for the synchronous client."""

import uuid

from lmnr.sdk.client.synchronous.resources.base import BaseResource
from lmnr.sdk.debug.outcome import CacheOutcome
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import SessionBlock, SessionBlockContent, SessionBlockType

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

    def add_block(
        self,
        session_id: uuid.UUID | str,
        type: SessionBlockType,
        content: SessionBlockContent,
        fail_on_not_found: bool = False,
    ) -> str | None:
        """Append a block to a debug session (debugger session blocks).

        A debug session renders as an ordered list of blocks; this writes one to
        the backend keyed by session id. The CLI uses it for `text` blocks —
        standalone agent notes attached post-factum — so a note is tied to the
        SESSION, not to a specific trace / evaluation (those blocks are written
        at ingest from `rollout.session_id` metadata).

        A 404 (the session is unknown for the project) is logged and swallowed
        unless `fail_on_not_found` is set — CLI callers pass it so an exit 0
        means the block actually landed. Any other non-OK status raises.

        Returns the created block id, or None when the response body can't be
        parsed.

        Raises:
            httpx.HTTPStatusError: If the request fails (non-404).
        """
        response = self._client.post(
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
        response.raise_for_status()
        try:
            return response.json().get("id")
        except Exception as e:
            logger.warning(f"Failed to parse add-block response: {e}")
            return None

    def list_blocks(self, session_id: uuid.UUID | str) -> list[SessionBlock]:
        """List a debug session's blocks in creation order.

        Returns every `trace` / `evaluation` / `text` block on the session — the
        same data the debugger UI renders. Returns an empty list when the
        session has no blocks or the body can't be parsed.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = self._client.get(
            f"{self._base_url}/v1/rollouts/{session_id}/blocks",
            headers=self._headers(),
        )
        response.raise_for_status()
        try:
            body = response.json()
            blocks = body if isinstance(body, list) else body.get("blocks")
            return blocks or []
        except Exception as e:
            logger.warning(f"Failed to parse list-blocks response: {e}")
            return []

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
        # A HIT must carry a response envelope to be servable; the provider
        # wrappers call cached_response_to_*(cached), which does cached.get().
        # A response-less HIT (omitted/null `response`) is malformed — degrade
        # to `live` so the call runs live (no latch) instead of raising.
        response = data.get("response")
        if response is None:
            logger.debug("Cache HIT without response body; running call live")
            return CacheOutcome(kind="live")
        return CacheOutcome(kind="hit", cached=response)
    if outcome == "miss":
        return CacheOutcome(kind="miss")
    return CacheOutcome(kind="live")

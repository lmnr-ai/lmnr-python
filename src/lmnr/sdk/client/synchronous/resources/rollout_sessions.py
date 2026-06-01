"""Debug (rollout) session registration resource for the synchronous client."""

import uuid

from lmnr.sdk.client.synchronous.resources.base import BaseResource
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

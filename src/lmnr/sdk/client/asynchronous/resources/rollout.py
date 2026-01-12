"""Rollout session management resource for asynchronous client."""

from typing import AsyncContextManager, Literal

import httpx
import uuid

from lmnr.sdk.client.asynchronous.resources.base import BaseAsyncResource
from lmnr.sdk.client.types import SpanStartData
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import RolloutParam

logger = get_default_logger(__name__)


class AsyncRollout(BaseAsyncResource):
    """
    Async rollout session management resource.

    Provides async methods for connecting to rollout sessions, streaming spans,
    and managing rollout lifecycle.
    """

    def connect(
        self,
        session_id: uuid.UUID,
        function_name: str,
        params: list[RolloutParam] | None = None,
    ) -> AsyncContextManager[httpx.Response]:
        """
        Connect to a rollout session and return the SSE streaming response.

        This creates the session and returns a streaming response that should
        be parsed for SSE events.

        Args:
            session_id: UUID for the rollout session
            function_name: Name of the function being rolled out
            params: List of parameter metadata for the function

        Returns:
            httpx.Response: Streaming response for SSE events

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        payload = {
            "name": function_name,
            "params": params or [],
        }

        headers = self._headers()
        headers["Accept"] = "text/event-stream"

        response = self._client.stream(
            "POST",
            f"{self._base_url}/v1/rollouts/{session_id}",
            headers=headers,
            json=payload,
        )
        return response

    async def update_span_info(
        self,
        session_id: str,
        span_data: SpanStartData,
    ) -> None:
        """
        Stream a span update in real-time to the backend.

        This is used during rollout execution to provide live updates
        in the Laminar UI.

        Args:
            session_id: UUID of the rollout session
            span_data: Span data to stream

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        try:
            response = await self._client.patch(
                f"{self._base_url}/v1/rollouts/{session_id}/update",
                headers=self._headers(),
                json={
                    "type": "spanStart",
                    "name": span_data["name"],
                    "spanId": str(span_data["span_id"]),
                    "parentSpanId": (
                        str(span_data["parent_span_id"])
                        if span_data["parent_span_id"]
                        else None
                    ),
                    "trace_id": str(span_data["trace_id"]),
                    "startTime": span_data["start_time"].isoformat(),
                    "attributes": span_data["attributes"],
                    "spanType": span_data["span_type"],
                },
                timeout=5.0,  # Short timeout for streaming
            )
            response.raise_for_status()
        except Exception as e:
            # Log but don't raise - streaming failures shouldn't break execution
            logger.debug(f"Failed to update span info for session {session_id}: {e}")

    async def delete_session(
        self,
        session_id: str,
    ) -> None:
        """
        Delete a rollout session.

        Args:
            session_id: UUID of the rollout session to delete

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        response = await self._client.delete(
            f"{self._base_url}/v1/rollouts/{session_id}",
            headers=self._headers(),
        )
        response.raise_for_status()

    async def update_status(
        self,
        session_id: str,
        status: Literal["PENDING", "RUNNING", "FINISHED", "STOPPED"],
    ) -> None:
        """
        Update the status of a rollout session.

        Args:
            session_id: UUID of the rollout session
            status: New status

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        response = await self._client.patch(
            f"{self._base_url}/v1/rollouts/{session_id}/status",
            headers=self._headers(),
            json={"status": status},
        )
        response.raise_for_status()

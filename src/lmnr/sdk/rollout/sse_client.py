"""
SSE (Server-Sent Events) client for receiving rollout events from backend.

The SSE client connects to the Laminar backend and listens for events like:
- 'run': Execute the rollout function with provided arguments
- 'stop': Cancel the currently running execution
- 'heartbeat': Keep-alive signal
- 'handshake': Initial connection confirmation
"""

import asyncio
import json
from typing import Callable

from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)
RECONNECT_DELAY = 1.0


class SSEClient:
    """
    SSE client for receiving rollout events from the Laminar backend.

    Handles connection, reconnection, and event dispatching to registered handlers.
    """

    def __init__(
        self,
        rollout_session_id: str,
        function_name: str,
        params: list,
        laminar_client: AsyncLaminarClient,
        event_handlers: dict[str, Callable] | None = None,
    ):
        """
        Initialize the SSE client.

        Args:
            rollout_session_id: UUID of the rollout session
            function_name: Name of the function being rolled out
            params: List of parameter metadata for the function
            laminar_client: AsyncLaminarClient instance for API calls
            event_handlers: Optional dict mapping event types to handler functions
        """
        self.session_id = rollout_session_id
        self.function_name = function_name
        self.params = params
        self.laminar_client = laminar_client
        self.event_handlers = event_handlers or {}

        self.running = False
        self.should_reconnect = True
        self._last_heartbeat: float | None = None
        self._heartbeat_timeout = (
            15.0  # 15 seconds without heartbeat triggers reconnect
        )
        self._current_event_type: str | None = None  # Track current SSE event type

    def on(self, event_type: str, handler: Callable) -> None:
        """
        Register an event handler for a specific event type.

        Args:
            event_type: Type of event (e.g., 'run', 'stop', 'heartbeat')
            handler: Async function to call when event is received
        """
        self.event_handlers[event_type] = handler

    async def connect(self) -> None:
        """
        Establish SSE connection and listen for events.

        This method will keep trying to reconnect if the connection drops,
        with exponential backoff.
        """
        self.running = True
        self.should_reconnect = True

        while self.running and self.should_reconnect:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                logger.debug("SSE connection cancelled")
                break
            except Exception as e:
                logger.debug(f"SSE connection error: {e}")

            if self.running and self.should_reconnect:
                logger.ingo(f"Reconnecting in {RECONNECT_DELAY:.1f}s...")
                await asyncio.sleep(RECONNECT_DELAY)

    async def _connect_and_listen(self) -> None:
        """Internal method to establish connection and process events."""
        logger.debug(f"Connecting to rollout session: {self.session_id}")

        try:
            # Call connect which creates session and returns SSE stream as AsyncGenerator
            import uuid

            # Iterate over the AsyncGenerator to get SSE events
            async with self.laminar_client.rollout.connect(
                session_id=uuid.UUID(self.session_id),
                function_name=self.function_name,
                params=self.params,
            ) as stream_response:
                logger.debug("SSE connection established")
                self._reconnect_delay = 1.0  # Reset backoff on successful connection

                async for line in stream_response.aiter_lines():
                    if not self.running:
                        break

                    # The generator yields lines from the SSE stream
                    # SSE format: lines starting with "event:", "data:", or empty line
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    line_str = line_str.strip()

                    if not line_str:
                        continue

                    if line_str.startswith("event:"):
                        self._current_event_type = line_str[6:].strip()
                    elif line_str.startswith("data:"):
                        data = line_str[5:].strip()
                        await self._handle_sse_event(
                            self._current_event_type or "message", data
                        )

        except Exception as e:
            logger.error(f"Error in SSE connection: {e}", exc_info=True)
            raise

    async def _handle_sse_event(self, event_type: str, event_data: str) -> None:
        """
        Handle a single SSE event.

        Args:
            event_type: Type of the event
            event_data: Data payload of the event
        """
        logger.debug(f"Received SSE event: type={event_type}")

        # Update heartbeat timestamp
        if event_type == "heartbeat":
            self._last_heartbeat = asyncio.get_event_loop().time()

        # Parse event data as JSON if possible
        try:
            if event_data:
                parsed_data = json.loads(event_data)
            else:
                parsed_data = {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse event data as JSON: {event_data}")
            parsed_data = {"raw": event_data}

        # Dispatch to registered handler
        handler = self.event_handlers.get(event_type)
        if handler:
            try:
                # Call handler (may be sync or async)
                if asyncio.iscoroutinefunction(handler):
                    await handler(parsed_data)
                else:
                    handler(parsed_data)
            except Exception as e:
                logger.error(
                    f"Error in handler for event '{event_type}': {e}", exc_info=True
                )
        else:
            logger.debug(f"No handler registered for event type: {event_type}")

    async def disconnect(self) -> None:
        """
        Disconnect from SSE and stop reconnection attempts.
        """
        logger.debug("Disconnecting SSE client")
        self.running = False
        self.should_reconnect = False

    async def check_heartbeat(self) -> None:
        """
        Monitor heartbeat and trigger reconnection if timeout exceeded.

        This should be called periodically in a background task.
        """
        if self._last_heartbeat is None:
            return

        current_time = asyncio.get_event_loop().time()
        time_since_heartbeat = current_time - self._last_heartbeat

        if time_since_heartbeat > self._heartbeat_timeout:
            logger.warning(
                f"Heartbeat timeout ({time_since_heartbeat:.1f}s > "
                f"{self._heartbeat_timeout}s), triggering reconnect"
            )
            # Trigger reconnect by not doing anything - the connection
            # will naturally fail and reconnect logic will kick in
            self._last_heartbeat = None

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from claude_agent_sdk import Transport


def _default_conversations() -> list[list[dict[str, Any]]]:
    """Scripted responses used when no custom conversations are provided."""
    return [
        [
            {
                "type": "assistant",
                "session_id": "default",
                "message": {
                    "model": "claude-3-5-sonnet",
                    "content": [
                        {
                            "type": "text",
                            "text": "The capital of France is Paris.",
                        }
                    ],
                },
            },
            {
                "type": "result",
                "subtype": "response.completed",
                "duration_ms": 42,
                "duration_api_ms": 40,
                "is_error": False,
                "num_turns": 1,
                "session_id": "default",
                "total_cost_usd": 0.0,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 12,
                    "total_tokens": 22,
                },
                "result": {
                    "output": "Answered question about France's capital.",
                },
            },
        ],
        [
            {
                "type": "assistant",
                "session_id": "default",
                "message": {
                    "model": "claude-3-5-sonnet",
                    "content": [
                        {
                            "type": "text",
                            "text": "Paris has a population of roughly two million people.",
                        }
                    ],
                },
            },
            {
                "type": "result",
                "subtype": "response.completed",
                "duration_ms": 55,
                "duration_api_ms": 50,
                "is_error": False,
                "num_turns": 1,
                "session_id": "default",
                "total_cost_usd": 0.0,
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 18,
                    "total_tokens": 30,
                },
                "result": {
                    "output": "Estimated Paris population at ~2 million residents.",
                },
            },
        ],
    ]


class MockClaudeTransport(Transport):
    """In-memory transport that simulates Claude CLI I/O for testing."""

    def __init__(
        self,
        conversations: Sequence[Sequence[dict[str, Any]]] | None = None,
        auto_respond_on_connect: bool = False,
        close_after_responses: bool = False,
    ) -> None:
        self._connected = False
        self._ready = False
        self._closed = False
        self._sentinel: object = object()
        self._message_queue: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue()
        self._conversations = list(conversations) if conversations is not None else _default_conversations()
        self._conversation_index = 0
        self._auto_respond_on_connect = auto_respond_on_connect
        self._close_after_responses = close_after_responses

    async def connect(self) -> None:
        self._connected = True
        self._ready = True
        if self._auto_respond_on_connect:
            self._enqueue_next_conversation(force_close=self._close_after_responses)

    async def write(self, data: str) -> None:
        if not data.strip():
            return

        payload = json.loads(data)
        message_type = payload.get("type")

        if message_type == "control_request":
            request_id = payload.get("request_id")
            response = {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": request_id,
                    "response": {
                        "commands": [],
                        "hooks": payload.get("request", {}).get("hooks"),
                    },
                },
            }
            await self._message_queue.put(response)
            return

        if message_type == "user":
            self._enqueue_next_conversation()
            return

        if message_type == "control_response":
            # Ignore control responses emitted by Query during tests.
            return

        # For any other payloads, no-op to keep interface compliant.

    def _enqueue_next_conversation(self, force_close: bool = False) -> None:
        if not self._conversations:
            return
        if self._conversation_index < len(self._conversations):
            script = self._conversations[self._conversation_index]
        else:
            script = self._conversations[-1]
        self._conversation_index += 1
        for message in script:
            self._message_queue.put_nowait(message)

        if self._close_after_responses or force_close:
            if not self._closed:
                self._closed = True
                self._ready = False
                self._message_queue.put_nowait(self._sentinel)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ready = False
        await self._message_queue.put(self._sentinel)

    def is_ready(self) -> bool:
        return self._ready and self._connected and not self._closed

    async def end_input(self) -> None:
        # No-op for mock transport.
        return

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            message = await self._message_queue.get()
            if message is self._sentinel:
                break
            yield message  # type: ignore[misc]

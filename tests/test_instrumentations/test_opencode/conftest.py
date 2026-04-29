"""Shared fixtures for opencode-ai instrumentation tests.

Each test uses :func:`make_opencode_client` / :func:`make_async_opencode_client`
to build an :class:`opencode_ai.Opencode` / :class:`opencode_ai.AsyncOpencode`
bound to an :class:`httpx.MockTransport`. The transport captures the request
body so tests can assert on what the instrumentation sent over the wire —
mirroring the ``nock``-based approach the TypeScript test suite uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest


@dataclass
class CapturedRequest:
    """Holds the last request observed by the mock transport."""

    url: str | None = None
    method: str | None = None
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)


def _make_handler(
    captured: CapturedRequest,
    *,
    status_code: int,
    response_json: dict[str, Any] | None,
):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.url = str(request.url)
        captured.method = request.method
        captured.headers = dict(request.headers)
        if request.content:
            try:
                captured.body = json.loads(request.content)
            except ValueError:
                captured.body = None
        else:
            captured.body = None
        if response_json is None:
            return httpx.Response(status_code)
        return httpx.Response(status_code, json=response_json)

    return handler


_DEFAULT_CHAT_RESPONSE: dict[str, Any] = {
    "info": {
        "id": "msg-1",
        "sessionID": "test-session",
        "role": "assistant",
    },
    "parts": [
        {"id": "part-1", "type": "text", "text": "Hello!"},
    ],
}


@pytest.fixture
def captured_request() -> CapturedRequest:
    return CapturedRequest()


@pytest.fixture
def make_opencode_client(captured_request: CapturedRequest):
    from opencode_ai import Opencode

    created_clients: list[httpx.Client] = []

    def factory(
        *,
        status_code: int = 200,
        response_json: dict[str, Any] | None = None,
    ):
        transport = httpx.MockTransport(
            _make_handler(
                captured_request,
                status_code=status_code,
                response_json=(
                    response_json
                    if response_json is not None
                    else _DEFAULT_CHAT_RESPONSE
                ),
            )
        )
        http_client = httpx.Client(transport=transport)
        created_clients.append(http_client)
        return Opencode(base_url="http://localhost:4096", http_client=http_client)

    yield factory

    for client in created_clients:
        client.close()


@pytest.fixture
def make_async_opencode_client(captured_request: CapturedRequest):
    from opencode_ai import AsyncOpencode

    created_clients: list[httpx.AsyncClient] = []

    def factory(
        *,
        status_code: int = 200,
        response_json: dict[str, Any] | None = None,
    ):
        transport = httpx.MockTransport(
            _make_handler(
                captured_request,
                status_code=status_code,
                response_json=(
                    response_json
                    if response_json is not None
                    else _DEFAULT_CHAT_RESPONSE
                ),
            )
        )
        http_client = httpx.AsyncClient(transport=transport)
        created_clients.append(http_client)
        return AsyncOpencode(
            base_url="http://localhost:4096", http_client=http_client
        )

    yield factory

    # httpx.AsyncClient.aclose is async; tests that use the factory are async
    # themselves, so close synchronously where possible.
    for client in created_clients:
        try:
            client.close()
        except Exception:  # pragma: no cover - defensive cleanup
            pass

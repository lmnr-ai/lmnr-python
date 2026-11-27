"""The connection to a trial's mock server: `LMNR_MOCK_URL`, the cached key
set (with retry and backoff while the server is unreachable), and the calls.

The env var is read once, on the first observed call (`get_client`), so an
app without it pays one cached check per call and makes no request.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import weakref

import httpx

from .errors import MOCK_FAILURE, MockedError
from .protocol import Outcome, fn_path, parse_keys, parse_outcome

MOCK_URL_ENV = "LMNR_MOCK_URL"
MOCK_TIMEOUT_ENV = "LMNR_MOCK_TIMEOUT_MS"
DEFAULT_TIMEOUT_MS = 30_000
KEYS_TIMEOUT_S = 2.0
RETRY_BACKOFF_S = 0.1
MAX_RETRY_BACKOFF_S = 2.0
JSON_HEADERS = {"content-type": "application/json"}


class MockClient:
    def __init__(self, url: str, timeout_s: float):
        # Never log `url`: it holds the trial's token.
        self.url: str = url.rstrip("/")
        self.timeout_s: float = timeout_s
        self._keys: frozenset[str] | None = None
        self._keys_lock: threading.Lock = threading.Lock()
        self._retry_at: float = 0.0
        self._backoff_s: float = RETRY_BACKOFF_S
        self._last_error: str = ""
        self._sync_client: httpx.Client | None = None
        # An `AsyncClient`'s connections belong to the loop that opened them.
        self._async_clients: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, httpx.AsyncClient
        ] = weakref.WeakKeyDictionary()

    # Keys

    def mocked_keys(self) -> frozenset[str]:
        if self._keys is not None:
            return self._keys
        with self._keys_lock:
            if self._keys is None:
                self._check_backoff()
                try:
                    response = self._client().get(f"{self.url}/keys", timeout=KEYS_TIMEOUT_S)
                except httpx.HTTPError as e:
                    raise self._unreachable(f"{type(e).__name__}: {e}") from None
                self._keys = self._parse_keys(response)
            return self._keys

    async def amocked_keys(self) -> frozenset[str]:
        if self._keys is not None:
            return self._keys
        # No lock: two concurrent first fetches are harmless.
        self._check_backoff()
        try:
            response = await self._aclient().get(f"{self.url}/keys", timeout=KEYS_TIMEOUT_S)
        except httpx.HTTPError as e:
            raise self._unreachable(f"{type(e).__name__}: {e}") from None
        self._keys = self._parse_keys(response)
        return self._keys

    def _parse_keys(self, response: httpx.Response) -> frozenset[str]:
        try:
            return parse_keys(response.status_code, response.content)
        except ValueError as e:
            raise self._unreachable(str(e)) from None

    def _check_backoff(self) -> None:
        if time.monotonic() < self._retry_at:
            raise MockedError(MOCK_FAILURE, f"mock server unreachable ({self._last_error})")

    def _unreachable(self, reason: str) -> MockedError:
        """Fail closed: until `/keys` answers, every observed call raises, and
        the next try waits for a growing backoff."""
        self._last_error = reason
        self._retry_at = time.monotonic() + self._backoff_s
        self._backoff_s = min(self._backoff_s * 2, MAX_RETRY_BACKOFF_S)
        return MockedError(MOCK_FAILURE, f"mock server unreachable ({reason})")

    # Calls

    def call(self, key: str, body: bytes) -> Outcome:
        try:
            response = self._client().post(
                self.url + fn_path(key), content=body, headers=JSON_HEADERS
            )
        except httpx.HTTPError as e:
            raise _call_failed(key, e) from None
        return parse_outcome(key, response.status_code, response.content)

    async def acall(self, key: str, body: bytes) -> Outcome:
        try:
            response = await self._aclient().post(
                self.url + fn_path(key), content=body, headers=JSON_HEADERS
            )
        except httpx.HTTPError as e:
            raise _call_failed(key, e) from None
        return parse_outcome(key, response.status_code, response.content)

    # HTTP clients. `trust_env=False`: the server is on loopback, never behind
    # the app's `HTTP_PROXY`.

    def _client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=self.timeout_s, trust_env=False)
        return self._sync_client

    def _aclient(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        client = self._async_clients.get(loop)
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout_s, trust_env=False)
            self._async_clients[loop] = client
        return client

    def close(self) -> None:
        if self._sync_client is not None:
            self._sync_client.close()
        # Async clients hold no open sockets once their loop is closed.
        self._async_clients.clear()


def _call_failed(key: str, e: httpx.HTTPError) -> MockedError:
    return MockedError(MOCK_FAILURE, f"{key}: mock server unreachable ({type(e).__name__}: {e})")


_resolved = False
_client: MockClient | None = None


def get_client() -> MockClient | None:
    """The mock client, or `None` when `LMNR_MOCK_URL` is unset or empty (the
    production case). Reads the environment once."""
    global _resolved, _client
    if _resolved:
        return _client
    url = os.environ.get(MOCK_URL_ENV, "").strip()
    _client = MockClient(url, _timeout_s()) if url else None
    _resolved = True
    return _client


def reset() -> None:
    """Forget the client so the next observed call re-reads the environment
    (tests)."""
    global _resolved, _client
    if _client is not None:
        _client.close()
    _resolved, _client = False, None


def _timeout_s() -> float:
    raw = os.environ.get(MOCK_TIMEOUT_ENV, "").strip()
    try:
        ms = float(raw) if raw else DEFAULT_TIMEOUT_MS
    except ValueError:
        ms = DEFAULT_TIMEOUT_MS
    return (ms if ms > 0 else DEFAULT_TIMEOUT_MS) / 1000

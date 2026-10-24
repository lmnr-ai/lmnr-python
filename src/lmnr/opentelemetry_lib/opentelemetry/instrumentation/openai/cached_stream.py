"""Stand-ins for the SDK's `Stream` / `AsyncStream`, served on a debug-run cache HIT.

A leaf module (no lmnr imports) so `shared/__init__.py::is_streaming_response` can
recognize them without importing `rollout.py`.

They are plain classes rather than generators on purpose: `ChatStream` is a wrapt
proxy whose `__init__` sets `_span` & co. on the WRAPPED object, which a generator
rejects (`'generator' object has no attribute '_span'`).
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Generic, Self

from typing_extensions import TypeVar

E = TypeVar("E")


class _CachedHTTPResponse:
    """Stand-in for the `httpx.Response` behind an SDK `Stream`. The SDK's
    `ResponseStream` helpers read `.response` off the stream they wrap and close
    it; nothing was fetched, so there is nothing to close."""

    status_code: int = 200
    headers: dict[str, str] = {}  # noqa: RUF012

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class CachedEventStream(Generic[E]):
    """Sync replay of cached events (chat chunks, Responses-API events), served in
    place of the SDK's `Stream[E]`. Supports what callers (and
    `client.responses.stream()`) use: iteration, the context manager, `close()`
    and `.response`."""

    def __init__(self, events: Sequence[E]):
        self._events: Sequence[E] = events
        self._index: int = 0
        self.response: _CachedHTTPResponse = _CachedHTTPResponse()

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> E:
        if self._index >= len(self._events):
            raise StopIteration
        event = self._events[self._index]
        self._index += 1
        return event

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._index = len(self._events)


class AsyncCachedEventStream(Generic[E]):
    """Async counterpart of `CachedEventStream`, replacing the SDK's
    `AsyncStream[E]` (whose `close()` is a coroutine)."""

    def __init__(self, events: Sequence[E]):
        self._events: Sequence[E] = events
        self._index: int = 0
        self.response: _CachedHTTPResponse = _CachedHTTPResponse()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> E:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        self._index = len(self._events)

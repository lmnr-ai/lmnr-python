"""`with_raw_response` support for the Stainless-generated SDKs (openai, anthropic).

`client.x.with_raw_response.create(...)` calls the same wrapped `create` with an
`X-Stainless-Raw-Response` extra header and gets a `LegacyAPIResponse` (anthropic
>= 1: `APIResponse`) back; the caller then reads the body with `.parse()`.
LangChain's OpenAI and Anthropic chat models call the SDKs this way, so a wrapper
that returns anything other than an object with `.parse()` breaks them. Two cases need help:

- a live stream: the wrapper must instrument the parsed `Stream`, not the
  `LegacyAPIResponse`, and hand the instrumented stream back through `.parse()`;
- a debug-replay cache hit: there is no HTTP response at all, only the cached
  body, which still has to come back behind `.parse()`.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, cast

import httpx

RAW_RESPONSE_HEADER = "X-Stainless-Raw-Response"


def is_raw_response_request(kwargs: Mapping[str, Any]) -> bool:
    """True for `with_raw_response` calls, which send the header as `"true"`
    (openai, anthropic < 1) or `"raw"` (anthropic >= 1). `with_streaming_response`
    sends `"stream"` and is excluded: its body belongs to the caller."""
    extra_headers = kwargs.get("extra_headers")
    return isinstance(extra_headers, Mapping) and extra_headers.get(RAW_RESPONSE_HEADER) in (
        "true",
        "raw",
    )


def is_api_response(response: object) -> bool:
    """True for the SDK's own `LegacyAPIResponse` / `APIResponse` (not a parsed
    body or a stream)."""
    return callable(getattr(response, "parse", None)) and hasattr(response, "http_response")


class RawResponse:
    """A `LegacyAPIResponse` / `APIResponse` stand-in whose `.parse()` returns `parsed`.

    With `awaitable_parse`, `.parse()` returns an awaitable instead, like anthropic
    >= 1's `AsyncAPIResponse`. Everything else (`headers`, `http_response`, ...) is
    read from the real response when there is one; a replayed response has empty
    headers.
    """

    def __init__(
        self, parsed: object, raw: Any | None = None, *, awaitable_parse: bool = False
    ) -> None:
        self._parsed: object = parsed
        self._raw: Any | None = raw
        self._awaitable_parse: bool = awaitable_parse

    def parse(self, *, to: type[Any] | None = None) -> Any:
        if to is not None and self._raw is not None:
            return self._raw.parse(to=to)
        if self._awaitable_parse:
            return _resolved(self._parsed)
        return self._parsed

    @property
    def headers(self) -> httpx.Headers:
        return self._raw.headers if self._raw is not None else httpx.Headers()

    def __getattr__(self, name: str) -> Any:
        if self._raw is None:
            raise AttributeError(name)
        return getattr(self._raw, name)


def parse_raw_stream(kwargs: Mapping[str, Any], response: Any) -> tuple[Any, Any | None]:
    """For a live `with_raw_response` stream, return `(parsed stream, raw response)`
    so the wrapper instruments the stream itself. Otherwise `(response, None)`."""
    if kwargs.get("stream") and is_raw_response_request(kwargs) and is_api_response(response):
        return response.parse(), response
    return response, None


async def aparse_raw_response(kwargs: Mapping[str, Any], response: Any) -> tuple[Any, Any | None]:
    """`parse_raw_stream` for async wrappers. anthropic >= 1's `AsyncAPIResponse.parse()`
    is a coroutine (the legacy response's is not), which the wrappers' sync-parsing
    attribute helpers can't read, so such a response is parsed here even when it is
    not a stream."""
    if not (is_raw_response_request(kwargs) and is_api_response(response)):
        return response, None
    async_parse = inspect.iscoroutinefunction(response.parse)
    if not (kwargs.get("stream") or async_parse):
        return response, None
    parsed = await response.parse() if async_parse else response.parse()
    return parsed, response


async def _resolved(value: object) -> object:
    return value


def as_raw_response(
    kwargs: Mapping[str, Any], result: Any, raw: Any | None, *, is_async: bool = False
) -> Any:
    """Give a `with_raw_response` caller back the shape it expects.

    `raw` is the response `parse_raw_stream` / `aparse_raw_response` split off.
    Without one, a result that is not an API response for a raw-response request
    was served from the replay cache, so it gets wrapped too. Async wrappers pass `is_async`: `.parse()`
    is then awaitable when the SDK's would be (the live response's `parse` is a
    coroutine function; for a replay, the `raw` header marks anthropic >= 1).
    """
    if raw is not None:
        return RawResponse(
            result, raw, awaitable_parse=is_async and inspect.iscoroutinefunction(raw.parse)
        )
    if is_raw_response_request(kwargs) and not is_api_response(result):
        extra_headers = cast(Mapping[str, Any], kwargs["extra_headers"])
        return RawResponse(
            result, awaitable_parse=is_async and extra_headers[RAW_RESPONSE_HEADER] == "raw"
        )
    return result

"""The SDK side of lmnr-mock v1 (`spec/lmnr-mock/lmnr-mock.md`): request
bodies and what each response means. No I/O here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import quote

import orjson

from lmnr.sdk.utils import json_dumps

from .errors import MOCK_FAILURE, MockedError

MAX_BODY_CHARS = 500


@dataclass(frozen=True)
class Outcome:
    action: Literal["respond", "raise", "passthrough"]
    value: Any = None
    error: MockedError | None = None


def fn_path(key: str) -> str:
    """`/fn/{key}`, the key percent-encoded as one path segment."""
    return "/fn/" + quote(key, safe="")


def request_body(
    args: list[Any], kwargs: dict[str, Any], trace_id: str | None, span_id: str | None
) -> bytes:
    """Arguments go through `json_dumps`, like span input: an unserializable
    value becomes its string form instead of failing the call."""
    return json_dumps(
        {
            "args": args,
            "kwargs": kwargs,
            "lang": "python",
            "trace_id": trace_id,
            "span_id": span_id,
        }
    ).encode()


def parse_keys(status: int, body: bytes) -> frozenset[str]:
    """`GET /keys`; raises `ValueError` with what was wrong."""
    keys: object = _object(body).get("keys") if status == 200 else None
    valid = isinstance(keys, list) and all(isinstance(k, str) for k in cast(list[object], keys))
    if not valid:
        raise ValueError(f"GET /keys answered {_describe(status, body)}")
    return frozenset(cast(list[str], keys))


def parse_outcome(key: str, status: int, body: bytes) -> Outcome:
    """`POST /fn/{key}`. Anything but the three 200 actions raises
    `MockedError(MOCK_FAILURE)`: a mocked call never falls back to the real
    function."""
    payload = _object(body)
    action, error = payload.get("action"), _error(payload)
    if status == 200:
        if action == "respond" and "value" in payload:
            return Outcome("respond", value=payload["value"])
        if action == "passthrough":
            return Outcome("passthrough")
        if action == "raise" and error is not None:
            return Outcome("raise", error=MockedError(*error))
    if status == 500 and action == "handler_error" and error is not None:
        reason = f"the mock handler raised {error[0]}: {error[1]}"
    elif status != 200 and error is not None:
        reason = f"the mock server answered {status} {error[0]}: {error[1]}"
    else:
        reason = f"unexpected mock server response {_describe(status, body)}"
    raise MockedError(MOCK_FAILURE, f"{key}: {reason}")


def _object(body: bytes) -> dict[str, Any]:
    """The body as a JSON object; `{}` when it is anything else."""
    try:
        payload = orjson.loads(body)
    except orjson.JSONDecodeError:
        return {}
    return cast(dict[str, Any], payload) if isinstance(payload, dict) else {}


def _error(payload: dict[str, Any]) -> tuple[str, str] | None:
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    error = cast(dict[str, Any], error)
    type, message = error.get("type"), error.get("message")
    if not isinstance(type, str) or not isinstance(message, str):
        return None
    return type, message


def _describe(status: int, body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").strip() or "<empty body>"
    if len(text) > MAX_BODY_CHARS:
        text = text[:MAX_BODY_CHARS] + "... (truncated)"
    return f"[{status}] {text}"

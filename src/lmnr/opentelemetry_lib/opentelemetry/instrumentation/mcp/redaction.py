"""Key-based redaction of MCP payloads (arguments, results, params).

Shared contract with the TS SDK and app-server: a key is secret when its
normalized form (lowercase, non-alphanumerics removed: `API-Key` -> `apikey`)
contains one of `SECRET_SUBSTRINGS`, ends with `token` (but not `tokens`), or
is one of `SECRET_EXACT`. Customers add exact names (normalized the same way).
The value, whatever its type, becomes `REDACTED`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, cast

REDACTED = "[REDACTED]"

SECRET_SUBSTRINGS = (
    "password",
    "passwd",
    "secret",
    "apikey",
    "privatekey",
    "accesskey",
    "authorization",
    "credential",
    "cookie",
    "signature",
)
SECRET_EXACT = frozenset({"auth", "pwd", "otp", "bearer", "jwt"})

_NON_ALNUM = re.compile(r"[^a-z0-9]")
_NO_KEYS: frozenset[str] = frozenset()


def normalize_key(key: str) -> str:
    return _NON_ALNUM.sub("", key.lower())


def normalize_keys(keys: Iterable[str] | None) -> frozenset[str]:
    return frozenset(normalize_key(k) for k in keys or () if isinstance(k, str))  # pyright: ignore[reportUnnecessaryIsInstance]


def is_secret_key(key: str, extra: frozenset[str] = _NO_KEYS) -> bool:
    normalized = normalize_key(key)
    return (
        any(part in normalized for part in SECRET_SUBSTRINGS)
        or normalized.endswith("token")
        or normalized in SECRET_EXACT
        or normalized in extra
    )


def redact(value: Any, extra: frozenset[str] = _NO_KEYS) -> Any:
    """A copy of `value` with every secret key's value replaced, at any depth.

    Pydantic models are dumped in their wire shape (aliases) first, so this is
    also what the span records. The input is never mutated.
    """
    if hasattr(value, "model_dump") and not isinstance(value, type):
        value = value.model_dump(by_alias=True, mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {
            key: (REDACTED if is_secret_key(str(key), extra) else redact(item, extra))
            for key, item in cast(Mapping[Any, Any], value).items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item, extra) for item in cast(Iterable[Any], value)]
    return value

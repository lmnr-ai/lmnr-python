"""`mock`: what `EvalRunner.mocks` handlers use.

    from lmnr.evals import EvalRunner, mock

    def get_customer(call, ctx):              # function mock: an observed function
        return ctx.case.fixture_json("crm_customers.json").get(call.args[0])

    def zendesk(req, ctx):                    # HTTP mock
        if req.method == "POST" and req.path.endswith("/comments"):
            return mock.Response(201, json={"ok": True})
        return None                           # unmatched -> 501

    class SupportEval(EvalRunner):
        mocks = {
            "crm.get_customer": get_customer,
            "https://api.zendesk.com": mock.http(zendesk, base_url_env="ZENDESK_URL"),
        }

Handlers run inside the trial's container, in the mock server, which imports
them by reference: they must be top-level functions of `eval.py` or of a
module under the evals root (`evals/_shared/mocks.py`), sync or async.

The mock server imports this module in the container, next to the handlers.
"""

from __future__ import annotations

import json as jsonlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from typing_extensions import override

Handler = Callable[..., Any]


class _Passthrough:
    @override
    def __repr__(self) -> str:
        return "mock.PASSTHROUGH"


PASSTHROUGH: Any = _Passthrough()
"""Return it from a function mock to run the real function. From an HTTP mock
it means unmatched (501): there is no egress proxy yet."""

_UNSET: Any = object()


class ProjectDependencies:
    """The type of `mock.PROJECT_DEPENDENCIES`."""

    @override
    def __repr__(self) -> str:
        return "mock.PROJECT_DEPENDENCIES"


PROJECT_DEPENDENCIES: Final = ProjectDependencies()
"""In `EvalRunner.mock_dependencies`, the evals project's dependencies (what
the default installs), so a list can extend them:
`mock_dependencies = [mock.PROJECT_DEPENDENCIES, "extra>=1"]`."""


class Error(Exception):
    """Raise it to make the mocked function raise (`MockedError` in Python, an
    `Error` named `type` in TypeScript), or an HTTP mock answer 500 with
    `{"error": {"type", "message"}}`."""

    def __init__(self, type: str, message: str = ""):
        super().__init__(f"{type}: {message}" if message else type)
        self.type: str = str(type)
        self.message: str = str(message)


class Unmatched(Exception):
    """Raise it from an HTTP mock for "no mock for this request": 501, logged
    as `unmatched` (as returning None does)."""


@dataclass
class Response:
    """An HTTP mock's response. Pass at most one of `json`, `text`, `body`."""

    status: int = 200
    json: Any = _UNSET
    text: str | None = None
    body: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def encode(self) -> tuple[int, dict[str, str], bytes]:
        """Status, headers (with a default Content-Type) and body bytes."""
        given = [
            name
            for name, is_set in (
                ("json", self.json is not _UNSET),
                ("text", self.text is not None),
                ("body", self.body is not None),
            )
            if is_set
        ]
        if len(given) > 1:
            raise ValueError(f"mock.Response takes one of json=, text=, body=; got {given}.")
        headers = {str(k): str(v) for k, v in self.headers.items()}
        if self.json is not _UNSET:
            content_type, data = "application/json", jsonlib.dumps(self.json, allow_nan=False)
            payload = data.encode()
        elif self.text is not None:
            content_type, payload = "text/plain; charset=utf-8", self.text.encode()
        elif self.body is not None:
            content_type, payload = "application/octet-stream", bytes(self.body)
        else:
            content_type, payload = None, b""
        if content_type and not any(k.lower() == "content-type" for k in headers):
            headers["Content-Type"] = content_type
        return int(self.status), headers, payload


@dataclass(frozen=True)
class HttpMock:
    """`mock.http(handler, base_url_env=...)`."""

    handler: Handler
    base_url_env: str


def http(handler: Handler, *, base_url_env: str) -> HttpMock:
    """An HTTP mock. The runner sets `base_url_env` in every command's env to
    the mock's base URL (`http://127.0.0.1:<port>/h/<host><path of the key>`),
    so point the app's API client at that variable.

    `handler(req, ctx)` gets a `mock.Request`; it returns a `mock.Response`, a
    JSON value (200), or None for unmatched (501).
    """
    return HttpMock(handler=handler, base_url_env=base_url_env)


@dataclass(frozen=True)
class Call:
    """A call of a mocked function."""

    key: str
    # Positional, in every language: Python sends every positional parameter
    # in order (defaults filled in), then the `*args` values.
    args: list[Any]
    # Python's keyword-only parameters and `**kwargs` entries; `{}` from
    # TypeScript.
    kwargs: dict[str, Any]
    # 0-based, per key.
    index: int


@dataclass(frozen=True)
class Request:
    """A request to an HTTP mock."""

    method: str
    # After the origin, with a leading "/".
    path: str
    query: dict[str, list[str]]
    # Lower-cased names.
    headers: dict[str, str]
    body: bytes
    index: int = 0

    def text(self) -> str:
        return self.body.decode()

    def json(self) -> Any:
        return jsonlib.loads(self.body) if self.body else None


def missing_fixture(case_name: str, path: str) -> FileNotFoundError:
    return FileNotFoundError(f"The case {case_name!r} has no tests/fixtures/{path}.")


class FixtureReader(ABC):
    """`fixture_bytes(path)` / `fixture_text` / `fixture_json`: a file under
    the case's `tests/fixtures/`, the same on `ctx.case` on the host (`Case`)
    and in a mock handler (`CaseInfo`). A missing one raises
    `FileNotFoundError` with its path."""

    @abstractmethod
    def fixture_bytes(self, path: str) -> bytes: ...

    def fixture_text(self, path: str) -> str:
        return self.fixture_bytes(path).decode()

    def fixture_json(self, path: str) -> Any:
        return jsonlib.loads(self.fixture_bytes(path))


@dataclass(frozen=True)
class CaseInfo(FixtureReader):
    """A handler's `ctx.case`: `name`, `metadata` (`[metadata]` in
    `task.toml`) and the fixture readers, over the case's `tests/fixtures/`
    loaded when the trial started."""

    name: str
    metadata: dict[str, Any]
    fixtures: Mapping[str, bytes] = field(default_factory=dict[str, bytes], repr=False)

    @override
    def fixture_bytes(self, path: str) -> bytes:
        key = path.strip("/")
        if key not in self.fixtures:
            raise missing_fixture(self.name, key)
        return self.fixtures[key]


class Context:
    """A handler's `ctx`.

    - `state`: one dict per trial, shared by every handler.
    - `calls`: earlier calls of this key, as logged (`request`, `outcome`,
      `response`, ...), oldest first.
    - `case`: `name`, `metadata` (`[metadata]` in `task.toml`), and
      `fixture_bytes(path)` / `fixture_text` / `fixture_json` over the case's
      `tests/fixtures/`, loaded when the trial started (as on the host's
      `ctx.case`).
    - `env`.
    """

    def __init__(
        self,
        *,
        state: dict[str, Any],
        calls: list[dict[str, Any]],
        case: CaseInfo,
        env: Mapping[str, str],
    ):
        self.state: dict[str, Any] = state
        self.calls: list[dict[str, Any]] = calls
        self.case: CaseInfo = case
        self.env: dict[str, str] = dict(env)

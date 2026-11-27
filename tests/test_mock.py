"""Observe-level mocks (`lmnr.sdk.mock`, LAM-2588 phase 4): `@observe`d
functions answered by a trial's mock server when `LMNR_MOCK_URL` is set."""

from __future__ import annotations

import asyncio
import json
import pickle
import socket
import sys
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator, Iterable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TypedDict
from unittest.mock import patch
from urllib.parse import unquote

import httpx
import pydantic
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typing_extensions import override

import lmnr
from lmnr import MockedError, observe
from lmnr.evals.mocks.server import MockServer, http_server
from lmnr.sdk.mock import MOCK_FAILURE, client, intercept

SPEC_DIR = Path(__file__).resolve().parent.parent / "spec" / "lmnr-mock"
PREFIX = "/_lmnr/v1/" + "t" * 32


class FakeServer:
    """Answers `/keys` with `keys` and each `/fn/<key>` with the next scripted
    `(status, body)` for that key; records every request."""

    def __init__(self, keys: list[str] | None = None):
        self.keys: list[str] = keys or []
        self.keys_status: list[int] = []
        self.replies: dict[str, list[tuple[int, Any]]] = {}
        self.requests: list[tuple[str, str, Any]] = []
        self.post_delay: threading.Event | None = None
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                fake.requests.append(("GET", self.path, None))
                status = fake.keys_status.pop(0) if fake.keys_status else 200
                self._send(status, {"keys": fake.keys} if status == 200 else "busy")

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["content-length"]))
                key = unquote(self.path.removeprefix(PREFIX + "/fn/"))
                fake.requests.append(("POST", key, json.loads(body)))
                if fake.post_delay is not None:
                    _ = fake.post_delay.wait(5)
                status, reply = fake.replies[key].pop(0)
                self._send(status, reply)

            def _send(self, status: int, reply: Any) -> None:
                payload = reply if isinstance(reply, bytes) else json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                _ = self.wfile.write(payload)

            @override
            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.httpd: ThreadingHTTPServer = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url: str = f"http://127.0.0.1:{self.httpd.server_address[1]}{PREFIX}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def reply(self, key: str, status: int, body: Any) -> None:
        self.replies.setdefault(key, []).append((status, body))

    def respond(self, key: str, value: Any) -> None:
        self.reply(key, 200, {"action": "respond", "value": value})

    def calls(self) -> list[tuple[str, Any]]:
        return [(key, body) for method, key, body in self.requests if method == "POST"]


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch):
    """Point `LMNR_MOCK_URL` at a server; `client.reset()` around each test."""

    def use(server_or_url: FakeServer | str) -> None:
        url = server_or_url if isinstance(server_or_url, str) else server_or_url.url
        monkeypatch.setenv("LMNR_MOCK_URL", url)
        client.reset()

    client.reset()
    yield use
    client.reset()


@pytest.fixture
def server(mock_env: Any) -> Generator[FakeServer, None, None]:
    fake = FakeServer(["crm.get_customer", "payments.refund", "search.web"])
    mock_env(fake)
    yield fake
    fake.httpd.shutdown()


def closed_port_url() -> tuple[socket.socket, str]:
    # Bound but not listening: connects are refused (tests/AGENTS.md).
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    return sock, f"http://127.0.0.1:{sock.getsockname()[1]}{PREFIX}"


# Off by default


def test_unset_env_runs_the_function_and_makes_no_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LMNR_MOCK_URL", raising=False)
    client.reset()

    @observe(name="crm.get_customer")
    def get_customer(email: str) -> str:
        return "real"

    with patch("httpx.Client.send") as send:
        assert get_customer("a") == "real"
    send.assert_not_called()
    assert client.get_client() is None


# Outcomes


def test_respond_returns_the_value_and_records_the_span(
    server: FakeServer, span_exporter: InMemorySpanExporter
):
    server.respond("crm.get_customer", {"id": "cus_1"})
    ran: list[str] = []

    @observe(name="crm.get_customer")
    def get_customer(email: str, plan: str = "pro") -> dict[str, Any]:
        ran.append(email)
        return {}

    assert get_customer("ana@example.com") == {"id": "cus_1"}
    assert ran == []
    [(_, body)] = server.calls()
    assert (body["args"], body["kwargs"]) == (["ana@example.com", "pro"], {})
    assert body["lang"] == "python"

    [span] = span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes["lmnr.span.mocked"] is True
    assert attributes["lmnr.mock.key"] == "crm.get_customer"
    assert attributes["lmnr.mock.outcome"] == "respond"
    assert json.loads(str(attributes["lmnr.span.input"])) == {
        "email": "ana@example.com",
        "plan": "pro",
    }
    assert json.loads(str(attributes["lmnr.span.output"])) == {"id": "cus_1"}
    context = span.get_span_context()
    assert context is not None
    assert body["trace_id"] == f"{context.trace_id:032x}"
    assert body["span_id"] == f"{context.span_id:016x}"


def test_raise_raises_mocked_error_and_records_it(
    server: FakeServer, span_exporter: InMemorySpanExporter
):
    server.reply(
        "payments.refund",
        200,
        {"action": "raise", "error": {"type": "DuplicateRefund", "message": "already refunded"}},
    )

    @observe(name="payments.refund")
    def refund(charge_id: str) -> None:
        raise AssertionError("the real function ran")

    with pytest.raises(MockedError) as raised:
        refund("ch_1")
    assert (raised.value.type, raised.value.message) == ("DuplicateRefund", "already refunded")
    assert str(raised.value) == "DuplicateRefund: already refunded"

    [span] = span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes["lmnr.span.mocked"] is True
    assert attributes["lmnr.mock.outcome"] == "raise"
    assert not span.status.is_ok
    [event] = span.events
    assert event.name == "exception"
    assert (event.attributes or {})["exception.message"] == "DuplicateRefund: already refunded"


def test_passthrough_runs_the_real_function(
    server: FakeServer, span_exporter: InMemorySpanExporter
):
    server.reply("search.web", 200, {"action": "passthrough"})

    @observe(name="search.web")
    def search(q: str) -> str:
        return f"real {q}"

    assert search("weather") == "real weather"
    [span] = span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert "lmnr.span.mocked" not in attributes
    assert attributes["lmnr.mock.outcome"] == "passthrough"
    assert json.loads(str(attributes["lmnr.span.output"])) == "real weather"


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (
            500,
            {"action": "handler_error", "error": {"type": "ValueError", "message": "broken"}},
            "crm.get_customer: the mock handler raised ValueError: broken",
        ),
        (
            404,
            {"error": {"type": "UnknownMockKey", "message": "no function mock"}},
            "crm.get_customer: the mock server answered 404 UnknownMockKey: no function mock",
        ),
        (
            400,
            {"error": {"type": "BadRequest", "message": "expected args"}},
            "crm.get_customer: the mock server answered 400 BadRequest: expected args",
        ),
        (
            200,
            {"action": "respond"},
            'crm.get_customer: unexpected mock server response [200] {"action": "respond"}',
        ),
        (200, b"not json", "crm.get_customer: unexpected mock server response [200] not json"),
        (502, b"", "crm.get_customer: unexpected mock server response [502] <empty body>"),
    ],
)
def test_failures_raise_and_never_call_the_function(
    server: FakeServer,
    span_exporter: InMemorySpanExporter,
    status: int,
    body: Any,
    message: str,
):
    server.reply("crm.get_customer", status, body)

    @observe(name="crm.get_customer")
    def get_customer(email: str) -> str:
        raise AssertionError("the real function ran")

    with pytest.raises(MockedError) as raised:
        _ = get_customer("a")
    assert (raised.value.type, raised.value.message) == (MOCK_FAILURE, message)
    [span] = span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes["lmnr.mock.outcome"] == "error"
    assert "lmnr.span.mocked" not in attributes


def test_a_slow_server_times_out(mock_env: Any, monkeypatch: pytest.MonkeyPatch):
    fake = FakeServer(["slow"])
    fake.post_delay = threading.Event()
    fake.respond("slow", 1)
    monkeypatch.setenv("LMNR_MOCK_TIMEOUT_MS", "200")
    mock_env(fake)

    @observe(name="slow")
    def slow() -> int:
        return 0

    started = time.monotonic()
    with pytest.raises(MockedError, match="slow: mock server unreachable \\(ReadTimeout"):
        _ = slow()
    assert time.monotonic() - started < 2
    fake.post_delay.set()
    fake.httpd.shutdown()


# Keys


def test_key_is_mock_key_then_name_then_function_name(server: FakeServer):
    server.keys = ["by_mock_key", "by_name", "by_function_name"]
    for key in server.keys:
        server.respond(key, key)

    @observe(name="by_name", mock_key="by_mock_key")
    def a() -> str:
        return "real"

    @observe(name="by_name")
    def b() -> str:
        return "real"

    @observe()
    def by_function_name() -> str:
        return "real"

    assert (a(), b(), by_function_name()) == ("by_mock_key", "by_name", "by_function_name")


def test_unmocked_keys_run_and_keys_are_fetched_once(server: FakeServer):
    @observe(name="not.mocked")
    def f(x: int) -> int:
        return x + 1

    assert [f(1), f(2)] == [2, 3]
    assert server.requests == [("GET", PREFIX + "/keys", None)]


def test_mockable_false_never_consults_the_server(server: FakeServer):
    @observe(name="crm.get_customer", mockable=False)
    def get_customer() -> str:
        return "real"

    assert get_customer() == "real"
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectError("refused"), httpx.Response(503, text="busy")],
    ids=["transport", "status"],
)
async def test_an_async_key_fetch_failing_after_another_succeeded_returns_the_keys(
    monkeypatch: pytest.MonkeyPatch, failure: Exception | httpx.Response
):
    mock = client.MockClient("http://127.0.0.1:1" + PREFIX, 1.0)
    release = asyncio.Event()

    class FirstFailsLate:
        calls: int = 0

        async def get(self, url: str, timeout: float) -> httpx.Response:
            self.calls += 1
            if self.calls > 1:
                return httpx.Response(200, json={"keys": ["k"]})
            _ = await release.wait()
            if isinstance(failure, Exception):
                raise failure
            return failure

    fake = FirstFailsLate()
    monkeypatch.setattr(mock, "_aclient", lambda: fake)
    first = asyncio.create_task(mock.amocked_keys())
    await asyncio.sleep(0)  # the first fetch is in flight
    assert await mock.amocked_keys() == frozenset({"k"})
    release.set()
    assert await first == frozenset({"k"})


def test_unreachable_server_fails_closed_then_retries_after_a_backoff(mock_env: Any):
    sock, url = closed_port_url()
    mock_env(url)

    @observe(name="anything")
    def f() -> str:
        return "real"

    with pytest.raises(MockedError, match="mock server unreachable \\(ConnectError"):
        _ = f()
    sock.close()

    fake = FakeServer(["other"])
    fake.keys_status = [503]
    mock_env(fake)
    with pytest.raises(MockedError, match=r"mock server unreachable \(GET /keys answered \[503\]"):
        _ = f()
    # Inside the backoff: no request.
    with pytest.raises(MockedError, match="mock server unreachable"):
        _ = f()
    assert len(fake.requests) == 1
    time.sleep(client.RETRY_BACKOFF_S + 0.05)
    assert f() == "real"
    assert len(fake.requests) == 2
    fake.httpd.shutdown()


def test_the_token_never_appears_in_errors(mock_env: Any):
    sock, url = closed_port_url()
    mock_env(url)

    @observe(name="anything")
    def f() -> None:
        pass

    with pytest.raises(MockedError) as raised:
        f()
    assert "t" * 32 not in str(raised.value)
    sock.close()


# Arguments


class Service:
    @observe(name="crm.get_customer", ignore_inputs=["secret"])
    def get(self, email: str, *rest: int, secret: str = "s", **extra: Any) -> Any:
        return None

    @classmethod
    @observe(name="search.web")
    def search(cls, q: str, limit: int = 3) -> Any:
        return None


def test_methods_send_args_without_self_or_cls(
    server: FakeServer, span_exporter: InMemorySpanExporter
):
    server.respond("crm.get_customer", 1)
    server.respond("search.web", 2)

    assert Service().get("a", 1, 2, flag=True) == 1
    assert Service.search("q") == 2
    assert [(body["args"], body["kwargs"]) for _, body in server.calls()] == [
        # `ignore_inputs` is about the span, the handler sees every argument
        (["a", 1, 2], {"secret": "s", "flag": True}),
        (["q", 3], {}),
    ]
    # The span's input is still by name.
    inputs = [
        json.loads(str((s.attributes or {})["lmnr.span.input"]))
        for s in span_exporter.get_finished_spans()
    ]
    assert inputs == [{"email": "a", "rest": [1, 2], "flag": True}, {"q": "q", "limit": 3}]


def positional_only(a: int, b: int = 2, /, c: int = 3) -> None: ...


def var_positional(a: int, *rest: int) -> None: ...


def keyword_only(a: int, *, b: int, c: int = 3) -> None: ...


def var_keyword(a: int, b: int = 2, **extra: Any) -> None: ...


def everything(a: int, /, b: int, *rest: int, c: int, d: int = 4, **extra: Any) -> None: ...


@pytest.mark.parametrize(
    ("fn", "args", "kwargs", "expected"),
    [
        (positional_only, (1,), {}, ([1, 2, 3], {})),
        (positional_only, (1,), {"c": 5}, ([1, 2, 5], {})),
        (var_positional, (1, 2, 3), {}, ([1, 2, 3], {})),
        (var_positional, (1,), {}, ([1], {})),
        (keyword_only, (1,), {"b": 2}, ([1], {"b": 2, "c": 3})),
        (keyword_only, (), {"a": 1, "b": 2}, ([1], {"b": 2, "c": 3})),
        (var_keyword, (), {"a": 1, "z": 9}, ([1, 2], {"z": 9})),
        (everything, (1, 2, 3), {"c": 5, "x": 0}, ([1, 2, 3], {"c": 5, "d": 4, "x": 0})),
        # Doesn't bind (raises TypeError when it runs): sent as passed.
        (keyword_only, (1, 2), {}, ([1, 2], {})),
        (positional_only, (), {"a": 1}, ([], {"a": 1})),
    ],
)
def test_args_are_bound_to_the_signature(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    expected: tuple[list[Any], dict[str, Any]],
):
    assert intercept._mock_call_args(fn, False, args, kwargs) == expected


def test_a_method_call_that_doesnt_bind_drops_self():
    def get(self: Any, email: str) -> None: ...

    assert intercept._mock_call_args(get, True, ("self", "a", "b"), {"x": 1}) == (
        ["a", "b"],
        {"x": 1},
    )


def test_a_call_that_doesnt_bind_is_sent_as_passed(server: FakeServer):
    server.respond("crm.get_customer", "mocked")

    @observe(name="crm.get_customer")
    def get_customer(email: str) -> str:
        return "real"

    assert get_customer("a", "b", extra=1) == "mocked"  # pyright: ignore[reportCallIssue]
    [(_, body)] = server.calls()
    assert (body["args"], body["kwargs"]) == (["a", "b"], {"extra": 1})


def test_unserializable_args_become_strings(server: FakeServer):
    server.respond("crm.get_customer", None)

    class Opaque:
        @override
        def __str__(self) -> str:
            return "<opaque>"

    @observe(name="crm.get_customer")
    def get_customer(value: Any, data: bytes) -> None:
        pass

    get_customer(Opaque(), b"\xff")
    [(_, body)] = server.calls()
    assert body["args"] == ["<opaque>", "/w=="]


def test_mocks_work_without_tracing(server: FakeServer):
    server.respond("crm.get_customer", "mocked")

    @observe(name="crm.get_customer")
    def get_customer() -> str:
        return "real"

    with patch("lmnr.opentelemetry_lib.decorators.is_tracing_initialized", return_value=False):
        assert get_customer() == "mocked"
    [(_, body)] = server.calls()
    assert (body["trace_id"], body["span_id"]) == (None, None)


# Return validation


class Customer(pydantic.BaseModel):
    id: str
    plan: str = "free"


@dataclass
class Order:
    id: str
    total: float


class Quote(TypedDict):
    sku: str
    price: int


class Opaque:
    """No pydantic schema."""


def test_values_are_validated_against_the_return_annotation(server: FakeServer):
    server.keys = ["customer", "order", "quotes", "untyped", "optional"]
    server.respond("customer", {"id": "cus_1"})
    server.respond("order", {"id": "o_1", "total": "9.5"})
    server.respond("quotes", [{"sku": "A", "price": 1}])
    server.respond("untyped", {"anything": True})
    server.respond("optional", None)

    @observe(name="customer")
    def customer() -> Customer:
        raise AssertionError

    @observe(name="order")
    def order() -> Order:
        raise AssertionError

    @observe(name="quotes")
    def quotes() -> list[Quote]:
        raise AssertionError

    @observe(name="untyped")
    def untyped():
        raise AssertionError

    @observe(name="optional")
    def optional() -> Customer | None:
        raise AssertionError

    assert customer() == Customer(id="cus_1", plan="free")
    assert order() == Order(id="o_1", total=9.5)
    assert quotes() == [{"sku": "A", "price": 1}]
    assert untyped() == {"anything": True}
    assert optional() is None


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        ("Customer", {"plan": "pro"}),  # missing a required field
        ("None", {"id": "cus_1"}),  # wrong: the function returns something
        ("str", 5),
    ],
)
def test_a_value_that_doesnt_fit_the_annotation_is_returned_as_is_with_a_warning(
    server: FakeServer,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    annotation: str,
    value: Any,
):
    monkeypatch.setattr(intercept.logger, "propagate", True)
    server.respond("crm.get_customer", value)

    def get_customer():
        raise AssertionError

    get_customer.__annotations__ = {"return": annotation}
    observed = observe(name="crm.get_customer")(get_customer)

    assert observed() == value
    [record] = [r for r in caplog.records if r.name == intercept.logger.name]
    assert record.levelname == "WARNING"
    assert record.getMessage().startswith(
        "observe, mock 'crm.get_customer': the mocked value doesn't match the return type "
    )
    [span] = span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes["lmnr.mock.outcome"] == "respond"
    assert json.loads(str(attributes["lmnr.span.output"])) == value


def test_an_unresolvable_annotation_returns_the_value_as_is(server: FakeServer):
    server.respond("crm.get_customer", {"id": "cus_1"})

    @observe(name="crm.get_customer")
    def get_customer() -> NotDefinedAnywhere:  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
        raise AssertionError

    assert get_customer() == {"id": "cus_1"}


# Async


@pytest.mark.asyncio
async def test_async_outcomes(server: FakeServer, span_exporter: InMemorySpanExporter):
    server.respond("crm.get_customer", {"id": "cus_1"})
    server.reply(
        "payments.refund",
        200,
        {"action": "raise", "error": {"type": "DuplicateRefund", "message": "again"}},
    )
    server.reply("search.web", 200, {"action": "passthrough"})

    @observe(name="crm.get_customer")
    async def get_customer(email: str) -> Customer:
        raise AssertionError

    @observe(name="payments.refund")
    async def refund() -> None:
        raise AssertionError

    @observe(name="search.web")
    async def search(q: str) -> str:
        await asyncio.sleep(0)
        return f"real {q}"

    assert await get_customer("a") == Customer(id="cus_1")
    with pytest.raises(MockedError, match="DuplicateRefund: again"):
        await refund()
    assert await search("x") == "real x"
    outcomes = [
        (span.name, (span.attributes or {})["lmnr.mock.outcome"])
        for span in span_exporter.get_finished_spans()
    ]
    assert outcomes == [
        ("crm.get_customer", "respond"),
        ("payments.refund", "raise"),
        ("search.web", "passthrough"),
    ]


def test_async_functions_work_across_event_loops(server: FakeServer):
    server.respond("crm.get_customer", 1)
    server.respond("crm.get_customer", 2)

    @observe(name="crm.get_customer")
    async def get_customer() -> int:
        raise AssertionError

    assert asyncio.run(get_customer()) == 1
    assert asyncio.run(get_customer()) == 2


# Unsupported shapes


def test_generators_cant_be_mocked(server: FakeServer):
    server.keys = ["gen", "agen"]

    @observe(name="gen")
    def gen() -> Generator[int, None, None]:
        yield 1

    @observe(name="agen")
    async def agen() -> AsyncGenerator[int, None]:
        yield 1

    @observe(name="unmocked")
    def unmocked() -> Generator[int, None, None]:
        yield 1

    for f in (gen, agen):
        with pytest.raises(MockedError, match="generators can't be mocked yet"):
            _ = f()
    assert list(unmocked()) == [1]
    assert server.calls() == []


@pytest.mark.asyncio
async def test_a_function_returning_an_iterator_gets_one(
    server: FakeServer, span_exporter: InMemorySpanExporter
):
    # Not generator functions: mocked like any other, and the list becomes
    # the annotated iterator (pydantic has no schema for these).
    server.keys = ["quotes", "skus", "aquotes", "async_def"]
    payload = [{"sku": "A", "price": 1}, {"sku": "B", "price": 2}]
    for key in server.keys:
        server.respond(key, payload)

    @observe(name="quotes")
    def quotes() -> Iterator[Quote]:
        raise AssertionError

    @observe(name="skus")
    def skus() -> Iterable[dict[str, Any]]:
        raise AssertionError

    @observe(name="aquotes")
    def aquotes() -> AsyncIterator[Quote]:
        raise AssertionError

    @observe(name="async_def")
    async def async_def() -> AsyncIterator[Quote]:
        raise AssertionError

    expected = [Quote(sku="A", price=1), Quote(sku="B", price=2)]
    iterator = quotes()
    assert next(iterator) == expected[0]
    assert list(iterator) == expected[1:]
    assert [s["sku"] for s in skus()] == ["A", "B"]
    assert [q async for q in aquotes()] == expected
    assert [q async for q in await async_def()] == expected
    # The span output is the items, as for a generator, not the iterator.
    outputs = {
        span.name: json.loads(str((span.attributes or {})["lmnr.span.output"]))
        for span in span_exporter.get_finished_spans()
    }
    assert outputs == dict.fromkeys(server.keys, payload)


def test_an_annotation_pydantic_cant_handle_returns_the_value_as_is_with_a_warning(
    server: FakeServer, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(intercept.logger, "propagate", True)
    server.respond("crm.get_customer", {"id": "cus_1"})

    @observe(name="crm.get_customer")
    def get_customer() -> Opaque:
        raise AssertionError

    assert get_customer() == {"id": "cus_1"}
    [record] = [r for r in caplog.records if r.name == intercept.logger.name]
    assert record.levelname == "WARNING"
    assert record.getMessage().startswith(
        "observe, mock 'crm.get_customer': mocked values can't be converted to the return type "
    )


# Public API


def test_mocked_error_is_exported_and_pickles():
    assert lmnr.MockedError is MockedError
    error = pickle.loads(pickle.dumps(MockedError("Kind", "message")))
    assert (error.type, error.message, str(error)) == ("Kind", "message", "Kind: message")
    assert isinstance(error, Exception)


# Conformance cases (spec/lmnr-mock/cases), against the reference server


def recording(ran: list[tuple[list[Any], dict[str, Any]]]) -> Callable[..., Any]:
    def real(*args: Any, **kwargs: Any) -> Any:
        ran.append((list(args), kwargs))
        return "real"

    return real


@pytest.mark.parametrize(
    "case_file",
    sorted(p.name for p in (SPEC_DIR / "cases").glob("*.json") if p.name != "mocks.json"),
)
def test_conformance_case(case_file: str, mock_env: Any):
    cases = SPEC_DIR / "cases"
    mapping = json.loads((cases / "mocks.json").read_text())
    saved_path = list(sys.path)
    try:
        reference = MockServer(
            {
                "version": 1,
                "token": "t" * 32,
                "src_dir": str(cases),
                "runner_file": None,
                "mocks": [
                    {
                        "key": key,
                        "type": "function",
                        "handler": {"kind": "python", "ref": f"lmnr_mock_reference:{name}"},
                    }
                    for key, name in mapping.items()
                ],
            }
        )
    finally:
        sys.path[:] = saved_path
    httpd = http_server(reference)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    mock_env(f"http://127.0.0.1:{httpd.server_address[1]}{PREFIX}")

    calls = [
        call
        for call in json.loads((cases / case_file).read_text())["calls"]
        if call["path"].startswith("/fn/")
        # What this SDK sends: an args array and a kwargs object, always,
        # and only for listed keys
        and call["body"]["lang"] == "python"
        and isinstance(call["body"].get("args"), list)
        and isinstance(call["body"].get("kwargs"), dict)
        and unquote(call["path"].removeprefix("/fn/")) in mapping
    ]
    if not calls:
        httpd.shutdown()
        pytest.skip("no function call this SDK would send")
    for call in calls:
        key = unquote(call["path"].removeprefix("/fn/"))
        args, kwargs = call["body"]["args"], call["body"]["kwargs"]
        ran: list[tuple[list[Any], dict[str, Any]]] = []
        observed = observe(name=key)(recording(ran))
        expected = call["sdk"]
        if "returns" in expected:
            assert observed(*args, **kwargs) == expected["returns"]
            assert ran == []
        elif "raises" in expected:
            with pytest.raises(MockedError) as raised:
                _ = observed(*args, **kwargs)
            assert raised.value.type == expected["raises"]["type"]
            assert raised.value.message == expected["raises"]["message"]
            assert ran == []
        elif "calls_real_function" in expected:
            assert observed(*args, **kwargs) == "real"
            assert ran == [(args, kwargs)]
        else:
            assert expected == {"fails": True}
            with pytest.raises(MockedError) as raised:
                _ = observed(*args, **kwargs)
            assert raised.value.type == MOCK_FAILURE
            assert ran == []
        # The server got the case's arguments.
        request = reference.calls[-1]["request"]
        assert (request["args"], request["kwargs"]) == (args, kwargs)
    httpd.shutdown()

"""Mocks (LAM-2588 phase 3): the mock server (`lmnr.evals.mocks.server`), the
handler references describe validates, the lifecycle `LaminarAgent` drives
(run on the host: commands in a local shell, the relay from this checkout),
`ctx.mocks`, and the `spec/lmnr-mock` conformance cases."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import pytest

from lmnr.evals import AgentContext, Case, EvalConfigError, VerifyContext, layer, relay
from lmnr.evals.context import HttpResponse, Secrets, relay_http
from lmnr.evals.describe import describe
from lmnr.evals.layer import LaminarLayer
from lmnr.evals.mocks import host
from lmnr.evals.mocks.server import MockServer, StartupError, http_server
from lmnr.evals.runner import find_runner, load_module

SPEC_DIR = Path(__file__).resolve().parent.parent / "spec" / "lmnr-mock"
TOKEN = "t" * 32
FORBIDDEN = ("mock", "eval", "test", "fixture")

HANDLERS = """
    import asyncio

    from lmnr.evals import EvalRunner, Image, mock
    from _shared.support import SupportRunner, get_customer


    def refund(call, ctx):
        if ctx.state.setdefault("refunded", False):
            raise mock.Error("DuplicateRefund", "charge already refunded")
        ctx.state["refunded"] = True
        return {"status": "ok", "index": call.index}


    def count(call, ctx):
        ctx.state["n"] = ctx.state.get("n", 0) + 1
        return {"index": call.index, "n": ctx.state["n"], "earlier": len(ctx.calls)}


    async def quote(call, ctx):
        await asyncio.sleep(0)
        return {"quote": 42}


    def passthrough(call, ctx):
        return mock.PASSTHROUGH


    def broken(call, ctx):
        raise ValueError("handler is broken")


    def wrong_kind(call, ctx):
        return mock.Response(200, json={})


    def zendesk(req, ctx):
        if req.method == "POST" and req.path.endswith("/comments"):
            return mock.Response(201, json={"ok": True, "got": req.json()})
        if req.path == "/api/v2/tickets/1":
            return {"id": 1, "query": req.query, "case": ctx.case.name, "env": ctx.env.get("TONE")}
        if req.path == "/text":
            return mock.Response(200, text="plain", headers={"X-Rate": "1"})
        if req.path == "/error":
            raise mock.Error("RateLimited", "slow down")
        if req.path == "/crash":
            raise KeyError("boom")
        if req.path == "/unmatched":
            raise mock.Unmatched()
        return None


    class RefundsEval(SupportRunner):
        mocks = {
            **SupportRunner.mocks,
            "payments.refund": refund,
            "counter.next": count,
            "pricing.quote": quote,
            "search.web": passthrough,
            "inventory.check": broken,
            "wrong.kind": wrong_kind,
            "https://api.zendesk.com/api": mock.http(zendesk, base_url_env="ZENDESK_URL"),
        }
"""

SHARED = """
    from lmnr.evals import EvalRunner, Image


    def get_customer(call, ctx):
        return ctx.case.fixture_json("crm.json").get(call.args[0])


    class SupportRunner(EvalRunner):
        image = Image.from_registry("python:3.13-slim")
        env = {"TONE": "formal"}
        mocks = {"crm.get_customer": get_customer}

        async def run(self, ctx):
            pass

        async def verify(self, ctx):
            return 1
"""


@pytest.fixture
def evals_root(tmp_path: Path):
    """`evals/_shared/support.py` (a runner with a handler) and
    `refunds/agent/eval.py`, which subclasses it and adds handlers."""
    root = tmp_path / "evals"
    (root / "_shared").mkdir(parents=True)
    _ = (root / "_shared" / "support.py").write_text(textwrap.dedent(SHARED))
    eval_path = root / "refunds" / "agent" / "eval.py"
    eval_path.parent.mkdir(parents=True)
    _ = eval_path.write_text(textwrap.dedent(HANDLERS))
    saved_path, saved_modules = list(sys.path), set(sys.modules)
    yield root
    sys.path[:] = saved_path
    for name in set(sys.modules) - saved_modules:
        del sys.modules[name]


def make_case(tmp_path: Path) -> Case:
    case_dir = tmp_path / "tasks" / "refund-1"
    (case_dir / "tests" / "fixtures" / "nested").mkdir(parents=True)
    _ = (case_dir / "task.toml").write_text('[task]\nname = "acme/refund-1"\n')
    _ = (case_dir / "tests" / "fixtures" / "crm.json").write_text(
        json.dumps({"ana@example.com": {"id": "cus_1", "plan": "pro"}})
    )
    _ = (case_dir / "tests" / "fixtures" / "nested" / "a.txt").write_text("nested")
    return Case.from_dir(case_dir)


def server_for(root: Path, case: Case) -> MockServer:
    """The server as the container starts it, minus the upload: the evals
    root is its `src_dir`."""
    runner = find_runner(load_module(root / "refunds" / "agent" / "eval.py"))
    return MockServer(
        {
            "version": 1,
            "token": TOKEN,
            "src_dir": str(root),
            "runner_file": "refunds/agent/eval.py",
            "fixtures_dir": str(case.dir / "tests" / "fixtures"),
            "case": {"name": case.name, "metadata": {}},
            "env": {"TONE": "formal"},
            "mocks": [spec.config() for spec in runner.mock_specs()],
            "ready_file": "/nonexistent",
        }
    )


def call_fn(server: MockServer, key: str, args: Any, **extra: Any) -> tuple[int, Any]:
    body = json.dumps({"args": args, "kwargs": {}, "lang": "python", **extra}).encode()
    status, _, payload = server.route(
        "POST", f"/_lmnr/v1/{TOKEN}/fn/{quote(key, safe='')}", {}, body
    )
    return status, json.loads(payload)


# The server


def test_function_outcomes(evals_root: Path, tmp_path: Path):
    server = server_for(evals_root, make_case(tmp_path))
    assert call_fn(server, "crm.get_customer", ["ana@example.com"]) == (
        200,
        {"action": "respond", "value": {"id": "cus_1", "plan": "pro"}},
    )
    assert call_fn(server, "crm.get_customer", ["bob@example.com"]) == (
        200,
        {"action": "respond", "value": None},
    )
    assert call_fn(server, "payments.refund", ["ch_1"]) == (
        200,
        {"action": "respond", "value": {"status": "ok", "index": 0}},
    )
    assert call_fn(server, "payments.refund", ["ch_1"]) == (
        200,
        {
            "action": "raise",
            "error": {"type": "DuplicateRefund", "message": "charge already refunded"},
        },
    )
    assert call_fn(server, "pricing.quote", []) == (
        200,
        {"action": "respond", "value": {"quote": 42}},
    )
    assert call_fn(server, "search.web", ["x"]) == (200, {"action": "passthrough"})
    assert call_fn(server, "inventory.check", []) == (
        500,
        {
            "action": "handler_error",
            "error": {"type": "ValueError", "message": "handler is broken"},
        },
    )
    status, body = call_fn(server, "wrong.kind", [])
    assert (status, body["action"], body["error"]["type"]) == (500, "handler_error", "TypeError")
    assert call_fn(server, "crm.delete", []) == (
        404,
        {"error": {"type": "UnknownMockKey", "message": "no function mock 'crm.delete'"}},
    )

    log = [
        json.loads(line)
        for line in server.route("GET", f"/_lmnr/v1/{TOKEN}/calls", {}, b"")[2].splitlines()
    ]
    assert [(c["seq"], c["key"], c["index"], c["outcome"]) for c in log] == [
        (0, "crm.get_customer", 0, "respond"),
        (1, "crm.get_customer", 1, "respond"),
        (2, "payments.refund", 0, "respond"),
        (3, "payments.refund", 1, "raise"),
        (4, "pricing.quote", 0, "respond"),
        (5, "search.web", 0, "passthrough"),
        (6, "inventory.check", 0, "handler_error"),
        (7, "wrong.kind", 0, "handler_error"),
        (8, "crm.delete", None, "unmatched"),
    ]
    assert log[3]["request"] == {"args": ["ch_1"], "kwargs": {}, "lang": "python"}
    assert log[3]["response"] == {
        "error": {"type": "DuplicateRefund", "message": "charge already refunded"}
    }
    assert log[5]["response"] == {}
    assert log[0]["ts"].endswith("Z") and log[0]["duration_ms"] >= 0
    # Handler tracebacks go to the server's own log, not to the caller.
    text = server.route("GET", f"/_lmnr/v1/{TOKEN}/log", {}, b"")[2].decode()
    assert "ValueError: handler is broken" in text


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"lang": "python"}, 'expected a JSON object with "args"'),
        ([], 'expected a JSON object with "args"'),
        ({"args": {"email": "a"}, "lang": "python"}, '"args" must be an array'),
        ({"args": None, "lang": "python"}, '"args" must be an array'),
        ({"args": [], "kwargs": [1], "lang": "python"}, '"kwargs" must be an object'),
        ({"args": [], "kwargs": None, "lang": "python"}, '"kwargs" must be an object'),
    ],
)
def test_function_calls_must_have_args_and_kwargs_of_the_right_type(
    evals_root: Path, tmp_path: Path, body: Any, message: str
):
    server = server_for(evals_root, make_case(tmp_path))
    status, _, payload = server.route(
        "POST", f"/_lmnr/v1/{TOKEN}/fn/crm.get_customer", {}, json.dumps(body).encode()
    )
    assert (status, json.loads(payload)) == (
        400,
        {"error": {"type": "BadRequest", "message": message}},
    )
    assert server.calls == []


def test_kwargs_default_to_an_empty_object(evals_root: Path, tmp_path: Path):
    server = server_for(evals_root, make_case(tmp_path))
    for body in (
        {"args": ["x"], "lang": "typescript"},
        {"args": ["x"], "kwargs": {"limit": 3}, "lang": "python"},
    ):
        status, _, _ = server.route(
            "POST", f"/_lmnr/v1/{TOKEN}/fn/search.web", {}, json.dumps(body).encode()
        )
        assert status == 200
    assert [c["request"] for c in server.calls] == [
        {"args": ["x"], "kwargs": {}, "lang": "typescript"},
        {"args": ["x"], "kwargs": {"limit": 3}, "lang": "python"},
    ]


def test_function_calls_carry_trace_ids(evals_root: Path, tmp_path: Path):
    server = server_for(evals_root, make_case(tmp_path))
    _ = call_fn(server, "search.web", [], trace_id="a" * 32, span_id="b" * 16)
    assert (server.calls[0]["trace_id"], server.calls[0]["span_id"]) == ("a" * 32, "b" * 16)


def http_call(
    server: MockServer,
    method: str,
    path: str,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    return server.route(method, f"/h/api.zendesk.com{path}", headers or {}, body)


def test_http_outcomes(evals_root: Path, tmp_path: Path):
    case = make_case(tmp_path)
    server = server_for(evals_root, case)
    status, headers, body = http_call(
        server,
        "POST",
        "/api/v2/tickets/1/comments",
        b'{"body": "hi"}',
        {"content-type": "application/json"},
    )
    assert (status, headers["Content-Type"]) == (201, "application/json")
    assert json.loads(body) == {"ok": True, "got": {"body": "hi"}}
    status, _, body = http_call(server, "GET", "/api/v2/tickets/1?include=users&include=x")
    assert (status, json.loads(body)) == (
        200,
        {"id": 1, "query": {"include": ["users", "x"]}, "case": "acme/refund-1", "env": "formal"},
    )
    status, headers, body = http_call(server, "GET", "/text")
    assert (status, body, headers["X-Rate"]) == (200, b"plain", "1")
    assert headers["Content-Type"].startswith("text/plain")
    for path, expected_status, error_type in [
        ("/error", 500, "RateLimited"),
        ("/crash", 500, "KeyError"),
        ("/unmatched", 501, "Unmatched"),
        ("/anything-else", 501, "Unmatched"),
    ]:
        status, _, body = http_call(server, "GET", path)
        assert (status, json.loads(body)["error"]["type"]) == (expected_status, error_type), path
    # Hosts are case-insensitive; an unknown host is a plain 404, not logged.
    assert server.route("GET", "/h/API.Zendesk.com/text", {}, b"")[0] == 200
    assert server.route("GET", "/h/api.stripe.com/v1/charges", {}, b"")[0] == 404

    outcomes = [(c["key"], c["index"], c["outcome"]) for c in server.calls]
    key = "https://api.zendesk.com/api"
    assert outcomes == [
        (key, 0, "respond"),
        (key, 1, "respond"),
        (key, 2, "respond"),
        (key, 3, "raise"),
        (key, 4, "handler_error"),
        (key, 5, "unmatched"),
        (key, 6, "unmatched"),
        (key, 7, "respond"),
    ]
    first = server.calls[0]
    assert first["request"] == {
        "method": "POST",
        "path": "/api/v2/tickets/1/comments",
        "query": {},
        "headers": {"content-type": "application/json"},
        "body_text": '{"body": "hi"}',
    }
    assert first["response"]["status"] == 201
    assert json.loads(first["response"]["body_text"]) == {"ok": True, "got": {"body": "hi"}}


def test_http_calls_take_trace_ids_from_traceparent_and_log_binary_bodies(
    evals_root: Path, tmp_path: Path
):
    server = server_for(evals_root, make_case(tmp_path))
    traceparent = f"00-{'c' * 32}-{'d' * 16}-01"
    _ = http_call(server, "PUT", "/x", b"\xff\x00", {"traceparent": traceparent})
    call = server.calls[0]
    assert (call["trace_id"], call["span_id"]) == ("c" * 32, "d" * 16)
    assert call["request"]["body_b64"] == "/wA="


def test_token_and_routes_reveal_nothing(evals_root: Path, tmp_path: Path):
    server = server_for(evals_root, make_case(tmp_path))
    assert server.route("GET", f"/_lmnr/v1/{TOKEN}/health", {}, b"")[0] == 200
    keys = json.loads(server.route("GET", f"/_lmnr/v1/{TOKEN}/keys", {}, b"")[2])["keys"]
    # Function keys only.
    assert keys == sorted(
        [
            "counter.next",
            "crm.get_customer",
            "inventory.check",
            "payments.refund",
            "pricing.quote",
            "search.web",
            "wrong.kind",
        ]
    )
    not_found = (404, {"Content-Type": "text/plain; charset=utf-8"}, b"Not Found")
    for method, path in [
        ("GET", f"/_lmnr/v1/{'x' * 32}/health"),
        ("GET", f"/_lmnr/v1/{'x' * 32}/calls"),
        ("POST", f"/_lmnr/v1/{'x' * 32}/fn/crm.get_customer"),
        ("GET", "/_lmnr/v1/health"),
        ("GET", f"/_lmnr/v1/{TOKEN}/mocks"),
        ("GET", f"/_lmnr/v1/{TOKEN}/handlers"),
        ("GET", f"/_lmnr/v1/{TOKEN}/fixtures"),
        ("GET", f"/_lmnr/v1/{TOKEN}/config"),
        ("GET", f"/_lmnr/v1/{TOKEN}/fn/crm.get_customer"),
        ("GET", "/"),
        ("GET", "/h/"),
    ]:
        assert server.route(method, path, {}, b"") == not_found, path
    assert call_fn(server, "crm.get_customer", ["x"])[0] == 200
    assert server.route("POST", f"/_lmnr/v1/{TOKEN}/fn/crm.get_customer", {}, b"{")[0] == 400


def test_fixtures_are_served_from_memory(evals_root: Path, tmp_path: Path):
    case = make_case(tmp_path)
    server = server_for(evals_root, case)
    assert server.fixtures["nested/a.txt"] == b"nested"
    shutil.rmtree(case.dir / "tests" / "fixtures")
    assert call_fn(server, "crm.get_customer", ["ana@example.com"])[1]["value"] == {
        "id": "cus_1",
        "plan": "pro",
    }


def test_host_and_handler_read_fixtures_the_same_way(evals_root: Path, tmp_path: Path):
    case = make_case(tmp_path)
    handler_case = server_for(evals_root, case).case
    for reader in (case, handler_case):
        assert reader.fixture_bytes("nested/a.txt") == b"nested"
        assert reader.fixture_text("nested/a.txt") == "nested"
        assert reader.fixture_json("crm.json") == {
            "ana@example.com": {"id": "cus_1", "plan": "pro"}
        }
        with pytest.raises(
            FileNotFoundError, match=r"^The case 'acme/refund-1' has no tests/fixtures/x\.json\.$"
        ):
            _ = reader.fixture_json("x.json")
    assert case.fixture_path("crm.json") == case.dir / "tests" / "fixtures" / "crm.json"
    assert not hasattr(handler_case, "fixture_path")


def test_concurrent_calls_are_serialized(evals_root: Path, tmp_path: Path):
    """Real HTTP from many threads: each call sees a distinct index and the
    state of every call before it."""
    server = server_for(evals_root, make_case(tmp_path))
    httpd = http_server(server)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/_lmnr/v1/{TOKEN}/fn/counter.next"

    def one(_: int) -> dict[str, Any]:
        request = urllib.request.Request(
            url, data=json.dumps({"args": [], "lang": "python"}).encode(), method="POST"
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())["value"]

    try:
        with ThreadPoolExecutor(16) as pool:
            values = list(pool.map(one, range(64)))
    finally:
        httpd.shutdown()
    assert sorted(v["index"] for v in values) == list(range(64))
    assert all(v["n"] == v["index"] + 1 == v["earlier"] + 1 for v in values)
    assert [c["index"] for c in server.calls] == list(range(64))


def test_startup_errors_are_clear(evals_root: Path):
    base = {
        "version": 1,
        "token": TOKEN,
        "src_dir": str(evals_root),
        "runner_file": "refunds/agent/eval.py",
        "fixtures_dir": None,
        "mocks": [],
    }
    with pytest.raises(StartupError, match="unknown handler kind 'command'"):
        _ = MockServer(
            {**base, "mocks": [{"key": "a", "type": "function", "handler": {"kind": "command"}}]}
        )
    with pytest.raises(StartupError, match="version 2"):
        _ = MockServer({**base, "version": 2})
    bad = evals_root / "bad" / "agent" / "eval.py"
    bad.parent.mkdir(parents=True)
    _ = bad.write_text("import anthropic_not_installed\n")
    with pytest.raises(StartupError, match="mock_dependencies"):
        _ = MockServer({**base, "runner_file": "bad/agent/eval.py"})


# Validation (describe)


def write_runner(root: Path, body: str, dataset: str = "support") -> Path:
    path = root / dataset / "agent" / "eval.py"
    path.parent.mkdir(parents=True)
    _ = path.write_text(textwrap.dedent(body))
    return path


RUNNER_TAIL = """
    class E(EvalRunner):
        image = Image.from_registry("python:3.13-slim")
        mocks = MOCKS
        run_command = "true"
        grade_with_test_sh = True
"""


@pytest.mark.parametrize(
    ("setup", "mocks_expr", "message"),
    [
        ("", '{"crm.get": lambda call, ctx: 1}', "a lambda can't be imported"),
        (
            "def make():\n    def inner(call, ctx):\n        return 1\n    return inner\n",
            '{"crm.get": make()}',
            "make.<locals>.inner is a method or a nested function",
        ),
        (
            "class H:\n    @staticmethod\n    def get(call, ctx):\n        return 1\n",
            '{"crm.get": H.get}',
            "H.get is a method",
        ),
        ("import json\n", '{"crm.get": json.dumps}', "outside the evals root"),
        ("", '{"crm.get": "get_customer"}', "must be a function"),
        ("def h(call, ctx):\n    return 1\n", '{"": h}', "is not a mock key"),
        ("def h(call, ctx):\n    return 1\n", '{" crm.get": h}', "is not a mock key"),
        ("def h(call, ctx):\n    return 1\n", '{"https://api.x.com": h}', "wrap an HTTP mock"),
        (
            "def h(call, ctx):\n    return 1\n",
            '{"crm.get": mock.http(h, base_url_env="X_URL")}',
            "key is the API's base URL",
        ),
        (
            "def h(call, ctx):\n    return 1\n",
            '{"https://api.x.com?a=1": mock.http(h, base_url_env="X_URL")}',
            "without credentials, query",
        ),
        (
            "def h(call, ctx):\n    return 1\n",
            '{"https://api.x.com": mock.http(h, base_url_env="X-URL")}',
            "base_url_env must be a variable name",
        ),
        (
            "def h(call, ctx):\n    return 1\n",
            '{"https://api.x.com": mock.http(h, base_url_env="A"), '
            + '"http://API.x.com/v2": mock.http(h, base_url_env="B")}',
            "one HTTP mock per host",
        ),
        (
            "def h(call, ctx):\n    return 1\n",
            '{"https://a.x.com": mock.http(h, base_url_env="A"), '
            + '"https://b.x.com": mock.http(h, base_url_env="A")}',
            "already sets A",
        ),
        ("def h(call, ctx):\n    return 1\n", '{"crm.get": h}', ""),
    ],
)
def test_describe_rejects_handlers_the_container_cant_import(
    tmp_path: Path, setup: str, mocks_expr: str, message: str
):
    header = "from lmnr.evals import EvalRunner, Image, mock\n" + setup
    body = textwrap.dedent(RUNNER_TAIL).replace("MOCKS", mocks_expr)
    path = write_runner(tmp_path / "evals", header + body)
    if not message:
        assert describe(path)["mocks"] == [
            {"key": "crm.get", "type": "function", "handler": "__runner__:h"}
        ]
        return
    with pytest.raises(EvalConfigError, match=message):
        _ = describe(path)


def test_base_url_env_cannot_shadow_other_env(tmp_path: Path):
    path = write_runner(
        tmp_path / "evals",
        """
        from lmnr.evals import EvalRunner, Image, mock

        def h(req, ctx):
            return None

        class E(EvalRunner):
            image = Image.from_registry("python:3.13-slim")
            env = {"API_URL": "https://real"}
            mocks = {"https://api.x.com": mock.http(h, base_url_env="API_URL")}
            run_command = "true"
            grade_with_test_sh = True
        """,
    )
    with pytest.raises(EvalConfigError, match="API_URL is also set by E.env"):
        _ = describe(path)


# mock_dependencies: the evals project's dependencies by default

PYPROJECT = """\
[project]
name = "evals"
version = "0"
dependencies = [
    "anthropic>=2.4.0",
    "lmnr[all]>=0.8",
    "pydantic",
    "shared-lib",
    "local-pkg @ file:///srv/local-pkg",
]

[tool.uv.sources]
shared-lib = { path = "../shared-lib", editable = true }
"""

SCRIPT = """\
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx>=0.27",
#   "anthropic>=2.4.0",
#   "./vendor/thing",
# ]
# ///
"""


def dependencies_runner(
    root: Path,
    *,
    pyproject: str | None = None,
    script: str = "",
    mocks: bool = True,
    deps: str | None = None,
) -> Path:
    if pyproject is not None:
        root.mkdir(parents=True, exist_ok=True)
        _ = (root / "pyproject.toml").write_text(pyproject)
    lines = [
        script,
        "from lmnr.evals import EvalRunner, Image, mock",
        "",
        "def h(call, ctx):",
        "    return None",
        "",
        "class E(EvalRunner):",
        '    image = Image.from_registry("python:3.13-slim")',
        '    run_command = "true"',
        "    grade_with_test_sh = True",
    ]
    if mocks:
        lines.append('    mocks = {"crm.get": h}')
    if deps is not None:
        lines.append(f"    mock_dependencies = {deps}")
    return write_runner(root, "\n".join(lines) + "\n")


def layer_install_line(result: dict[str, Any]) -> str:
    text = result["images"]["support-evals"]["dockerfile_text"]
    return next(line for line in text.splitlines() if line.startswith("RUN uv pip install"))


def test_mock_dependencies_default_to_the_pyproject(tmp_path: Path):
    path = dependencies_runner(tmp_path / "evals", pyproject=PYPROJECT)
    result = describe(path)
    # lmnr is dropped silently: the layer pins it.
    assert result["mock_dependencies"] == ["anthropic>=2.4.0", "pydantic"]
    assert layer_install_line(result).endswith(" 'anthropic>=2.4.0' pydantic")
    [uv_source, local_path] = result["warnings"]
    assert "'shared-lib' from pyproject.toml" in uv_source
    assert "[tool.uv.sources]" in uv_source
    assert (
        'add an installable spec to mock_dependencies: [mock.PROJECT_DEPENDENCIES, "shared-lib'
        in (uv_source)
    )
    assert "'local-pkg @ file:///srv/local-pkg' from pyproject.toml" in local_path
    assert "a local path" in local_path


def test_mock_dependencies_default_to_the_pep_723_block(tmp_path: Path):
    path = dependencies_runner(tmp_path / "evals", script=SCRIPT)
    result = describe(path)
    assert result["mock_dependencies"] == ["httpx>=0.27", "anthropic>=2.4.0"]
    [warning] = result["warnings"]
    assert "'./vendor/thing' from eval.py's # /// script block" in warning


def test_mock_dependencies_default_to_both_without_duplicates(tmp_path: Path):
    path = dependencies_runner(tmp_path / "evals", pyproject=PYPROJECT, script=SCRIPT)
    result = describe(path)
    assert result["mock_dependencies"] == ["anthropic>=2.4.0", "pydantic", "httpx>=0.27"]
    assert len(result["warnings"]) == 3


def test_project_dependencies_marker_expands_in_place(tmp_path: Path):
    path = dependencies_runner(
        tmp_path / "evals",
        pyproject=PYPROJECT,
        deps='["extra>=1", mock.PROJECT_DEPENDENCIES, "lmnr==0.1", "shared-lib>=1"]',
    )
    result = describe(path)
    assert result["mock_dependencies"] == [
        "extra>=1",
        "anthropic>=2.4.0",
        "pydantic",
        "shared-lib>=1",
    ]
    # shared-lib has an installable spec now; local-pkg still warns.
    [warning] = result["warnings"]
    assert "local-pkg" in warning


def test_a_mock_dependencies_list_replaces_the_default(tmp_path: Path):
    path = dependencies_runner(
        tmp_path / "evals", pyproject=PYPROJECT, script=SCRIPT, deps='["httpx"]'
    )
    result = describe(path)
    assert (result["mock_dependencies"], result["warnings"]) == (["httpx"], [])
    path = dependencies_runner(tmp_path / "empty", pyproject=PYPROJECT, deps="[]")
    assert describe(path)["mock_dependencies"] == []


@pytest.mark.parametrize(
    "pyproject",
    [None, "[tool.ruff]\nline-length = 100\n", '[project]\nname = "evals"\nversion = "0"\n'],
)
def test_no_project_dependencies(tmp_path: Path, pyproject: str | None):
    path = dependencies_runner(tmp_path / "evals", pyproject=pyproject)
    result = describe(path)
    assert (result["mock_dependencies"], result["warnings"]) == ([], [])
    assert layer_install_line(result).endswith(f"lmnr=={importlib.metadata.version('lmnr')}")


def test_runners_without_mocks_install_no_dependencies(tmp_path: Path):
    path = dependencies_runner(tmp_path / "evals", pyproject=PYPROJECT, mocks=False)
    result = describe(path)
    assert (result["mock_dependencies"], result["warnings"]) == ([], [])
    assert layer_install_line(result).endswith(f"lmnr=={importlib.metadata.version('lmnr')}")
    path = dependencies_runner(tmp_path / "explicit", mocks=False, deps='["httpx"]')
    result = describe(path)
    assert result["mock_dependencies"] == []
    assert result["warnings"] == [
        "E.mock_dependencies is ignored: the runner has no mocks, so nothing imports eval.py "
        + "in the container."
    ]


@pytest.mark.parametrize(
    ("pyproject", "script", "deps", "message"),
    [
        (None, "", '"httpx"', "must be a list of pip requirements"),
        (None, "", "[1]", "must be a list of pip requirements"),
        (None, "", '[""]', "must be a list of pip requirements"),
        (None, "", "[mock.PROJECT_DEPENDENCIES, None]", "must be a list of pip requirements"),
        ("[project\n", "", None, "pyproject.toml is not valid TOML"),
        ('[project]\ndependencies = "x"\n', "", None, "must be a list of requirement strings"),
        (None, SCRIPT + "\n" + SCRIPT, None, "more than one `# /// script` block"),
        (None, "# /// script\n# dependencies = [\n# ///\n", None, "is not valid TOML"),
    ],
)
def test_invalid_mock_dependencies_fail_describe(
    tmp_path: Path, pyproject: str | None, script: str, deps: str | None, message: str
):
    path = dependencies_runner(tmp_path / "evals", pyproject=pyproject, script=script, deps=deps)
    with pytest.raises(EvalConfigError, match=re.escape(message)):
        _ = describe(path)


def test_describe_lists_mocks_and_the_layer(evals_root: Path):
    result = describe(evals_root / "refunds" / "agent" / "eval.py", layer_lmnr="wheel")
    assert result["mocks"][0] == {
        "key": "crm.get_customer",
        "type": "function",
        # Defined in the shared module, so the container imports it from there.
        "handler": "_shared.support:get_customer",
    }
    assert result["mocks"][-1] == {
        "key": "https://api.zendesk.com/api",
        "type": "http",
        "handler": "__runner__:zendesk",
        "base_url_env": "ZENDESK_URL",
    }
    assert result["modules"] == ["_shared/support.py", "refunds/agent/eval.py"]
    assert result["laminar_layer"] == {
        "python": "3.13",
        "lmnr": None,
        "build_contexts": {"lmnr-dev": "lmnr wheel"},
    }
    text = result["images"]["refunds-evals"]["dockerfile_text"]
    assert "COPY --from=lmnr-dev / /tmp/lmnr-dist/" in text
    assert (
        "RUN uv pip install --python /opt/lmnr/venv/bin/python "
        + '--python-platform "$(uname -m)-manylinux_2_36" /tmp/lmnr-dist/*.whl'
    ) in text


def test_describe_cli_takes_the_layer_requirement(evals_root: Path):
    path = evals_root / "refunds" / "agent" / "eval.py"
    spec = "lmnr @ git+https://github.com/lmnr-ai/lmnr-python@feat/x"
    out = subprocess.run(
        [sys.executable, "-m", "lmnr.evals.describe", str(path), f"--layer-lmnr={spec}"],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(out.stdout)
    assert result["laminar_layer"]["lmnr"] == spec
    text = result["images"]["refunds-evals"]["dockerfile_text"]
    assert "apt-get install -y --no-install-recommends git" in text
    assert f"'{spec}'" in text


# The lifecycle, on the host


# How `ctx.container.sh` runs a command: `relay.py exec` reading its payload file.
RELAY_EXEC = re.compile(r"^\{ rm -f (\S+) 2>/dev/null; .+ exec; \} < \S+$")


class LocalEnvironment:
    """Runs commands on the host with bash (users are ignored). `commands`
    has a `relay.py exec` as the command it runs."""

    def __init__(self):
        self.commands: list[str] = []

    async def exec(self, command: str, cwd=None, env=None, timeout_sec=None, user=None) -> Any:
        relayed = RELAY_EXEC.match(command)
        self.commands.append(
            json.loads(Path(relayed[1]).read_text())["cmd"] if relayed else command
        )
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return SimpleNamespace(
            stdout=out.decode(), stderr=err.decode(), return_code=proc.returncode
        )

    async def upload_file(self, source_path, target_path) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        _ = Path(target_path).write_bytes(Path(source_path).read_bytes())

    async def upload_dir(self, source_dir, target_dir) -> None:
        _ = await asyncio.to_thread(
            subprocess.run, ["cp", "-a", f"{source_dir}/.", target_dir], check=True
        )

    async def download_file(self, source_path, target_path) -> None:
        _ = Path(target_path).write_bytes(Path(source_path).read_bytes())


@pytest.fixture
def host_relay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(layer, "RELAY_COMMAND", f"{sys.executable} -I {relay.__file__}")
    artifacts = tmp_path / "container" / "logs" / "artifacts"
    monkeypatch.setattr(host, "ARTIFACTS_DIR", str(artifacts))
    return artifacts


def test_lifecycle_in_a_container_like_shell(evals_root: Path, tmp_path: Path, host_relay: Path):
    eval_path = evals_root / "refunds" / "agent" / "eval.py"
    runner = find_runner(load_module(eval_path))
    env = LocalEnvironment()
    case = make_case(tmp_path)
    ctx = AgentContext(
        environment=env,
        case=case,
        runner_dir=eval_path.parent,
        env={"TONE": "formal"},
        host_logs_dir=tmp_path / "trial" / "agent",
    )
    # What the agent under test might plant before collect.
    host_relay.mkdir(parents=True)
    _ = (host_relay / "mocks.jsonl").write_text('{"planted": true}\n')

    async def scenario() -> host.RunningMocks:
        running = await host.start_mock_server(ctx, runner.mock_specs(), eval_path=eval_path)
        ctx.add_env(running.env)
        # The uploaded files are gone before setup.
        temp = next(c.split()[-1] for c in env.commands if c.startswith("mkdir -m 700 "))
        assert not Path(temp).exists()
        # The process: named `relay`, nothing telling in its arguments.
        comm = Path(f"/proc/{running.pid}/comm").read_text().strip()
        args = Path(f"/proc/{running.pid}/cmdline").read_bytes().split(b"\0")
        after_script = b" ".join(args[args.index(relay.__file__.encode()) + 1 :]).decode()
        assert comm == "relay"
        assert after_script.startswith("serve /tmp/.")
        assert not any(word in after_script for word in FORBIDDEN), after_script

        url = running.env["LMNR_MOCK_URL"]
        assert url.startswith("http://127.0.0.1:") and len(url.rsplit("/", 1)[1]) == 32
        assert running.env["ZENDESK_URL"].endswith("/h/api.zendesk.com/api")
        # The app's two routes, from a command that got the env.
        fn = await relay_http(
            ctx.container,
            "POST",
            f"{url}/fn/crm.get_customer",
            body=json.dumps({"args": ["ana@example.com"], "lang": "python"}).encode(),
        )
        assert fn.json() == {"action": "respond", "value": {"id": "cus_1", "plan": "pro"}}
        out = await ctx.container.sh(
            f'{sys.executable} -c "import os, urllib.request; print(urllib.request.urlopen('
            + "os.environ['ZENDESK_URL'] + '/v2/tickets/1').read().decode())\""
        )
        assert json.loads(out.stdout)["case"] == "acme/refund-1"
        await host.collect_mocks(ctx, running)
        return running

    running = asyncio.run(scenario())
    lines = (host_relay / "mocks.jsonl").read_text().splitlines()
    assert [json.loads(line)["key"] for line in lines] == [
        "crm.get_customer",
        "https://api.zendesk.com/api",
    ]
    assert "listening on 127.0.0.1" in (tmp_path / "trial" / "agent" / "mocks.log").read_text()
    deadline = 50
    while relay.alive(running.pid) and deadline:
        deadline -= 1
        asyncio.run(asyncio.sleep(0.1))
    assert not relay.alive(running.pid)

    # verify reads it back from the trial's artifacts.
    artifacts = tmp_path / "trial" / "artifacts" / "logs" / "artifacts"
    artifacts.mkdir(parents=True)
    _ = shutil.copy(host_relay / "mocks.jsonl", artifacts / "mocks.jsonl")
    verify_ctx = VerifyContext(
        environment=env,
        case=case,
        runner_dir=eval_path.parent,
        host_agent_dir=tmp_path / "trial" / "agent",
        secrets=Secrets([], {}),
        mock_keys=list(runner.mocks),
    )
    assert verify_ctx.mocks.calls("crm.get_customer")[0]["request"] == {
        "args": ["ana@example.com"],
        "kwargs": {},
        "lang": "python",
    }
    assert verify_ctx.mocks.calls("payments.refund") == []
    assert len(verify_ctx.mocks.all()) == 2
    assert verify_ctx.mocks.unmatched() == []
    with pytest.raises(KeyError, match="not in the runner's mocks"):
        _ = verify_ctx.mocks.calls("payments.refunds")


def test_a_server_that_fails_to_start_raises_with_its_output(
    evals_root: Path, tmp_path: Path, host_relay: Path
):
    eval_path = evals_root / "refunds" / "agent" / "eval.py"
    runner = find_runner(load_module(eval_path))
    env = LocalEnvironment()
    ctx = AgentContext(
        environment=env,
        case=make_case(tmp_path),
        runner_dir=eval_path.parent,
        host_logs_dir=tmp_path / "trial" / "agent",
    )
    # The container's copy of eval.py fails to import (e.g. a missing package).
    _ = eval_path.write_text(eval_path.read_text() + "\nimport not_installed_anywhere\n")
    with pytest.raises(RuntimeError, match="The mock server exited") as info:
        _ = asyncio.run(host.start_mock_server(ctx, runner.mock_specs(), eval_path=eval_path))
    assert "mock_dependencies" in str(info.value)
    temp = next(c.split()[-1] for c in env.commands if c.startswith("mkdir -m 700 "))
    assert not Path(temp).exists()


def test_a_server_that_fails_its_health_check_is_killed(
    evals_root: Path, tmp_path: Path, host_relay: Path, monkeypatch: pytest.MonkeyPatch
):
    """Started, then failed: collect_mocks never gets it, so start stops it."""
    eval_path = evals_root / "refunds" / "agent" / "eval.py"
    runner = find_runner(load_module(eval_path))
    env = LocalEnvironment()
    ctx = AgentContext(
        environment=env,
        case=make_case(tmp_path),
        runner_dir=eval_path.parent,
        host_logs_dir=tmp_path / "trial" / "agent",
    )

    async def unhealthy(*args: Any, **kwargs: Any) -> HttpResponse:
        return HttpResponse(503, {}, b"")

    monkeypatch.setattr(host, "relay_http", unhealthy)
    with pytest.raises(RuntimeError, match="answered 503 to its health check"):
        _ = asyncio.run(host.start_mock_server(ctx, runner.mock_specs(), eval_path=eval_path))
    kill = next(c for c in env.commands if c.startswith("kill "))
    pid = int(kill.split()[1])
    deadline = 50
    while relay.alive(pid) and deadline:
        deadline -= 1
        time.sleep(0.1)
    assert not relay.alive(pid)
    temp = next(c.split()[-1] for c in env.commands if c.startswith("mkdir -m 700 "))
    assert not Path(temp).exists()


@pytest.mark.parametrize(
    ("lmnr", "deps"),
    [
        ("lmnr @ git+https://github.com/lmnr-ai/lmnr-python@feat/x", ()),
        ("lmnr==0.8.0", ("crm-sdk @ git+https://github.com/acme/crm-sdk@v2",)),
        (None, ("crm-sdk @ git+https://github.com/acme/crm-sdk@v2",)),
    ],
)
def test_layer_installs_git_for_any_vcs_requirement(lmnr: str | None, deps: tuple[str, ...]):
    text = "\n".join(LaminarLayer(lmnr, deps).bundle_stages(str))
    assert "apt-get install -y --no-install-recommends git" in text
    assert all(shlex.quote(d) in text for d in deps)


def test_layer_skips_git_without_vcs_requirements():
    text = "\n".join(LaminarLayer("lmnr==0.8.0", ("anthropic>=2",)).bundle_stages(str))
    assert "git" not in text


def test_mock_calls_without_mocks_or_log(tmp_path: Path):
    case = make_case(tmp_path)

    def ctx(keys: list[str]) -> VerifyContext:
        return VerifyContext(
            environment=LocalEnvironment(),
            case=case,
            runner_dir=tmp_path,
            host_agent_dir=tmp_path / "trial" / "agent",
            secrets=Secrets([], {}),
            mock_keys=keys,
        )

    with pytest.raises(EvalConfigError, match="declares no mocks"):
        _ = ctx([]).mocks.all()
    with pytest.raises(FileNotFoundError, match="collect didn't save"):
        _ = ctx(["a"]).mocks.calls("a")


# Conformance cases (spec/lmnr-mock/cases)


@pytest.mark.parametrize(
    "case_file",
    sorted(p.name for p in (SPEC_DIR / "cases").glob("*.json") if p.name != "mocks.json"),
)
def test_conformance_case(case_file: str):
    cases = SPEC_DIR / "cases"
    mapping = json.loads((cases / "mocks.json").read_text())
    saved_path = list(sys.path)
    try:
        server = MockServer(
            {
                "version": 1,
                "token": TOKEN,
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
    scenario = json.loads((cases / case_file).read_text())
    assert scenario["description"]
    for call in scenario["calls"]:
        body = json.dumps(call["body"]).encode() if "body" in call else b""
        status, _, payload = server.route(
            call["method"], f"/_lmnr/v1/{TOKEN}{call['path']}", {}, body
        )
        assert (status, json.loads(payload)) == (call["status"], call["response"]), call["path"]
        if call["path"].startswith("/fn/"):
            assert len(call["sdk"]) == 1
            assert next(iter(call["sdk"])) in ("returns", "raises", "calls_real_function", "fails")

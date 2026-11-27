"""The mock server: one per trial, inside the agent container (`lmnr-mock` v1).

Started by the runner before `setup` as `relay.py serve <config>` (the Laminar
layer's Python), it loads everything it needs into memory (the runner's
modules, the case's fixtures, the config), and the runner deletes those files
before the agent under test starts. The call log and the server's own log
stay in memory until `collect` fetches them. The wire protocol is
`spec/lmnr-mock/lmnr-mock.md`; change it only together with the SDKs.

Config (JSON, the only argument):

    {"version": 1, "token": "...", "src_dir": "<tmp>/src",
     "runner_file": "refunds/agent/eval.py", "fixtures_dir": "<tmp>/fixtures",
     "case": {"name": "...", "metadata": {}}, "env": {},
     "mocks": [{"key": "crm.get_customer", "type": "function",
                "handler": {"kind": "python", "ref": "_shared.support:get_customer"}},
               {"key": "https://api.zendesk.com", "type": "http", "host": "api.zendesk.com",
                "handler": {"kind": "python", "ref": "__runner__:zendesk"}}],
     "ready_file": "<tmp>/ready"}
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import parse_qs, unquote, urlsplit

from typing_extensions import override

from lmnr.evals.mock import (
    PASSTHROUGH,
    Call,
    CaseInfo,
    Context,
    Error,
    Request,
    Response,
    Unmatched,
)

PROTOCOL_PREFIX = "/_lmnr/v1/"
HTTP_PREFIX = "/h/"
RUNNER_MODULE = "__runner__"
CONFIG_VERSION = 1
_TRACEPARENT = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")

Reply = tuple[int, dict[str, str], bytes]
Bound = Callable[[Any, Context], Any]


class StartupError(Exception):
    """The config, a module or a handler can't be loaded."""


# Handler kinds: v1 has "python" (a module-level function by reference).
# Later kinds (a declared response, a command) plug in here.


def _python_handler(spec: dict[str, Any]) -> Bound:
    ref = spec.get("ref")
    if not isinstance(ref, str) or ":" not in ref:
        raise StartupError(f"python handler ref must be '<module>:<function>', got {ref!r}")
    module_name, _, name = ref.partition(":")
    try:
        module = sys.modules[module_name] if module_name == RUNNER_MODULE else None
        module = module or importlib.import_module(module_name)
    except Exception as e:
        raise StartupError(f"can't import {module_name} for {ref}: {e}") from e
    handler = getattr(module, name, None)
    if not callable(handler):
        raise StartupError(f"{ref} is not a function")
    return handler


HANDLER_KINDS: dict[str, Callable[[dict[str, Any]], Bound]] = {"python": _python_handler}


class Mock:
    def __init__(self, entry: dict[str, Any]):
        self.key: str = str(entry["key"])
        self.type: str = str(entry["type"])
        self.host: str | None = str(entry["host"]).lower() if entry.get("host") else None
        handler = entry.get("handler") or {}
        kind = handler.get("kind")
        factory = HANDLER_KINDS.get(kind) if isinstance(kind, str) else None
        if factory is None:
            raise StartupError(
                f"mock {self.key!r}: unknown handler kind {kind!r} "
                + f"(this lmnr knows {', '.join(sorted(HANDLER_KINDS))})"
            )
        self.handler: Bound = factory(handler)


class MockServer:
    """Routing and handlers, without sockets (`serve` puts it behind HTTP)."""

    def __init__(self, config: dict[str, Any]):
        if config.get("version") != CONFIG_VERSION:
            raise StartupError(f"config version {config.get('version')!r} is not supported")
        self.token: str = str(config["token"])
        self.env: dict[str, str] = {str(k): str(v) for k, v in (config.get("env") or {}).items()}
        case = config.get("case") or {}
        self.case: CaseInfo = CaseInfo(
            name=str(case.get("name", "")), metadata=dict(case.get("metadata") or {})
        )
        self.fixtures: dict[str, bytes] = _load_fixtures(config.get("fixtures_dir"))
        self.state: dict[str, Any] = {}
        self.log_lines: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self._lock: threading.Lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._indexes: dict[str, int] = {}

        src_dir = config.get("src_dir")
        if src_dir:
            # Shared modules (`_shared.support`) resolve as on the host.
            if src_dir not in sys.path:
                sys.path.append(str(src_dir))
            runner_file = config.get("runner_file")
            if runner_file:
                _load_runner(Path(src_dir) / str(runner_file))
        self.functions: dict[str, Mock] = {}
        self.hosts: dict[str, Mock] = {}
        for entry in config.get("mocks") or []:
            item = Mock(entry)
            if item.type == "function":
                self.functions[item.key] = item
            elif item.type == "http" and item.host:
                self.hosts[item.host] = item
            else:
                raise StartupError(f"mock {item.key!r}: unknown type {item.type!r}")
        self.log(
            f"loaded {len(self.functions)} function and {len(self.hosts)} HTTP handlers, "
            + f"{len(self.fixtures)} files"
        )

    def log(self, line: str) -> None:
        stamp = _timestamp()
        self.log_lines.append(f"{stamp} {line}")

    # Routing

    def route(self, method: str, target: str, headers: dict[str, str], body: bytes) -> Reply:
        parts = urlsplit(target)
        path = parts.path
        if path.startswith(PROTOCOL_PREFIX):
            token, _, rest = path[len(PROTOCOL_PREFIX) :].partition("/")
            if hmac.compare_digest(token.encode(), self.token.encode()):
                reply = self._protocol(method, rest, body)
                if reply is not None:
                    return reply
            return _not_found()
        if path.startswith(HTTP_PREFIX):
            host, _, rest = path[len(HTTP_PREFIX) :].partition("/")
            item = self.hosts.get(unquote(host).lower())
            if item is not None:
                return self._http(item, method, "/" + rest, parts.query, headers, body)
            self.log(f"{method} {path}: no HTTP mock for {unquote(host)!r}")
        return _not_found()

    def _protocol(self, method: str, rest: str, body: bytes) -> Reply | None:
        if method == "GET" and rest == "health":
            return _json(200, {"ok": True})
        if method == "GET" and rest == "keys":
            return _json(200, {"keys": sorted(self.functions)})
        if method == "GET" and rest == "calls":
            with self._lock:
                text = "".join(json.dumps(c) + "\n" for c in self.calls)
            return 200, {"Content-Type": "application/x-ndjson"}, text.encode()
        if method == "GET" and rest == "log":
            with self._lock:
                text = "".join(line + "\n" for line in self.log_lines)
            return 200, {"Content-Type": "text/plain; charset=utf-8"}, text.encode()
        if method == "POST" and rest.startswith("fn/") and "/" not in rest[3:]:
            return self._function(unquote(rest[3:]), body)
        return None

    # Function mocks

    def _function(self, key: str, body: bytes) -> Reply:
        try:
            request = json.loads(body or b"null")
        except ValueError:
            request = None
        if not isinstance(request, dict) or "args" not in request:
            return _error(400, "BadRequest", 'expected a JSON object with "args"')
        logged = {"args": request.get("args"), "lang": request.get("lang")}
        trace = (request.get("trace_id"), request.get("span_id"))
        item = self.functions.get(key)
        start = time.perf_counter()
        with self._lock:
            if item is None:
                message = f"no function mock {key!r}"
                error = {"type": "UnknownMockKey", "message": message}
                self._record("function", key, logged, "unmatched", {"error": error}, start, trace)
                return _json(404, {"error": error})
            index = self._next_index(key)
            call = Call(key=key, args=request.get("args"), index=index)
            try:
                result = self._call(item, call)
                if isinstance(result, Response):
                    raise TypeError("Response is for HTTP mocks; return the value itself")
                if result is PASSTHROUGH:
                    outcome, reply, logged_reply = "passthrough", {"action": "passthrough"}, {}
                else:
                    payload = json.dumps(result, allow_nan=False)
                    outcome = "respond"
                    reply = {"action": "respond", "value": json.loads(payload)}
                    logged_reply = {"value": reply["value"]}
                status = 200
            except Error as e:
                outcome, status = "raise", 200
                logged_reply = {"error": {"type": e.type, "message": e.message}}
                reply = {"action": "raise", **logged_reply}
            except Exception as e:
                outcome, status = "handler_error", 500
                logged_reply = {"error": self._handler_error(key, e)}
                reply = {"action": "handler_error", **logged_reply}
            self._record("function", key, logged, outcome, logged_reply, start, trace, index)
        return _json(status, reply)

    # HTTP mocks

    def _http(
        self,
        item: Mock,
        method: str,
        path: str,
        query: str,
        headers: dict[str, str],
        body: bytes,
    ) -> Reply:
        start = time.perf_counter()
        parsed_query = parse_qs(query, keep_blank_values=True)
        logged = {
            "method": method,
            "path": path,
            "query": parsed_query,
            "headers": headers,
            **_body_fields(body),
        }
        trace = _traceparent(headers.get("traceparent"))
        with self._lock:
            index = self._next_index(item.key)
            request = Request(
                method=method,
                path=path,
                query=parsed_query,
                headers=dict(headers),
                body=body,
                index=index,
            )
            try:
                result = self._call(item, request)
                if result is None or result is PASSTHROUGH:
                    raise Unmatched()
                response = result if isinstance(result, Response) else Response(json=result)
                status, out_headers, payload = response.encode()
                outcome = "respond"
            except Unmatched:
                outcome = "unmatched"
                status, out_headers, payload = _error(
                    501, "Unmatched", f"no mock response for {method} {path}"
                )
            except Error as e:
                outcome = "raise"
                status, out_headers, payload = _error(500, e.type, e.message)
            except Exception as e:
                outcome = "handler_error"
                error = self._handler_error(item.key, e)
                status, out_headers, payload = _json(500, {"error": error})
            logged_reply = {"status": status, "headers": out_headers, **_body_fields(payload)}
            self._record("http", item.key, logged, outcome, logged_reply, start, trace, index)
        return status, out_headers, payload

    # Internals (under the lock)

    def _call(self, item: Mock, arg: Any) -> Any:
        ctx = Context(
            state=self.state,
            calls=[dict(c) for c in self.calls if c["key"] == item.key],
            fixtures=self.fixtures,
            case=self.case,
            env=self.env,
        )
        result = item.handler(arg, ctx)
        if inspect.isawaitable(result):
            result = self._await(result)
        return result

    def _await(self, awaitable: Awaitable[Any]) -> Any:
        """Async handlers share one event loop on its own thread."""
        if self._loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="relay-loop", daemon=True).start()
            self._loop = loop

        async def wrap() -> Any:
            return await awaitable

        return asyncio.run_coroutine_threadsafe(wrap(), self._loop).result()

    def _next_index(self, key: str) -> int:
        index = self._indexes.get(key, 0)
        self._indexes[key] = index + 1
        return index

    def _handler_error(self, key: str, error: Exception) -> dict[str, str]:
        self.log(f"handler for {key!r} raised:\n{traceback.format_exc().rstrip()}")
        return {"type": type(error).__name__, "message": str(error)}

    def _record(
        self,
        type: str,
        key: str,
        request: dict[str, Any],
        outcome: str,
        response: dict[str, Any],
        start: float,
        trace: tuple[Any, Any],
        index: int | None = None,
    ) -> None:
        entry = {
            "seq": len(self.calls),
            "ts": _timestamp(),
            "type": type,
            "key": key,
            "index": index,
            "request": request,
            "outcome": outcome,
            "response": response,
            "duration_ms": round((time.perf_counter() - start) * 1000, 3),
            "trace_id": trace[0] if isinstance(trace[0], str) else None,
            "span_id": trace[1] if isinstance(trace[1], str) else None,
        }
        self.calls.append(entry)
        status = response.get("status", "")
        self.log(f"#{entry['seq']} {type} {key} -> {outcome} {status}".rstrip())


# HTTP


class _Handler(BaseHTTPRequestHandler):
    server_version: str = "nginx"
    sys_version: str = ""
    protocol_version: str = "HTTP/1.1"
    mock_server: ClassVar[MockServer]

    def dispatch(self) -> None:
        try:
            body = self._read_body()
        except ValueError:
            self.send_error(400)
            return
        headers = {k.lower(): v for k, v in self.headers.items()}
        status, out_headers, payload = self.mock_server.route(
            self.command, self.path, headers, body
        )
        self.send_response(status)
        for key, value in out_headers.items():
            if key.lower() not in ("content-length", "transfer-encoding", "connection"):
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            _ = self.wfile.write(payload)

    def _read_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            chunks: list[bytes] = []
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return b"".join(chunks)
                chunks.append(self.rfile.read(size))
                _ = self.rfile.readline()
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length > 0 else b""

    @override
    def log_message(self, format: str, *args: Any) -> None:
        pass  # each call is logged by MockServer


for _method in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
    setattr(_Handler, f"do_{_method}", _Handler.dispatch)


def http_server(server: MockServer) -> ThreadingHTTPServer:
    """`server` behind HTTP on 127.0.0.1, on a free port (not serving yet)."""
    handler = type("Handler", (_Handler,), {"mock_server": server})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    return httpd


def serve(config_path: str) -> int:
    """Load, bind 127.0.0.1 on a free port, write the port to `ready_file`,
    serve until killed. A startup error goes to stderr with exit code 1.
    `relay.py serve` already named the process."""
    try:
        config = json.loads(Path(config_path).read_text())
        server = MockServer(config)
        ready_file = Path(str(config["ready_file"]))
    except Exception:
        traceback.print_exc()
        return 1
    httpd = http_server(server)
    port = httpd.server_address[1]
    server.log(f"listening on 127.0.0.1:{port}")
    partial = ready_file.with_name(ready_file.name + ".part")
    _ = partial.write_text(str(port))
    os.replace(partial, ready_file)
    # The runner deletes the files this process was started with; nothing
    # from here on may write to them.
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        _ = os.dup2(devnull, fd)
    httpd.serve_forever()
    return 0


# Helpers


def _load_runner(path: Path) -> None:
    """`eval.py` by path as `__runner__`: dataset folder names aren't always
    valid module names."""
    spec = importlib.util.spec_from_file_location(RUNNER_MODULE, path)
    if spec is None or spec.loader is None:
        raise StartupError(f"can't load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[RUNNER_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        raise StartupError(
            f"importing {path.name} failed: {e}. Handlers run inside the container, so what "
            + "eval.py and its shared modules import at the top level must be installed "
            + "there too: add it to the runner's mock_dependencies."
        ) from e


def _load_fixtures(directory: Any) -> dict[str, bytes]:
    if not directory:
        return {}
    root = Path(str(directory))
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _timestamp() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _json(status: int, value: Any) -> Reply:
    return status, {"Content-Type": "application/json"}, json.dumps(value).encode()


def _error(status: int, type: str, message: str) -> Reply:
    return _json(status, {"error": {"type": type, "message": message}})


def _not_found() -> Reply:
    return 404, {"Content-Type": "text/plain; charset=utf-8"}, b"Not Found"


def _body_fields(body: bytes) -> dict[str, str]:
    if not body:
        return {"body_text": ""}
    try:
        return {"body_text": body.decode()}
    except UnicodeDecodeError:
        return {"body_b64": base64.b64encode(body).decode()}


def _traceparent(value: str | None) -> tuple[str | None, str | None]:
    match = _TRACEPARENT.match(value.strip()) if value else None
    return (match.group(1), match.group(2)) if match else (None, None)

"""`lmnr.evals.relay`: the runner's in-container helpers (`relay.py http` /
`wait` / `rows`), run in-process and as the process the runner starts."""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lmnr.evals import relay


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path.startswith("/status/"):
            code = int(self.path.rsplit("/", 1)[1])
            body = json.dumps({"code": code}).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            _ = self.wfile.write(body)
        elif self.path == "/chunked":
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for part in (b"hello ", b"chunked ", b"world"):
                _ = self.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
            _ = self.wfile.write(b"0\r\n\r\n")
        elif self.path.startswith("/slow"):
            time.sleep(3)
            self.send_response(204)
            self.end_headers()
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/status/200")
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        reply = json.dumps({"got": body.decode(), "type": self.headers["Content-Type"]}).encode()
        self.send_response(201)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        _ = self.wfile.write(reply)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def reserved_port() -> socket.socket:
    """A bound, not listening socket: connecting to its port is refused."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    return sock


# http


@pytest.mark.parametrize("code", [200, 404, 500])
def test_http_passes_any_status_through(server: str, code: int):
    response = relay.http({"url": f"{server}/status/{code}"})
    assert response["status"] == code
    assert json.loads(base64.b64decode(response["body_b64"])) == {"code": code}


def test_http_sends_the_body_and_headers(server: str):
    response = relay.http(
        {
            "method": "POST",
            "url": f"{server}/seed",
            "headers": {"Content-Type": "application/json"},
            "body_b64": base64.b64encode(b'{"a": 1}').decode(),
        }
    )
    assert response["status"] == 201
    assert json.loads(base64.b64decode(response["body_b64"])) == {
        "got": '{"a": 1}',
        "type": "application/json",
    }


def test_http_reads_chunked_responses_and_does_not_follow_redirects(server: str):
    chunked = relay.http({"url": f"{server}/chunked"})
    assert base64.b64decode(chunked["body_b64"]) == b"hello chunked world"
    assert relay.http({"url": f"{server}/redirect"})["status"] == 302


def test_http_transport_errors_and_timeouts_raise(server: str):
    with reserved_port() as sock, pytest.raises(OSError):
        _ = relay.http({"url": f"http://127.0.0.1:{sock.getsockname()[1]}/"})
    with pytest.raises(OSError):
        _ = relay.http({"url": f"{server}/slow", "timeout": 0.5})


def run_relay(
    command: str, payload: object, timeout: float = 30
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, relay.__file__, command],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_http_as_a_process_keeps_the_request_out_of_argv(server: str):
    """What `ps` shows during a long request (a Rystic `settle()`): the relay's
    name and command, never the URL or the token."""
    url = f"{server}/slow?token=secret-token"
    process = subprocess.Popen(
        [sys.executable, relay.__file__, "http"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    _ = process.stdin.write(json.dumps({"url": url}))
    process.stdin.close()
    time.sleep(1)
    cmdline = Path(f"/proc/{process.pid}/cmdline").read_bytes().decode()
    comm = Path(f"/proc/{process.pid}/comm").read_text().strip()
    assert process.stdout is not None
    out = process.stdout.read()
    _ = process.wait(timeout=30)
    assert "secret-token" not in cmdline and "/slow" not in cmdline
    assert comm == "relay"
    assert process.returncode == 0
    assert json.loads(out)["status"] == 204

    with reserved_port() as sock:
        failed = run_relay("http", {"url": f"http://127.0.0.1:{sock.getsockname()[1]}/"})
    assert failed.returncode == 1
    assert "relay http" in failed.stderr


# wait


def test_wait_port_http_file_and_cmd(server: str, tmp_path: Path):
    port = int(server.rsplit(":", 1)[1])
    ready = [
        {"kind": "port", "host": "127.0.0.1", "port": port},
        {"kind": "http", "url": f"{server}/status/200"},
        {"kind": "cmd", "command": "test -d /"},
    ]
    for spec in ready:
        assert relay.wait({"ready": spec, "timeout": 5}) == relay.READY, spec

    target = tmp_path / "ready"
    threading.Timer(0.5, lambda: target.write_text("1234")).start()
    assert relay.wait({"ready": {"kind": "file", "path": str(target)}, "timeout": 5}) == 0


def test_wait_times_out(server: str):
    with reserved_port() as sock:
        port = sock.getsockname()[1]
        assert (
            relay.wait({"ready": {"kind": "port", "host": "127.0.0.1", "port": port}, "timeout": 1})
            == 1
        )
    not_ready = [
        {"kind": "http", "url": f"{server}/status/503"},
        {"kind": "cmd", "command": "exit 3"},
        {"kind": "file", "path": "/nonexistent/ready"},
    ]
    for spec in not_ready:
        started = time.monotonic()
        assert relay.wait({"ready": spec, "timeout": 1}) == relay.TIMED_OUT, spec
        assert time.monotonic() - started < 5


def test_wait_stops_when_the_process_exits_or_is_a_zombie():
    never = {"kind": "cmd", "command": "false"}
    # Exited and reaped.
    done = subprocess.Popen(["true"])
    _ = done.wait()
    started = time.monotonic()
    assert relay.wait({"ready": never, "timeout": 30, "pid": done.pid}) == relay.EXITED
    # Exited, never reaped (Harbor's PID 1 is `sleep infinity`): a zombie.
    zombie = subprocess.Popen(["true"])
    time.sleep(0.5)
    assert Path(f"/proc/{zombie.pid}/status").read_text().count("State:\tZ") == 1
    assert relay.wait({"ready": never, "timeout": 30, "pid": zombie.pid}) == relay.EXITED
    assert time.monotonic() - started < 10
    _ = zombie.wait()
    # Alive: waits for the readiness check.
    alive = subprocess.Popen(["sleep", "30"])
    try:
        assert relay.wait({"ready": never, "timeout": 1, "pid": alive.pid}) == relay.TIMED_OUT
    finally:
        alive.kill()
        _ = alive.wait()


def test_wait_as_a_process_exits_with_its_status():
    assert run_relay("wait", {"ready": {"kind": "cmd", "command": "true"}}).returncode == 0
    timed_out = run_relay("wait", {"ready": {"kind": "cmd", "command": "false"}, "timeout": 0.5})
    assert timed_out.returncode == 1
    exited = run_relay(
        "wait", {"ready": {"kind": "cmd", "command": "false"}, "timeout": 30, "pid": 2**22 + 7}
    )
    assert exited.returncode == 2


# rows


def test_rows_writes_json_arrays(tmp_path: Path):
    relay.rows([])
    out, empty = tmp_path / "events.json", tmp_path / "empty.json"
    relay.rows(
        [
            {
                "command": """printf '{"id":1,"n":"18446744073709551615"}\\n{"id":2}\\n'""",
                "out": str(out),
            },
            {"command": "true", "out": str(empty)},
        ]
    )
    assert json.loads(out.read_text()) == [{"id": 1, "n": "18446744073709551615"}, {"id": 2}]
    assert json.loads(empty.read_text()) == []


def test_rows_fails_on_a_failing_producer_or_garbage(tmp_path: Path):
    out = tmp_path / "events.json"
    with pytest.raises(RuntimeError, match="exit code 60"):
        relay.rows(
            [{"command": "echo '{\"id\":1}'; echo 'no table' >&2; exit 60", "out": str(out)}]
        )
    assert not out.exists()
    with pytest.raises(ValueError):
        relay.rows([{"command": "echo 'Code: 60. not json'", "out": str(out)}])
    failed = run_relay("rows", [{"command": "exit 60", "out": str(out)}])
    assert failed.returncode == 1
    assert "exit code 60" in failed.stderr


def test_relay_has_no_lmnr_imports():
    """The launcher runs it by path, without importing the `lmnr` package
    (which takes seconds); `serve` imports the server only when called."""
    source = Path(relay.__file__).read_text()
    top_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert all("lmnr" not in line for line in top_level), top_level
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import runpy, sys; runpy.run_path({relay.__file__!r}); "
            + "print(sorted(m for m in sys.modules if m.startswith('lmnr')))",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert probe.stdout.strip() == "[]"

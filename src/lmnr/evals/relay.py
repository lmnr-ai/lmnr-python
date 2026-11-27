"""The runner's helpers inside the trial's container: `relay.py <command>`.

The Laminar layer installs Python and `lmnr` into `/opt/lmnr` in every image
of an `eval.py` runner and writes `/opt/lmnr/relay.py`, a launcher that runs
THIS file by path. Importing the `lmnr` package takes seconds (OpenTelemetry,
instrumentations), so this module uses only the standard library and never
imports `lmnr`; only `serve` does, because it loads the user's handlers.

Commands read one JSON document on stdin, so URLs, tokens and payloads never
show in `ps` (the runner feeds it from a temp file it deletes first):

- `http`: `{method, url, headers, body_b64, timeout}` -> prints
  `{status, headers, body_b64}`. Exits 0 on any HTTP status, 1 on a transport
  error (message on stderr), so callers decide what a 4xx means.
- `wait`: `{ready, timeout, pid}` -> exits 0 once ready, 1 on timeout, 2 when
  `pid` exited first (a zombie counts as exited). `ready` is
  `{"kind": "port", "host", "port"}`, `{"kind": "http", "url"}`,
  `{"kind": "file", "path"}` or `{"kind": "cmd", "command"}` (`sh -c`).
- `rows`: `[{command, out}]` -> runs each `command` (`sh -c`), which prints
  JSON objects one per line (ClickHouse's JSONEachRow), and writes them to
  `out` as one JSON array. Fails when a command exits non-zero, so a failed
  query never saves `[]`.
- `serve <config>`: the mock server (`lmnr.evals.mocks.server`).

No command line or process name may contain `mock`, `eval`, `test` or
`fixture`: the process is named `relay` (`PR_SET_NAME`).
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

PROCESS_NAME = b"relay"
POLL_INTERVAL_SEC = 0.25
# One readiness attempt (a TCP connect, an HTTP request) may take this long.
ATTEMPT_TIMEOUT_SEC = 5.0
DEFAULT_HTTP_TIMEOUT_SEC = 60.0

READY = 0
TIMED_OUT = 1
EXITED = 2

_PR_SET_NAME = 15


def set_process_name(name: bytes = PROCESS_NAME) -> None:
    """Name the calling thread (and so the threads it starts) `relay`, as `ps`
    and `/proc/<pid>/comm` show it. Best effort, Linux only."""
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        _ = libc.prctl(_PR_SET_NAME, ctypes.c_char_p(name[:15]), 0, 0, 0)
    except (OSError, AttributeError):
        pass


# http


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def http(request: dict[str, Any]) -> dict[str, Any]:
    """One HTTP request; any status is a result. Raises `OSError` on transport
    errors (connection refused, timeout, TLS).

    Proxy variables are ignored: every request goes to a process in this
    container (the mock server, a mirror, a readiness URL), and urllib on
    Linux would send even 127.0.0.1 to `http_proxy` unless `no_proxy` lists
    it."""
    body_b64 = request.get("body_b64")
    data = base64.b64decode(body_b64) if body_b64 else None
    req = urllib.request.Request(
        str(request["url"]),
        data=data,
        method=str(request.get("method") or "GET").upper(),
        headers={str(k): str(v) for k, v in (request.get("headers") or {}).items()},
    )
    timeout = float(request.get("timeout") or DEFAULT_HTTP_TIMEOUT_SEC)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
    try:
        with opener.open(req, timeout=timeout) as response:
            status, headers, body = response.status, response.headers, response.read()
    except urllib.error.HTTPError as e:
        status, headers, body = e.code, e.headers, e.read()
    return {
        "status": status,
        "headers": _headers(headers),
        "body_b64": base64.b64encode(body).decode(),
    }


def _headers(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items() if headers is not None else []:
        name = str(key).lower()
        out[name] = f"{out[name]}, {value}" if name in out else str(value)
    return out


# wait


def wait(request: dict[str, Any]) -> int:
    ready = request["ready"]
    timeout = float(request.get("timeout") or 60)
    pid = request.get("pid")
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if _ready(ready, max(min(remaining, ATTEMPT_TIMEOUT_SEC), 0.1)):
            return READY
        if pid is not None and not alive(int(pid)):
            return EXITED
        if time.monotonic() >= deadline:
            return TIMED_OUT
        time.sleep(POLL_INTERVAL_SEC)


def _ready(ready: dict[str, Any], timeout: float) -> bool:
    kind = ready.get("kind")
    try:
        if kind == "port":
            with socket.create_connection((str(ready["host"]), int(ready["port"])), timeout):
                return True
        if kind == "http":
            status = http({"url": ready["url"], "timeout": timeout})["status"]
            return 200 <= int(status) < 400
        if kind == "file":
            return os.path.isfile(str(ready["path"])) and os.path.getsize(str(ready["path"])) > 0
        if kind == "cmd":
            result = subprocess.run(
                ["sh", "-c", str(ready["command"])],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return result.returncode == 0
    except (OSError, ValueError):
        return False
    raise ValueError(f"unknown readiness kind {kind!r}")


def alive(pid: int) -> bool:
    """`pid` exists and isn't a zombie. Harbor's containers run `sleep
    infinity` as PID 1, which never reaps, and signal 0 succeeds on a zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # another user's process (a root service, a non-root relay)
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    return line.split()[1] != "Z"
    except OSError:
        pass
    return True


# rows


def rows(jobs: list[dict[str, Any]]) -> None:
    """Raises `RuntimeError` when a command fails or prints something that
    isn't one JSON value per line."""
    for job in jobs:
        command, out = str(job["command"]), str(job["out"])
        result = subprocess.run(
            ["sh", "-c", command], stdin=subprocess.DEVNULL, capture_output=True, check=False
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[-2000:].rstrip()
            raise RuntimeError(f"exit code {result.returncode} from {command}\n{stderr}")
        lines = [line for line in result.stdout.decode().splitlines() if line.strip()]
        for line in lines:
            _ = json.loads(line)  # ValueError on garbage
        # Kept as printed: ClickHouse's settings decide how 64-bit ints look.
        with open(out, "w") as f:
            _ = f.write("[" + ",\n".join(lines) + "]\n")


# Entry point


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    set_process_name()
    command = args[0] if args else ""
    if command == "serve" and len(args) == 2:
        # Heavy (imports `lmnr` and the user's handlers), so only here.
        from lmnr.evals.mocks.server import serve

        return serve(args[1])
    try:
        payload: Any = json.load(sys.stdin) if command in ("http", "wait", "rows") else None
        if command == "http":
            print(json.dumps(http(payload)))
            return 0
        if command == "wait":
            return wait(payload)
        if command == "rows":
            rows(payload)
            return 0
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        print(f"relay {command}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(
        "usage: relay.py http|wait|rows < request.json, or relay.py serve <config>", file=sys.stderr
    )
    return 64


if __name__ == "__main__":
    raise SystemExit(main())

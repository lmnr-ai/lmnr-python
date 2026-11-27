"""Readiness checks for `ctx.container.service(...)` and `ctx.container.wait_for(...)`.

The runner polls them inside the container with `relay.py wait` (the Laminar
layer's Python), so `port` and `http` need nothing from the image; `cmd` runs
the command with `sh -c`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Readiness:
    kind: str
    description: str
    # What `relay.py wait` gets as `ready` (with `kind`).
    params: dict[str, Any] = field(default_factory=dict)

    def spec(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.params}

    def __str__(self) -> str:
        return self.description


def port(number: int, host: str = "127.0.0.1") -> Readiness:
    """Ready once something accepts TCP connections on `host:number`."""
    return Readiness("port", f"port {host}:{number}", {"host": host, "port": int(number)})


def http(url: str) -> Readiness:
    """Ready once `url` answers with a 2xx or 3xx status."""
    return Readiness("http", f"HTTP {url}", {"url": url})


def file(path: str) -> Readiness:
    """Ready once `path` exists and isn't empty."""
    return Readiness("file", f"file {path}", {"path": path})


def cmd(command: str) -> Readiness:
    """Ready once `command` exits 0 (`sh -c`)."""
    return Readiness("cmd", f"`{command}`", {"command": command})

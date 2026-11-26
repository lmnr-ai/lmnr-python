"""Readiness checks for `ctx.service(...)` and `ctx.wait_for(...)`.

Each check is a shell command that exits 0 once the thing is ready. It runs
inside the container, so it only relies on bash and whatever the image has.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class Readiness:
    command: str
    description: str

    def __str__(self) -> str:
        return self.description


def port(number: int, host: str = "127.0.0.1") -> Readiness:
    """Ready once something accepts TCP connections on `host:number`."""
    target = f"/dev/tcp/{host}/{int(number)}"
    return Readiness(
        command=f"(exec 3<>{shlex.quote(target)}) 2>/dev/null",
        description=f"port {host}:{number}",
    )


def http(url: str) -> Readiness:
    """Ready once `url` answers with a 2xx or 3xx status (curl, wget or python3)."""
    quoted = shlex.quote(url)
    python = shlex.quote(f"import urllib.request; urllib.request.urlopen({url!r}, timeout=5)")
    return Readiness(
        command=(
            f"if command -v curl >/dev/null 2>&1; then curl -fsS -o /dev/null {quoted}; "
            f"elif command -v wget >/dev/null 2>&1; then wget -q -O /dev/null {quoted}; "
            f"else python3 -c {python}; fi"
        ),
        description=f"HTTP {url}",
    )


def cmd(command: str) -> Readiness:
    """Ready once `command` exits 0."""
    return Readiness(command=command, description=f"`{command}`")

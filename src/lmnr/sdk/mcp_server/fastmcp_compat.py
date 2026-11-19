"""Turning FastMCP's own spans off for a wrapped server's requests.

Standalone FastMCP (`fastmcp`) opens native SERVER spans of its own on the
global tracer provider, which is Laminar's once Laminar is initialized. A
wrapped server must produce exactly one span per request, so the wrapper runs
every request of the server with FastMCP's spans suppressed:

- FastMCP 4 (on mcp 2.x) ships `fastmcp.telemetry.suppress_fastmcp_telemetry()`,
  a scoped switch to its `propagation_only` mode.
- FastMCP 3 (on mcp 1.x) has no switch. `install_fastmcp3_suppression` swaps
  `fastmcp.server.telemetry.get_tracer` (the tracer its server spans use) for
  one that returns a pass-through tracer while `suppress_fastmcp3()` is active,
  and the original tracer otherwise.

`fastmcp` is optional: it is imported only for a server that is a FastMCP one.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Any, cast

from opentelemetry.trace import INVALID_SPAN, NoOpTracer, Span
from typing_extensions import override

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

_SUPPRESSED: ContextVar[bool] = ContextVar("lmnr_mcp_suppress_fastmcp", default=False)
_PATCHED_MARKER = "_lmnr_suppressible"


class _PassThroughTracer(NoOpTracer):
    """Creates no span and, unlike `NoOpTracer`, leaves the current context alone."""

    @contextmanager
    @override
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Generator[Span]:  # pyright: ignore[reportIncompatibleMethodOverride]
        yield INVALID_SPAN


_PASS_THROUGH = _PassThroughTracer()


def is_fastmcp_server(server: object) -> bool:
    return type(server).__module__.split(".")[0] == "fastmcp"


@contextmanager
def suppress_fastmcp3() -> Generator[None]:
    token = _SUPPRESSED.set(True)
    try:
        yield
    finally:
        _SUPPRESSED.reset(token)


def install_fastmcp3_suppression() -> bool:
    """Make FastMCP 3's server spans honor `suppress_fastmcp3()`. Idempotent."""
    try:
        module = importlib.import_module("fastmcp.server.telemetry")
    except Exception:
        logger.debug("Failed to import fastmcp.server.telemetry", exc_info=True)
        return False
    original: Any = getattr(module, "get_tracer", None)
    if not callable(original):
        return False
    if getattr(original, _PATCHED_MARKER, False):
        return True

    def get_tracer(*args: Any, **kwargs: Any) -> Any:
        if _SUPPRESSED.get():
            return _PASS_THROUGH
        return original(*args, **kwargs)

    setattr(get_tracer, _PATCHED_MARKER, True)
    module.get_tracer = get_tracer  # pyright: ignore[reportAttributeAccessIssue]
    return True


def native_span_suppression(server: object) -> Callable[[], AbstractContextManager[Any]] | None:
    """The block that turns `server`'s FastMCP spans off, or None for other servers."""
    if not is_fastmcp_server(server):
        return None
    try:
        telemetry = importlib.import_module("fastmcp.telemetry")
    except Exception:
        logger.debug("Failed to import fastmcp.telemetry", exc_info=True)
        return None
    suppress: Any = getattr(telemetry, "suppress_fastmcp_telemetry", None)
    if callable(suppress):
        # FastMCP 4+
        return cast(Callable[[], AbstractContextManager[Any]], suppress)
    if install_fastmcp3_suppression():
        return suppress_fastmcp3
    logger.warning(
        "Laminar could not turn off this FastMCP version's own spans; "
        + "each MCP request may show up twice."
    )
    return None

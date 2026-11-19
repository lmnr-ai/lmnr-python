"""Markers shared with `lmnr.sdk.mcp_server` (`wrap_mcp_server`).

A server wrapped with `wrap_mcp_server` gets its spans from the wrapper, so
the instrumentor's SERVER wrappers stand aside for it: one span per request.
Leaf module (stdlib only), so both sides import it without a cycle.
"""

from __future__ import annotations

from contextvars import ContextVar

# Set on the low-level `Server` (and the high-level object wrapping it) to the
# wrapper's config.
OBSERVED_SERVER_ATTR = "_lmnr_mcp_observability"

# The `ServerSession` of the request a wrapped server is handling (mcp 2.x
# builds one per request), set by the wrapper's middleware for the rest of the
# chain, so an `OpenTelemetryMiddleware` added after wrapping stands aside. A
# session, not a flag: a tool of a wrapped server may call another, unwrapped,
# in-process server in the same context.
OBSERVED_REQUEST: ContextVar[object | None] = ContextVar("lmnr_mcp_observed_request", default=None)


def is_observed_server(server: object) -> bool:
    return getattr(server, OBSERVED_SERVER_ATTR, None) is not None


def is_observed_request(ctx: object) -> bool:
    session = getattr(ctx, "session", None)
    return session is not None and OBSERVED_REQUEST.get() is session

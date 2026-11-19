"""The id generator installed on Laminar's `TracerProvider`.

It is OTel's `RandomIdGenerator`, except that a ROOT span started inside
`forced_trace_id(trace_id)` gets that trace id. OTel only asks the id
generator for a trace id when the span has no valid parent, so the forced id
never affects child spans. The MCP server wrapper uses it to put every call of
one user turn in one trace while keeping each call's span a root (a fake remote
parent would leave a dangling parent span id that the trace view can't render).

Leaf module: stdlib + the OTel SDK only.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import INVALID_TRACE_ID
from typing_extensions import override

_FORCED_TRACE_ID: ContextVar[int | None] = ContextVar("lmnr_forced_trace_id", default=None)


class LaminarIdGenerator(RandomIdGenerator):
    """Random ids, or the trace id forced with `forced_trace_id` for a root span."""

    @override
    def generate_trace_id(self) -> int:
        forced = _FORCED_TRACE_ID.get()
        if forced is not None and forced != INVALID_TRACE_ID:
            return forced
        return super().generate_trace_id()


@contextmanager
def forced_trace_id(trace_id: int | None) -> Generator[None]:
    """Root spans started in this block get `trace_id` (None: a random one).

    Only honored by a provider whose id generator is a `LaminarIdGenerator`
    (see `can_force_trace_id`); any other provider ignores it and the span gets
    a random trace id.
    """
    token = _FORCED_TRACE_ID.set(trace_id)
    try:
        yield
    finally:
        _FORCED_TRACE_ID.reset(token)


def can_force_trace_id(provider: object) -> bool:
    """Whether root spans from `provider` honor `forced_trace_id`."""
    return isinstance(getattr(provider, "id_generator", None), LaminarIdGenerator)

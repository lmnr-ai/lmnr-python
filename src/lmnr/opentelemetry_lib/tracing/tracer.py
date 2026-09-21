from collections.abc import Generator
from contextlib import contextmanager
from typing import cast

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span as SDKSpan

from lmnr.opentelemetry_lib.tracing import TRACER_NAME, get_tracer_wrapper
from lmnr.opentelemetry_lib.tracing.context import (
    get_current_context,
    pop_span_context,
    push_span,
)
from lmnr.opentelemetry_lib.tracing.span import LaminarSpan


def get_laminar_tracer_provider() -> trace.TracerProvider:
    wrapper = get_tracer_wrapper()
    if wrapper is None:
        return trace.get_tracer_provider()
    return wrapper.tracer_provider


def _resolve_tracer() -> trace.Tracer:
    """Laminar's tracer, or the global provider's when tracing is not
    initialized. Never initializes tracing as a side effect."""
    wrapper = get_tracer_wrapper()
    if wrapper is None:
        return trace.get_tracer_provider().get_tracer(TRACER_NAME)
    return wrapper.get_tracer()


@contextmanager
def get_tracer(flush_on_exit: bool = False) -> Generator[trace.Tracer, None, None]:
    try:
        yield LaminarTracer(_resolve_tracer())
    finally:
        if flush_on_exit:
            wrapper = get_tracer_wrapper()
            if wrapper is not None:
                wrapper.flush()


@contextmanager
def get_tracer_with_context(
    flush_on_exit: bool = False,
) -> Generator[tuple[trace.Tracer, Context], None, None]:
    """Get tracer with isolated context. Returns (tracer, context) tuple."""
    try:
        yield LaminarTracer(_resolve_tracer()), get_current_context()
    finally:
        if flush_on_exit:
            wrapper = get_tracer_wrapper()
            if wrapper is not None:
                wrapper.flush()


class LaminarTracer(trace.Tracer):
    _instance: trace.Tracer

    def __init__(self, instance: trace.Tracer):
        self._instance = instance

    def start_span(self, *args, **kwargs) -> trace.Span:
        span = self._instance.start_span(*args, **kwargs)
        return LaminarSpan(cast(SDKSpan, span))

    @contextmanager
    def start_as_current_span(self, *args, **kwargs) -> Generator[trace.Span]:  # pyright: ignore[reportIncompatibleMethodOverride]
        with self._instance.start_as_current_span(*args, **kwargs) as span:
            push_span(span)
            try:
                yield LaminarSpan(cast(SDKSpan, span))
            finally:
                pop_span_context()

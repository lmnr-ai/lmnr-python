from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, cast

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span as SDKSpan
from typing_extensions import override

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
                _flush_success = wrapper.flush()


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
                _flush_success = wrapper.flush()


class LaminarTracer(trace.Tracer):
    _instance: trace.Tracer

    def __init__(self, instance: trace.Tracer):
        self._instance = instance

    @override
    def start_span(self, *args: Any, **kwargs: Any) -> trace.Span:  # pyright: ignore[reportExplicitAny, reportAny]
        span = self._instance.start_span(*args, **kwargs)  # pyright: ignore[reportAny]
        return LaminarSpan(cast(SDKSpan, span))

    @contextmanager
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Generator[trace.Span]:  # pyright: ignore[reportIncompatibleMethodOverride, reportExplicitAny, reportAny]
        with self._instance.start_as_current_span(*args, **kwargs) as span:  # pyright: ignore[reportAny]
            _new_context = push_span(span)
            try:
                yield LaminarSpan(cast(SDKSpan, span))
            finally:
                pop_span_context()

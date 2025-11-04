from contextlib import contextmanager
from typing import Generator, Iterator, Tuple

from opentelemetry import trace
from opentelemetry.context import Context
from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.opentelemetry_lib.tracing.span import LaminarSpan


def get_laminar_tracer_provider() -> trace.TracerProvider:
    return TracerWrapper.instance.__tracer_provider or trace.get_tracer_provider()


@contextmanager
def get_tracer(flush_on_exit: bool = False):
    wrapper = TracerWrapper()
    try:
        yield LaminarTracer(wrapper.get_tracer())
    finally:
        if flush_on_exit:
            wrapper.flush()


@contextmanager
def get_tracer_with_context(
    flush_on_exit: bool = False,
) -> Generator[Tuple[trace.Tracer, Context], None, None]:
    """Get tracer with isolated context. Returns (tracer, context) tuple."""
    wrapper = TracerWrapper()
    try:
        tracer = LaminarTracer(wrapper.get_tracer())
        context = wrapper.get_isolated_context()
        yield tracer, context
    finally:
        if flush_on_exit:
            wrapper.flush()


class LaminarTracer(trace.Tracer):
    _instance: trace.Tracer

    def __init__(self, instance: trace.Tracer):
        self._instance = instance

    def start_span(self, *args, **kwargs) -> trace.Span:
        span = LaminarSpan(self._instance.start_span(*args, **kwargs))
        return span

    @contextmanager
    def start_as_current_span(self, *args, **kwargs) -> Iterator[trace.Span]:
        with self._instance.start_as_current_span(*args, **kwargs) as span:
            yield LaminarSpan(span)

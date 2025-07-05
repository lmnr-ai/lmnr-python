from contextlib import contextmanager
from typing import Tuple

from opentelemetry import trace
from opentelemetry.context import Context
from lmnr.opentelemetry_lib.tracing import TracerWrapper


def get_laminar_tracer_provider() -> trace.TracerProvider:
    return TracerWrapper.instance.__tracer_provider or trace.get_tracer_provider()


@contextmanager
def get_tracer(flush_on_exit: bool = False):
    wrapper = TracerWrapper()
    try:
        yield wrapper.get_tracer()
    finally:
        if flush_on_exit:
            wrapper.flush()


@contextmanager
def get_tracer_with_context(
    flush_on_exit: bool = False,
) -> Tuple[trace.Tracer, Context]:
    """Get tracer with isolated context. Returns (tracer, context) tuple."""
    wrapper = TracerWrapper()
    try:
        tracer = wrapper.get_tracer()
        context = wrapper.get_isolated_context()
        yield tracer, context
    finally:
        if flush_on_exit:
            wrapper.flush()

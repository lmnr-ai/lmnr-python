from contextlib import contextmanager

from opentelemetry import trace
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

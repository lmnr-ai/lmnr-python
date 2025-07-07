from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import TraceFlags, SpanContext
from typing import Collection
from wrapt import wrap_function_wrapper
import logging


def _wrap_span_context(fn, instance, args, kwargs):
    """
    DataDog does something to the OpenTelemetry Contexts, so that when any code
    tries to access the current active span, it returns a non-recording span.

    There is nothing wrong about that per se, but they create their
    NonRecordingSpan from an invalid SpanContext, because they don't
    wrap the trace flags int/bitmap into a TraceFlags object.

    It is an easy to miss bug, because `TraceFlags.SAMPLED` looks like an
    instance of `TraceFlags`, but is actually just an integer 1, and  the
    proper way to create it is actually
    `TraceFlags(TraceFlags.SAMPLED)` or `TraceFlags(0x1)`.

    This is a problem because the trace flags are used to determine if a span
    is sampled or not. If the trace flags are not wrapped, then the check
    for sampling will fail, causing any span creation to fail, and sometimes
    breaking the entire application.

    Issue: https://github.com/DataDog/dd-trace-py/issues/12585
    PR: https://github.com/DataDog/dd-trace-py/pull/12596
    The PR only fixed the issue in one place, but it is still there in other places.
    https://github.com/DataDog/dd-trace-py/pull/12596#issuecomment-2718239507

    https://github.com/DataDog/dd-trace-py/blob/a8419a40fe9e73e0a84c4cab53094c384480a5a6/ddtrace/internal/opentelemetry/context.py#L83

    We patch the `get_span_context` method to return a valid SpanContext.
    """
    res = fn(*args, **kwargs)

    new_span_context = SpanContext(
        trace_id=res.trace_id,
        span_id=res.span_id,
        is_remote=res.is_remote,
        trace_state=res.trace_state,
        trace_flags=TraceFlags(res.trace_flags),
    )

    return new_span_context


class OpentelemetryInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return ("opentelemetry-api>=1.0.0",)

    def _instrument(self, **kwargs):
        try:
            wrap_function_wrapper(
                "opentelemetry.trace.span",
                "NonRecordingSpan.get_span_context",
                _wrap_span_context,
            )

        except Exception as e:
            logging.debug(f"Error wrapping SpanContext: {e}")

    def _uninstrument(self, **kwargs):
        unwrap("opentelemetry.trace.span", "NonRecordingSpan.get_span_context")

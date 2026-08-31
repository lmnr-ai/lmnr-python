from importlib.metadata import version
from typing import Any, Collection, Sequence

from opentelemetry.trace import TraceFlags, SpanContext

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)


def _wrap_span_context(
    to_wrap: WrappedFunctionSpec,
    fn,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
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


class OpentelemetryInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return ("opentelemetry-api>=1.0.0",)

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                otel_version = version("opentelemetry-api")
            except Exception:
                otel_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="opentelemetry",
                version=otel_version,
            )
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config = LaminarInstrumentorConfig(
            wrapped_functions=[
                WrappedFunctionSpec(
                    package_name="opentelemetry.trace.span",
                    object_name="NonRecordingSpan",
                    method_name="get_span_context",
                    is_async=False,
                    is_streaming=False,
                    # This wrapper repairs a SpanContext rather than tracing
                    # anything, so it creates no span and reads no span_name /
                    # span_type off the spec.
                    span_name=None,
                    span_type=None,
                    instrumentation_scope=self.instrumentation_scope(),
                    wrapper_function=_wrap_span_context,
                ),
            ]
        )

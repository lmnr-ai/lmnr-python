from typing import Any, Callable, Sequence

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    WrappedFunctionSpec,
)


def wrap_completion(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable,
    instance: Any,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
):
    print(" Wrapping completion")
    span = Laminar.start_span(
        name=to_wrap["span_name"],
        span_type=to_wrap["span_type"],
    )
    result = wrapped(*args, **kwargs)
    span.end()
    return result


def awrap_completion(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable,
    instance: Any,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
):
    return wrapped(*args, **kwargs)


def wrap_responses(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable,
    instance: Any,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
):
    return wrapped(*args, **kwargs)


def awrap_responses(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable,
    instance: Any,
    args: Sequence[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
):
    return wrapped(*args, **kwargs)

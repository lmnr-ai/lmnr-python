from opentelemetry.trace import Tracer
from typing import Any, Callable, Sequence

from .types import WrappedFunctionSpec


def with_tracer_wrapper(
    func: Callable[
        [
            Tracer,
            WrappedFunctionSpec,
            Callable,
            Any,
            Sequence[Any] | None,
            dict[str, Any] | None,
        ],
        Any,
    ],
):
    def wrapper(tracer: Tracer, to_wrap: WrappedFunctionSpec):
        def wrapped(
            wrapped: Callable,
            instance: Any,
            args: Sequence[Any] | None = None,
            kwargs: dict[str, Any] | None = None,
        ):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapped

    return wrapper

from typing import Any, Callable, Sequence

from opentelemetry.trace import Span

from .types import (
    LaminarInstrumentationScopeAttributes,
    SpecT,
    WrappedFunctionSpec,
    WraptWrapper,
    WrapperHandler,
)
from .utils import set_span_attribute


def add_spec_wrapper(
    wrapt_handler: WrapperHandler[SpecT],
    wrapped_spec: SpecT,
    **handler_kwargs: Any,
) -> WraptWrapper:
    """Creates a wrapt-compatible wrapper function.

    This may be difficult to reason about because we wrap a wrapper. In simplest
    terms, this function takes a function that needs a WrappedFunctionSpec and
    returns a function that can be passed into wrapt.wrap_function_wrapper.

    Example usage:
    ```python
    # This function's signature expects a WrappedFunctionSpec as the first argument.
    # This is because we need the spec to determine the span name, etc.

    def handler(wrapped_spec: WrappedFunctionSpec, wrapped: Callable, instance: Any, args, kwargs):
        # Some handler logic, in our case, set the span attributes
        return wrapped(*args, **kwargs)

    wrapped_spec = {
        # ... other fields ...
        "wrapper_function": handler,
    }

    # In wrapt:
    wrapt.wrap_function_wrapper(
        "package.subpackage.module",
        "Object.method_name",
        add_spec_wrapper(handler, wrapped_spec),
    )
    ```

    `handler_kwargs` are forwarded to the handler as keyword arguments on every
    call. This is how instrumentor-level (rather than per-method) collaborators
    reach a wrapper — see `BaseLaminarInstrumentor.wrapper_kwargs`.

    Args:
        wrapt_handler (WrapperHandler): actual handler that will wrap the function.
        wrapped_spec (WrappedFunctionSpec): specification of the function to wrap.

    Returns:
        WraptWrapper: function that can be passed into wrapt.wrap_function_wrapper.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Sequence[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        # wrapt always passes args/kwargs, but the parameters are declared
        # optional for direct callers; normalize so the handler's signature does
        # not have to admit None.
        return wrapt_handler(
            wrapped_spec, wrapped, instance, args or (), kwargs or {}, **handler_kwargs
        )

    return wrapper


def set_instrumentation_scope_attributes(
    span: Span | None, scope: LaminarInstrumentationScopeAttributes | None
) -> None:
    """Record which instrumentation produced this span.

    Instrumentors on this shape create spans through `Laminar.start_span`, which
    resolves the single `lmnr.tracer` OTel tracer — so the real OTel
    InstrumentationScope is the same for all of them and cannot identify the
    source library. These attributes carry that information instead.

    Prefer `stamp_instrumentation_scope`, which reads the scope off the spec.
    This lower-level entry point exists for the handful of span-creating code
    paths that are not spec wrappers and so have no `to_wrap` to read (e.g.
    skyvern's global `LLM_API_HANDLER` swap), which take the scope from their
    instrumentor directly.
    """
    if span is None or not scope:
        return
    set_span_attribute(
        span, "lmnr.span.instrumentation_scope.name", scope.get("name")
    )
    set_span_attribute(
        span, "lmnr.span.instrumentation_scope.version", scope.get("version")
    )


def stamp_instrumentation_scope(
    span: Span | None, to_wrap: WrappedFunctionSpec
) -> None:
    """Record which instrumentation produced this span, reading the spec.

    See `set_instrumentation_scope_attributes` for why these attributes exist.
    """
    set_instrumentation_scope_attributes(
        span, to_wrap.get("instrumentation_scope")
    )

from typing import Any, Callable, Sequence

from .types import WrappedFunctionSpec


def add_spec_wrapper(
    wrapt_handler: Callable[
        [WrappedFunctionSpec, Callable, Any, Sequence[Any], dict[str, Any]], Any
    ],
    wrapped_spec: WrappedFunctionSpec,
    **handler_kwargs,
) -> Callable[[Callable, Any, Sequence[Any], dict[str, Any]], Any]:
    """Creates a wrapt-compatible wrapper function.

    This may be difficult to reason about because we wrap a wrapper. In simplest
    terms, this function takes a function that needs a WrappedFunctionSpec and
    returns a function that can be passed into wrapt.wrap_function_wrapper.

    Example usage:
    ```python
    # This functions signature expects a WrappedFunctionSpec as the first argument.
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

    Args:
        wrapt_handler (Callable): actual handler that will wrap the function.
        wrapped_spec (WrappedFunctionSpec): specification of the function to wrap.

    Returns:
        Callable[[Callable, Any, Sequence[Any], dict[str, Any]], Any]: function that
        can be passed into wrapt.wrap_function_wrapper.
    """

    def wrapper(
        wrapped: Callable,
        instance: Any,
        args: Sequence[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        return wrapt_handler(
            wrapped_spec, wrapped, instance, args, kwargs, **handler_kwargs
        )

    return wrapper

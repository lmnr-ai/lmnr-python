from typing import Callable, Optional

from .utils import is_async

from traceloop.sdk.decorators.base import (
    entity_method,
    aentity_method,
)


def observe(
    *,
    name: Optional[str] = None,
    release: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """The main decorator entrypoint for Laminar. This is used to wrap functions and methods to create spans.

    Args:
        name (Optional[str], optional): Name of the span. Function name is used if not specified. Defaults to None.

    Raises:
        Exception: re-raises the exception if the wrapped function raises an exception

    Returns:
        Any: Returns the result of the wrapped function
    """

    def decorator(func: Callable):
        return (
            aentity_method(name=name)(func)
            if is_async(func)
            else entity_method(name=name)(func)
        )

    return decorator

from lmnr.traceloop_sdk.decorators.base import (
    entity_method,
    aentity_method,
)
from opentelemetry.trace import INVALID_SPAN, get_current_span

from typing import Callable, Optional, TypeVar, cast
from typing_extensions import ParamSpec

from lmnr.traceloop_sdk.tracing.attributes import SESSION_ID, USER_ID
from lmnr.traceloop_sdk.tracing.tracing import update_association_properties

from .utils import is_async


P = ParamSpec("P")
R = TypeVar("R")


def observe(
    *,
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """The main decorator entrypoint for Laminar. This is used to wrap
    functions and methods to create spans.

    Args:
        name (Optional[str], optional): Name of the span. Function
                        name is used if not specified.
                        Defaults to None.
        user_id (Optional[str], optional): User ID to associate
                        with the span and the following context.
                        Defaults to None.
        session_id (Optional[str], optional): Session ID to associate with the
                        span and the following context. Defaults to None.

    Raises:
        Exception: re-raises the exception if the wrapped function raises
                   an exception

    Returns:
        R: Returns the result of the wrapped function
    """

    def decorator(func: Callable) -> Callable:
        current_span = get_current_span()
        if current_span != INVALID_SPAN:
            if session_id is not None:
                current_span.set_attribute(SESSION_ID, session_id)
            if user_id is not None:
                current_span.set_attribute(USER_ID, user_id)
        association_properties = {}
        if session_id is not None:
            association_properties["session_id"] = session_id
        if user_id is not None:
            association_properties["user_id"] = user_id
        update_association_properties(association_properties)
        return (
            aentity_method(name=name)(func)
            if is_async(func)
            else entity_method(name=name)(func)
        )

    return cast(Callable, decorator)

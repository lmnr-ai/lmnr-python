from lmnr.opentelemetry_lib.decorators import (
    entity_method,
    aentity_method,
    json_dumps,
)
from opentelemetry.trace import INVALID_SPAN, get_current_span

from typing import Any, Callable, Literal, TypeVar, cast
from typing_extensions import ParamSpec

from lmnr.opentelemetry_lib.tracing.attributes import SESSION_ID

from .utils import is_async


P = ParamSpec("P")
R = TypeVar("R")


def observe(
    *,
    name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    ignore_input: bool = False,
    ignore_output: bool = False,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    ignore_inputs: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """The main decorator entrypoint for Laminar. This is used to wrap
    functions and methods to create spans.

    Args:
        name (str | None, optional): Name of the span. Function name is used if\
            not specified. Defaults to None.
        session_id (str | None, optional): Session ID to associate with the\
            span and the following context. Defaults to None.
        user_id (str | None, optional): User ID to associate with the span and\
            the following context. This is different from ID of a Laminar user.
            Defaults to None.
        ignore_input (bool, optional): Whether to ignore ALL input of the\
            wrapped function. Defaults to False.
        ignore_output (bool, optional): Whether to ignore ALL output of the\
            wrapped function. Defaults to False.
        span_type (Literal["DEFAULT", "LLM", "TOOL"], optional): Type of the span.
            Defaults to "DEFAULT".
        ignore_inputs (list[str] | None, optional): List of input keys to\
            ignore. For example, if the wrapped function takes three arguments\
            def foo(a, b, `sensitive_data`), and you want to ignore the\
            `sensitive_data` argument, you can pass ["sensitive_data"] to\
            this argument. Defaults to None.
        metadata (dict[str, Any] | None, optional): Metadata to associate with\
            the trace. Must be JSON serializable. Defaults to None.
    Raises:
        Exception: re-raises the exception if the wrapped function raises an\
            exception

    Returns:
        R: Returns the result of the wrapped function
    """

    def decorator(func: Callable) -> Callable:
        current_span = get_current_span()
        if current_span != INVALID_SPAN:
            if session_id is not None:
                current_span.set_attribute(SESSION_ID, session_id)
        association_properties = {}
        if session_id is not None:
            association_properties["session_id"] = session_id
        if user_id is not None:
            association_properties["user_id"] = user_id
        if metadata is not None:
            association_properties.update(
                {
                    f"metadata.{k}": (
                        v if isinstance(v, (str, int, float, bool)) else json_dumps(v)
                    )
                    for k, v in metadata.items()
                }
            )
        result = (
            aentity_method(
                name=name,
                ignore_input=ignore_input,
                ignore_output=ignore_output,
                span_type=span_type,
                ignore_inputs=ignore_inputs,
                association_properties=association_properties,
            )(func)
            if is_async(func)
            else entity_method(
                name=name,
                ignore_input=ignore_input,
                ignore_output=ignore_output,
                span_type=span_type,
                ignore_inputs=ignore_inputs,
                association_properties=association_properties,
            )(func)
        )
        return result

    return cast(Callable, decorator)

from lmnr.opentelemetry_lib.decorators import (
    observe_base,
    async_observe_base,
)

from typing import Any, Callable, Coroutine, Literal, TypeVar, overload
from typing_extensions import ParamSpec

from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import TraceType

from .utils import is_async
import os

logger = get_default_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# Overload for synchronous functions
@overload
def observe(
    *,
    name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    ignore_input: bool = False,
    ignore_output: bool = False,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    ignore_inputs: list[str] | None = None,
    input_formatter: Callable[..., str] | None = None,
    output_formatter: Callable[..., str] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    preserve_global_context: bool = False,
    rollout_entrypoint: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


# Overload for asynchronous functions
@overload
def observe(
    *,
    name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    ignore_input: bool = False,
    ignore_output: bool = False,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    ignore_inputs: list[str] | None = None,
    input_formatter: Callable[..., str] | None = None,
    output_formatter: Callable[..., str] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    preserve_global_context: bool = False,
    rollout_entrypoint: bool = False,
) -> Callable[
    [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
]: ...


# Implementation
def observe(
    *,
    name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    ignore_input: bool = False,
    ignore_output: bool = False,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    ignore_inputs: list[str] | None = None,
    input_formatter: Callable[..., str] | None = None,
    output_formatter: Callable[..., str] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    preserve_global_context: bool = False,
    rollout_entrypoint: bool = False,
):
    # Return type is determined by overloads above
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
        input_formatter (Callable[P, str] | None, optional): A custom function\
            to format the input of the wrapped function. This function should\
            accept the same parameters as the wrapped function and return a string.\
            All function arguments are passed to this function. Ignored if\
            `ignore_input` is True. Does not respect `ignore_inputs` argument.
            Defaults to None.
        output_formatter (Callable[[R], str] | None, optional): A custom function\
            to format the output of the wrapped function. This function should\
            accept a single parameter (the return value of the wrapped function)\
            and return a string. Ignored if `ignore_output` is True.\
            Defaults to None.
        metadata (dict[str, Any] | None, optional): Metadata to associate with\
            the trace. Must be JSON serializable. Defaults to None.
        tags (list[str] | None, optional): Tags to associate with the trace.
            Defaults to None.
        preserve_global_context (bool, optional): Whether to preserve the global\
            OpenTelemetry context. If set to True, Laminar spans will continue\
            traces started in the global context. Defaults to False.
        rollout_entrypoint (bool, optional): Whether to mark this function as a\
            rollout entrypoint. When True and in rollout mode, the function will\
            be registered for rollout execution via the CLI. Defaults to False.
    Raises:
        Exception: re-raises the exception if the wrapped function raises an\
            exception

    Returns:
        R: Returns the result of the wrapped function
    """

    def decorator(
        func: Callable[P, R] | Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, R] | Callable[P, Coroutine[Any, Any, R]]:
        # Handle rollout entrypoint registration
        if rollout_entrypoint:
            from lmnr.sdk.rollout_control import is_rollout_mode, register_entrypoint

            if is_rollout_mode():
                # Register the function for rollout execution
                entrypoint_name = name if name is not None else func.__name__
                register_entrypoint(entrypoint_name, func)

        # Get rollout session ID from environment if in rollout mode
        rollout_session_id = os.environ.get("LMNR_ROLLOUT_SESSION_ID")

        association_properties = {}
        if session_id is not None:
            association_properties["session_id"] = session_id
        if user_id is not None:
            association_properties["user_id"] = user_id
        if rollout_session_id is not None:
            association_properties["rollout_session_id"] = rollout_session_id
        if span_type in ["EVALUATION", "EXECUTOR", "EVALUATOR"]:
            association_properties["trace_type"] = TraceType.EVALUATION.value
        if tags is not None:
            if not isinstance(tags, list) or not all(
                isinstance(tag, str) for tag in tags
            ):
                logger.warning("Tags must be a list of strings. Tags will be ignored.")
            else:
                # list(set(tags)) to deduplicate tags
                association_properties["tags"] = list(set(tags))
        if input_formatter is not None and ignore_input:
            logger.warning(
                f"observe, function {func.__name__}: Input formatter"
                " is ignored because `ignore_input` is True. Specify only one of"
                " `ignore_input` or `input_formatter`."
            )
        if input_formatter is not None and ignore_inputs is not None:
            logger.warning(
                f"observe, function {func.__name__}: Both input formatter and"
                " `ignore_inputs` are specified. Input formatter"
                " will pass all arguments to the formatter regardless of"
                " `ignore_inputs`."
            )
        if output_formatter is not None and ignore_output:
            logger.warning(
                f"observe, function {func.__name__}: Output formatter"
                " is ignored because `ignore_output` is True. Specify only one of"
                " `ignore_output` or `output_formatter`."
            )

        # Merge rollout.session_id into metadata if in rollout mode
        merged_metadata = metadata.copy() if metadata else {}
        if rollout_session_id is not None:
            merged_metadata["rollout.session_id"] = rollout_session_id

        if is_async(func):
            return async_observe_base(
                name=name,
                ignore_input=ignore_input,
                ignore_output=ignore_output,
                span_type=span_type,
                metadata=merged_metadata if merged_metadata else None,
                ignore_inputs=ignore_inputs,
                input_formatter=input_formatter,
                output_formatter=output_formatter,
                association_properties=association_properties,
                preserve_global_context=preserve_global_context,
            )(func)
        else:
            return observe_base(
                name=name,
                ignore_input=ignore_input,
                ignore_output=ignore_output,
                span_type=span_type,
                metadata=merged_metadata if merged_metadata else None,
                ignore_inputs=ignore_inputs,
                input_formatter=input_formatter,
                output_formatter=output_formatter,
                association_properties=association_properties,
                preserve_global_context=preserve_global_context,
            )(func)

    return decorator

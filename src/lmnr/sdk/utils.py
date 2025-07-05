import asyncio
import datetime
import dataclasses
import dotenv
import enum
import inspect
import os
import pydantic
import queue
import typing
import uuid

from opentelemetry.trace import Span, NonRecordingSpan, SpanContext, TraceFlags


def is_span_context_trace_flags_valid(span: Span) -> bool:
    """As of writing this function, some telemetry library,
    possibly datadog lambda handler, returns `trace_flags` as a raw
    int, not a `TraceFlags(int)` object. But tracer.start_span() checks for
    `TraceFlags(int).sampled` on the current (parent) span when creating a span.

    To avoid errors in start_span/start_as_current_span, we check for the
    presence of `TraceFlags(int).sampled` on the span.
    """
    result = hasattr(span.get_span_context().trace_flags, "sampled")
    print(f"\n\n\n{result}\n\n\n")
    return result


def fix_span_context_trace_flags(span: Span) -> Span:
    """Fix the span context trace flags if they are not a `TraceFlags(int)` object."""
    print("\n\n\nhealing span context\n\n\n")
    return NonRecordingSpan(
        SpanContext(
            trace_id=span.get_span_context().trace_id,
            span_id=span.get_span_context().span_id,
            is_remote=span.get_span_context().is_remote,
            trace_state=span.get_span_context().trace_state,
            trace_flags=TraceFlags(span.get_span_context().trace_flags),
        )
    )


def is_method(func: typing.Callable) -> bool:
    # inspect.ismethod is True for bound methods only, but in the decorator,
    # the method is not bound yet, so we need to check if the first parameter
    # is either 'self' or 'cls'. This only relies on naming conventions

    # `signature._parameters` is an OrderedDict,
    # so the order of insertion is preserved
    params = list(inspect.signature(func).parameters)
    return len(params) > 0 and params[0] in ["self", "cls"]


def is_async(func: typing.Callable) -> bool:
    # `__wrapped__` is set automatically by `functools.wraps` and
    # `functools.update_wrapper`
    # so we can use it to get the original function
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__

    if not inspect.isfunction(func):
        return False

    # Check if the function is asynchronous
    if asyncio.iscoroutinefunction(func):
        return True

    # Fallback: check if the function's code object contains 'async'.
    # This is for cases when a decorator (not ours) did not properly use
    # `functools.wraps` or `functools.update_wrapper`
    return (func.__code__.co_flags & inspect.CO_COROUTINE) != 0


def is_async_iterator(o: typing.Any) -> bool:
    return hasattr(o, "__aiter__") and hasattr(o, "__anext__")


def is_iterator(o: typing.Any) -> bool:
    return hasattr(o, "__iter__") and hasattr(o, "__next__")


def serialize(obj: typing.Any) -> str | dict[str, typing.Any]:
    def serialize_inner(o: typing.Any):
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        elif o is None:
            return None
        elif isinstance(o, (int, float, str, bool)):
            return o
        elif isinstance(o, uuid.UUID):
            return str(o)  # same as in final return, but explicit
        elif isinstance(o, enum.Enum):
            return o.value
        elif dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif isinstance(o, bytes):
            return o.decode("utf-8")
        elif isinstance(o, pydantic.BaseModel):
            return o.model_dump()
        elif isinstance(o, (tuple, set, frozenset)):
            return [serialize_inner(item) for item in o]
        elif isinstance(o, list):
            return [serialize_inner(item) for item in o]
        elif isinstance(o, dict):
            return {serialize_inner(k): serialize_inner(v) for k, v in o.items()}
        elif isinstance(o, queue.Queue):
            return type(o).__name__

        return str(o)

    return serialize_inner(obj)


def get_input_from_func_args(
    func: typing.Callable,
    is_method: bool = False,
    func_args: list[typing.Any] = [],
    func_kwargs: dict[str, typing.Any] = {},
    ignore_inputs: list[str] | None = None,
) -> dict[str, typing.Any]:
    # Remove implicitly passed "self" or "cls" argument for
    # instance or class methods
    res = {
        k: v
        for k, v in func_kwargs.items()
        if not (ignore_inputs and k in ignore_inputs)
    }
    for i, k in enumerate(inspect.signature(func).parameters.keys()):
        if is_method and k in ["self", "cls"]:
            continue
        if ignore_inputs and k in ignore_inputs:
            continue
        # If param has default value, then it's not present in func args
        if i < len(func_args):
            res[k] = func_args[i]
    return res


def from_env(key: str) -> str | None:
    if val := os.getenv(key):
        return val
    dotenv_path = dotenv.find_dotenv(usecwd=True)
    # use DotEnv directly so we can set verbose to False
    return dotenv.main.DotEnv(dotenv_path, verbose=False, encoding="utf-8").get(key)


def is_otel_attribute_value_type(value: typing.Any) -> bool:
    def is_primitive_type(value: typing.Any) -> bool:
        return isinstance(value, (int, float, str, bool))

    if is_primitive_type(value):
        return True
    elif isinstance(value, typing.Sequence):
        if len(value) > 0:
            return is_primitive_type(value[0]) and all(
                isinstance(v, type(value[0])) for v in value
            )
        return True
    return False

import asyncio
import datetime
import dataclasses
import enum
import inspect
import pydantic
import queue
import typing
import uuid


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
    # This is for cases when a decorator did not properly use
    # `functools.wraps` or `functools.update_wrapper`
    CO_COROUTINE = inspect.CO_COROUTINE
    return (func.__code__.co_flags & CO_COROUTINE) != 0


def is_async_iterator(o: typing.Any) -> bool:
    return hasattr(o, "__aiter__") and hasattr(o, "__anext__")


def is_iterator(o: typing.Any) -> bool:
    return hasattr(o, "__iter__") and hasattr(o, "__next__")


def serialize(obj: typing.Any) -> dict[str, typing.Any]:
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
) -> dict[str, typing.Any]:
    # Remove implicitly passed "self" or "cls" argument for
    # instance or class methods
    res = func_kwargs.copy()
    for i, k in enumerate(inspect.signature(func).parameters.keys()):
        if is_method and k in ["self", "cls"]:
            continue
        # If param has default value, then it's not present in func args
        if i < len(func_args):
            res[k] = func_args[i]
    return res

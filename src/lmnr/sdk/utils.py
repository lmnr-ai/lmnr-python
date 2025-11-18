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
    if inspect.iscoroutinefunction(func):
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


def get_otel_env_var(var_name: str) -> str | None:
    """Get OTEL environment variable with priority order.

    Checks in order:
    1. OTEL_EXPORTER_OTLP_TRACES_{var_name}
    2. OTEL_EXPORTER_OTLP_{var_name}
    3. OTEL_{var_name}

    Args:
        var_name: The variable name (e.g., 'ENDPOINT', 'HEADERS', 'TIMEOUT')

    Returns:
        str | None: The environment variable value or None if not found
    """
    candidates = [
        f"OTEL_EXPORTER_OTLP_TRACES_{var_name}",
        f"OTEL_EXPORTER_OTLP_{var_name}",
        f"OTEL_{var_name}",
    ]

    for candidate in candidates:
        if value := from_env(candidate):
            return value
    return None


def parse_otel_headers(headers_str: str | None) -> dict[str, str]:
    """Parse OTEL headers string into dictionary.

    Format: key1=value1,key2=value2
    Values are URL-decoded.

    Args:
        headers_str: Headers string in OTEL format

    Returns:
        dict[str, str]: Parsed headers dictionary
    """
    if not headers_str:
        return {}

    headers = {}
    for pair in headers_str.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            import urllib.parse

            headers[key.strip()] = urllib.parse.unquote(value.strip())
    return headers


def format_id(id_value: str | int | uuid.UUID) -> str:
    """Format trace/span/evaluation ID to a UUID string, or return valid UUID strings as-is.

    Args:
        id_value: The ID in various formats (UUID, int, or valid UUID string)

    Returns:
        str: UUID string representation

    Raises:
        ValueError: If id_value cannot be converted to a valid UUID
    """
    if isinstance(id_value, uuid.UUID):
        return str(id_value)
    elif isinstance(id_value, int):
        return str(uuid.UUID(int=id_value))
    elif isinstance(id_value, str):
        uuid.UUID(id_value)
        return id_value
    else:
        raise ValueError(f"Invalid ID type: {type(id_value)}")

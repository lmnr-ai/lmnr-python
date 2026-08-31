import base64
import collections.abc
import dataclasses
import datetime
import enum
import functools
import inspect
import os
import queue
import re
import typing
import uuid

import dotenv
import orjson
import pydantic
from opentelemetry.trace import Tracer

from lmnr.sdk.log import get_default_logger

if typing.TYPE_CHECKING:
    from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
    from lmnr.sdk.client.synchronous.sync_client import LaminarClient

logger = get_default_logger(__name__)

WrappedFunction = typing.Callable[..., typing.Any]

#: The shape wrapt's `wrap_function_wrapper` expects.
InstrumentedWrapper = typing.Callable[
    [WrappedFunction, typing.Any, tuple[typing.Any, ...], dict[str, typing.Any]],
    typing.Any,
]

#: Deliberately UNBOUND. `to_wrap` has no single shape across the legacy
#: instrumentations — most pass a `dict`, but langgraph passes a bare `str`
#: method path. Binding this to a spec type would break that caller. The
#: instrumentations already on `BaseLaminarInstrumentor` use the typed
#: `WrappedFunctionSpec` contract instead of these helpers.
ToWrapT = typing.TypeVar("ToWrapT")


def with_tracer_wrapper(
    func: typing.Callable[
        [
            Tracer,
            ToWrapT,
            WrappedFunction,
            typing.Any,
            tuple[typing.Any, ...],
            dict[str, typing.Any],
        ],
        typing.Any,
    ],
) -> typing.Callable[[Tracer, ToWrapT], InstrumentedWrapper]:
    """Bind a tracer and a per-instrumented-method config into an instrumentation
    function, producing the wrapper factory `wrapt.wrap_function_wrapper` expects.

    `func` must accept `(tracer, to_wrap, wrapped, instance, args, kwargs)`; the
    type of `to_wrap` flows through, so a wrapper annotating it as its own spec
    type gets that type checked at the `wrap_function_wrapper` call site.

    Usage:
    `wrap_function_wrapper(mod, "method", with_tracer_wrapper(f)(tracer, to_wrap))`.
    Use `with_tracer_only_wrapper` when there is no per-method config.
    """

    def _with_tracer(tracer: Tracer, to_wrap: ToWrapT) -> InstrumentedWrapper:
        @functools.wraps(func)
        def wrapper(
            wrapped: WrappedFunction,
            instance: typing.Any,
            args: tuple[typing.Any, ...],
            kwargs: dict[str, typing.Any],
        ) -> typing.Any:
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def with_tracer_only_wrapper(
    func: typing.Callable[
        [
            Tracer,
            WrappedFunction,
            typing.Any,
            tuple[typing.Any, ...],
            dict[str, typing.Any],
        ],
        typing.Any,
    ],
) -> typing.Callable[[Tracer], InstrumentedWrapper]:
    """`with_tracer_wrapper` for instrumentations with no per-method config.

    `func` must accept `(tracer, wrapped, instance, args, kwargs)`. Every wrapper
    in the openai tree is of this shape — it wraps a fixed set of hand-written
    targets, so there is nothing per-method to thread through.
    """

    def _with_tracer(tracer: Tracer) -> InstrumentedWrapper:
        @functools.wraps(func)
        def wrapper(
            wrapped: WrappedFunction,
            instance: typing.Any,
            args: tuple[typing.Any, ...],
            kwargs: dict[str, typing.Any],
        ) -> typing.Any:
            return func(tracer, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def with_tracer_and_client_wrapper(
    func: typing.Callable[
        [
            Tracer,
            "LaminarClient | AsyncLaminarClient",
            ToWrapT,
            WrappedFunction,
            typing.Any,
            tuple[typing.Any, ...],
            dict[str, typing.Any],
        ],
        typing.Any,
    ],
) -> typing.Callable[
    [Tracer, "LaminarClient | AsyncLaminarClient", ToWrapT], InstrumentedWrapper
]:
    """Same as `with_tracer_wrapper`, but also binds a Laminar client.

    `func` must accept
    `(tracer, client, to_wrap, wrapped, instance, args, kwargs)`.
    """

    def _with_tracer_and_client(
        tracer: Tracer,
        client: "LaminarClient | AsyncLaminarClient",
        to_wrap: ToWrapT,
    ) -> InstrumentedWrapper:
        @functools.wraps(func)
        def wrapper(
            wrapped: WrappedFunction,
            instance: typing.Any,
            args: tuple[typing.Any, ...],
            kwargs: dict[str, typing.Any],
        ) -> typing.Any:
            return func(tracer, client, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer_and_client


def is_method(func: typing.Callable[..., typing.Any]) -> bool:
    # inspect.ismethod is True for bound methods only, but in the decorator,
    # the method is not bound yet, so we need to check if the first parameter
    # is either 'self' or 'cls'. This only relies on naming conventions

    # `signature._parameters` is an OrderedDict,
    # so the order of insertion is preserved
    params = list(inspect.signature(func).parameters)
    return len(params) > 0 and params[0] in ["self", "cls"]


def is_async(func: typing.Callable[..., typing.Any]) -> bool:
    # `__wrapped__` is set automatically by `functools.wraps` and
    # `functools.update_wrapper`
    # so we can use it to get the original function
    try:
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
    except Exception:
        logger.debug("Failed to check if function is asynchronous", exc_info=True)
        return False


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
            return serialize(o.model_dump())
        elif isinstance(o, (tuple, set, frozenset, list)):
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
    try:
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
    except Exception:
        logger.warning("Failed to get input from func args")
        return {}


def from_env(key: str) -> str | None:
    if val := os.getenv(key):
        return val
    try:
        dotenv_path = dotenv.find_dotenv(usecwd=True)
        # use DotEnv directly so we can set verbose to False
        return dotenv.main.DotEnv(dotenv_path, verbose=False, encoding="utf-8").get(key)
    except Exception:
        logger.warning(f"Failed to get environment variable from dotenv. Key: {key}")
        return None


def get_frontend_url(
    base_url: str | None = None, frontend_port: int | None = None
) -> str:
    """
    Get the frontend URL from the base API URL.

    Converts API URLs to frontend URLs:
    - https://api.lmnr.ai -> https://www.laminar.sh
    - http://localhost:8000 -> http://localhost:5667 (or custom frontend_port)
    - http://127.0.0.1:8000 -> http://127.0.0.1:5667 (or custom frontend_port)

    Args:
        base_url: Base API URL (defaults to https://api.lmnr.ai)
        frontend_port: Optional frontend port for localhost (defaults to 5667)

    Returns:
        Frontend URL
    """
    if not base_url or base_url == "https://api.lmnr.ai":
        base_url = "https://www.laminar.sh"
        return base_url

    url = base_url.rstrip("/")

    # Handle localhost/127.0.0.1 - set frontend port
    if "localhost" in url or "127.0.0.1" in url:
        # Remove existing port if present
        url = re.sub(r":\d+$", "", url)
        # Add frontend port (default 5667)
        port = frontend_port or 5667
        url = f"{url}:{port}"

    return url


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


DEFAULT_PLACEHOLDER = {}


_UNWRAP_MISS = object()


def _unwrap_container(o: typing.Any) -> typing.Any:
    """Open a container into a plain dict/list, or return `_UNWRAP_MISS`.

    Single source of truth for "what counts as a container", shared by
    `default_json` and `_stringify_dict_keys`. Those two MUST agree: the retry
    walker can only repair a bad mapping key if it recurses into every container
    `default_json` is willing to open, and every past divergence here (plain
    dict/list only, then missing `Mapping`, then missing `BaseModel`) silently
    collapsed a whole span attribute to "{}". Add new container types here, once.

    Note `str`/`bytes`/`bytearray` are `Sequence`s and must NOT be opened — that
    would explode them into per-character lists.
    """
    if isinstance(o, pydantic.BaseModel):
        return o.model_dump()
    if isinstance(o, collections.abc.Mapping):
        return dict(o)
    if isinstance(o, (set, frozenset)):
        return list(o)
    if isinstance(o, collections.abc.Sequence) and not isinstance(
        o, (str, bytes, bytearray)
    ):
        return list(o)
    return _UNWRAP_MISS


def default_json(o):
    # STANDARD base64 (`+`/`/`), not pydantic's URL-safe `ser_json_bytes`
    # alphabet: consumers decode with `base64.b64decode`, which defaults to
    # `validate=False` and silently DROPS out-of-alphabet characters instead of
    # raising, shifting every following byte. Without this branch bytes fall
    # through to `str(o)` and serialize as the useless repr "b'\\xfb\\xef'".
    # Must precede the container check — bytes are Sequences.
    if isinstance(o, (bytes, bytearray)):
        return base64.b64encode(o).decode("utf-8")

    # Opening a dict-like / list-like BEFORE the str() fallback matters: a
    # Mapping or custom Sequence would otherwise stringify to a Python repr with
    # SINGLE quotes — "{'a': 1}" — which is not JSON and no consumer can parse.
    unwrapped = _unwrap_container(o)
    if unwrapped is not _UNWRAP_MISS:
        return unwrapped

    try:
        return str(o)
    except Exception:
        logger.debug("Failed to serialize data to JSON, inner type: %s", type(o))
        pass
    return DEFAULT_PLACEHOLDER


MAX_ERROR_BODY_CHARS = 2000


def describe_response(response) -> str:
    """Render an HTTP error response as a readable one-liner.

    Error bodies are not always JSON — app-server returns plain text for
    payload-limit (413) rejections and some proxies return HTML — so this never
    assumes a shape, and truncates so a large body can't flood the caller's logs.
    """
    if response is None:
        return "no response received"
    body = (response.text or "").strip()
    if len(body) > MAX_ERROR_BODY_CHARS:
        body = body[:MAX_ERROR_BODY_CHARS] + "... (truncated)"
    if not body:
        body = "<empty body>"
    return f"[{response.status_code}] {body}"


_JSON_DUMPS_OPTIONS = (
    orjson.OPT_SERIALIZE_DATACLASS
    | orjson.OPT_SERIALIZE_UUID
    | orjson.OPT_UTC_Z
    | orjson.OPT_NON_STR_KEYS
)


#: Key types `OPT_NON_STR_KEYS` encodes itself. Left untouched on the retry path
#: so a key's name never depends on whether some unrelated sibling key forced
#: the retry — orjson renders these differently from `str()` (`None` -> "null",
#: `True` -> "true", datetimes -> RFC 3339, enums -> their value).
_ORJSON_NATIVE_KEY_TYPES = (
    str,
    int,  # covers bool and IntEnum
    float,
    type(None),
    uuid.UUID,
    datetime.datetime,
    datetime.date,
    datetime.time,
    enum.Enum,
)


def _stringify_dict_keys(value: typing.Any) -> typing.Any:
    """Coerce the mapping keys orjson cannot encode into strings.

    `OPT_NON_STR_KEYS` only covers a fixed set of scalar key types, and orjson
    raises on the rest (tuples, bytes, Decimal, arbitrary objects) — which would
    collapse the whole payload to "{}". Only called on the retry path, so the
    common case pays nothing.

    Recurses through EVERY container `_unwrap_container` opens — pydantic models
    and Mappings included, not just the builtins. A bad key nested inside one
    `default_json` would open but this walker skips survives the retry, fails the
    second dump too, and loses every sibling field with it.
    """
    # bytes are a Sequence, so check them before unwrapping.
    if isinstance(value, (bytes, bytearray)):
        return value

    opened = _unwrap_container(value)
    if opened is _UNWRAP_MISS:
        return value

    if isinstance(opened, dict):
        return {
            (
                key
                if isinstance(key, _ORJSON_NATIVE_KEY_TYPES)
                else (
                    base64.b64encode(key).decode("utf-8")
                    if isinstance(key, (bytes, bytearray))
                    else str(key)
                )
            ): _stringify_dict_keys(inner)
            for key, inner in opened.items()
        }
    return [_stringify_dict_keys(item) for item in opened]


def json_dumps(data: dict | list) -> str:
    try:
        return orjson.dumps(
            data,
            default=default_json,
            option=_JSON_DUMPS_OPTIONS,
        ).decode("utf-8")
    except Exception:
        pass
    try:
        # An unencodable mapping key is the one failure worth retrying — it
        # aborts the whole document, losing every sibling value with it.
        return orjson.dumps(
            _stringify_dict_keys(data),
            default=default_json,
            option=_JSON_DUMPS_OPTIONS,
        ).decode("utf-8")
    except Exception:
        # Log the exception and return a placeholder if serialization completely fails
        logger.info("Failed to serialize data to JSON, type: %s", type(data))
        return "{}"  # Return an empty JSON object as a fallback

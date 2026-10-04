import functools
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any, ParamSpec, overload

from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel
from typing_extensions import TypeVar

from lmnr.opentelemetry_lib.tracing.attributes import SPAN_TYPE
from lmnr.opentelemetry_lib.tracing.tracer import get_tracer_with_context
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import LaminarSpanType
from lmnr.sdk.utils import is_async

logger = get_default_logger(__name__)
T = TypeVar("T")
P = ParamSpec("P")


@overload
def dont_throw(
    func: Callable[P, Awaitable[T]],
) -> Callable[P, Awaitable[T | None]]: ...


@overload
def dont_throw(func: Callable[P, T]) -> Callable[P, T | None]: ...


def dont_throw(func: Callable[P, Any]) -> Callable[P, Any]:  # pyright: ignore[reportExplicitAny]
    """
    A decorator that wraps the passed in function and logs exceptions instead of
    throwing them. Works for both synchronous and asynchronous functions.
    """
    func_logger = get_default_logger(func.__module__)

    if is_async(func):

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:  # pyright: ignore[reportExplicitAny]
            try:
                return await func(*args, **kwargs)
            except Exception:
                func_logger.debug(
                    "Laminar failed to trace in %s", func.__name__, exc_info=True
                )
                return None

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:  # pyright: ignore[reportExplicitAny]
        try:
            return func(*args, **kwargs)
        except Exception:
            func_logger.debug(
                "Laminar failed to trace in %s", func.__name__, exc_info=True
            )
            return None

    return wrapper


def set_span_attribute(
    span: Span, attribute_name: str, attribute_value: AttributeValue | None
):
    if attribute_value is not None and attribute_value != "":
        span.set_attribute(attribute_name, attribute_value)


def to_dict(obj: Any) -> dict[str, Any]:  # pyright: ignore[reportAny, reportExplicitAny]
    try:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return deepcopy(obj)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        elif obj is None:
            return {}
        else:
            return dict(obj)  # pyright: ignore[reportAny]
    except Exception:
        logger.debug(f"Error converting to dict: {obj}", exc_info=True)
        return {}


def model_as_dict(model: Any) -> dict[str, Any]:  # pyright: ignore[reportAny, reportExplicitAny]
    """Convert a pydantic model or raw API response (`.parse()`) to a dict.

    Dicts are returned as-is (no copy). Returns `{}` if conversion fails.
    """
    try:
        if isinstance(model, dict):
            return model  # pyright: ignore[reportUnknownVariableType]
        if hasattr(model, "model_dump"):  # pyright: ignore[reportAny]
            return model.model_dump()  # pyright: ignore[reportAny]
        if hasattr(model, "parse"):  # pyright: ignore[reportAny]
            # Raw API response
            return model_as_dict(model.parse())  # pyright: ignore[reportAny]
        return dict(model)  # pyright: ignore[reportAny]
    except Exception:
        logger.debug(f"Failed to convert model to dict: {model}", exc_info=True)
        return {}


def extract_json_schema(schema: dict[str, Any] | BaseModel) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
    if isinstance(schema, dict):
        return schema
    elif hasattr(schema, "model_json_schema") and callable(schema.model_json_schema):
        return schema.model_json_schema()
    else:
        return {}


def safe_start_span(
    name: str,
    context: Context | None = None,
    attributes: dict[str, AttributeValue] | None = None,
    span_type: LaminarSpanType = "DEFAULT",
    start_time: int | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Span | None:
    """Start a span, returning None instead of raising if that is not possible.

    `start_time` (ns) and `kind` are deliberately NOT exposed on the public
    `Laminar.start_span`, but some instrumentations genuinely need them: the
    OpenAI responses/assistants wrappers only learn a call happened once it has
    finished, so they open the span retroactively at the recorded start time.
    When either is requested we go through the tracer directly and stamp the
    Laminar-specific attributes ourselves, so the public API stays unchanged.
    """
    if not Laminar.is_initialized():
        return None
    try:
        if start_time is None and kind is SpanKind.INTERNAL:
            return Laminar.start_span(
                name, context=context, attributes=attributes, span_type=span_type
            )
        with get_tracer_with_context() as (tracer, isolated_context):
            return tracer.start_span(
                name,
                context=context or isolated_context,
                kind=kind,
                start_time=start_time,
                attributes={**(attributes or {}), SPAN_TYPE: span_type},
            )
    except Exception:
        logger.debug(f"Failed to start span: {name}", exc_info=True)
        return None

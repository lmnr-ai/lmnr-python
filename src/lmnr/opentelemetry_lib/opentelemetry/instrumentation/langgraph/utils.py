import logging
import traceback

import pydantic
from opentelemetry.trace import Span
from typing import Any


def set_span_attribute(span: Span, name: str, value: str):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.debug(
                "Laminar failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )

    return wrapper


def to_dict(obj: pydantic.BaseModel | dict) -> dict[str, Any]:
    try:
        if isinstance(obj, pydantic.BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return obj
        else:
            return dict(obj)
    except Exception:
        return dict(obj)


def with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer

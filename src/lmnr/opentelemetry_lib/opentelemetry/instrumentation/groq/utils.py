import logging
import os
from collections.abc import Callable
from importlib.metadata import version
from typing import Any, TypeVar, cast

from opentelemetry import context as context_api
from opentelemetry.trace import Span

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.groq.config import Config

_PYDANTIC_VERSION = version("pydantic")

LMNR_TRACE_CONTENT = "LMNR_TRACE_CONTENT"


T = TypeVar("T")


def set_span_attribute(
    span: Span,
    name: str,
    value: Any,  # pyright: ignore[reportExplicitAny, reportAny]
):
    if value is not None and value != "":
        span.set_attribute(name, value)  # pyright: ignore[reportAny]


def should_send_prompts():
    return (
        os.getenv(LMNR_TRACE_CONTENT) or "true"
    ).lower() == "true" or cast(bool, context_api.get_value("override_enable_content_tracing"))


def dont_throw(func: Callable[..., T])-> Callable[..., T]:
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    def wrapper(*args: Any, **kwargs: Any)-> Any:  # pyright: ignore[reportAny, reportExplicitAny]
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(
                "Laminar failed to trace in %s",
                func.__name__,
                exc_info=True,
            )
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper


def model_as_dict(model: Any) -> dict[str, Any]:  # pyright: ignore[reportAny, reportExplicitAny]
    if _PYDANTIC_VERSION < "2.0.0":
        return model.dict()  # pyright: ignore[reportAny]
    if hasattr(model, "model_dump"):  # pyright: ignore[reportAny]
        return model.model_dump()  # pyright: ignore[reportAny]
    elif hasattr(model, "parse"):    # pyright: ignore[reportAny]
        # Raw API response
        return model_as_dict(model.parse())  # pyright: ignore[reportAny]
    else:
        return model  # pyright: ignore[reportAny]

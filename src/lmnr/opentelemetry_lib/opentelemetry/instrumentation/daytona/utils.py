import logging
import traceback
from typing import Any

from opentelemetry.trace import Span

from .config import Config


def set_span_attribute(span: Span, name: str, value: Any):
    """Set a span attribute if the value is not None or empty."""
    if value is not None and value != "":
        span.set_attribute(name, value)


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    logger = logging.getLogger(func.__module__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(
                "Laminar failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper


def with_tracer_wrapper(func):
    """Helper for providing tracer and event_logger for wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, event_logger, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def async_with_tracer_wrapper(func):
    """Helper for providing tracer and event_logger for async wrapper functions."""

    def _with_tracer(tracer, event_logger, to_wrap):
        async def wrapper(wrapped, instance, args, kwargs):
            return await func(
                tracer, event_logger, to_wrap, wrapped, instance, args, kwargs
            )

        return wrapper

    return _with_tracer


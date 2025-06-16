import asyncio
import logging
import openai
import os
import re
import traceback

from importlib.metadata import version
from opentelemetry import context as context_api

_OPENAI_VERSION = version("openai")
_PYDANTIC_VERSION = version("pydantic")


def is_openai_v1():
    return _OPENAI_VERSION >= "1.0.0"


def is_azure_openai(instance):
    return is_openai_v1() and isinstance(
        instance._client, (openai.AsyncAzureOpenAI, openai.AzureOpenAI)
    )


def with_tracer_wrapper(func):
    def _with_tracer(tracer):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.
    Works for both synchronous and asynchronous functions.
    """
    logger = logging.getLogger(func.__module__)

    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    def _handle_exception(e, func, logger):
        logger.debug(
            "Laminar failed to trace in %s, error: %s",
            func.__name__,
            traceback.format_exc(),
        )

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def set_span_attribute(span, name, value):
    if value is None or value == "":
        return

    if hasattr(openai, "NOT_GIVEN") and value == openai.NOT_GIVEN:
        return

    span.set_attribute(name, value)


def should_send_prompts():
    return (
        os.getenv("LMNR_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def model_as_dict(model):
    if isinstance(model, dict):
        return model
    if _PYDANTIC_VERSION < "2.0.0":
        return model.dict()
    if hasattr(model, "model_dump"):
        return model.model_dump()
    elif hasattr(model, "parse"):  # Raw API response
        return model_as_dict(model.parse())
    else:
        return model


def is_validator_iterator(content):
    return re.search(r"pydantic.*ValidatorIterator'>$", str(type(content)))

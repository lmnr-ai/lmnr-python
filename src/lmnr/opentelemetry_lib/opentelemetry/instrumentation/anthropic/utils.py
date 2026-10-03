import asyncio
import inspect
import json
import logging
import os
import threading
import traceback
from collections.abc import Awaitable, Callable
from importlib.metadata import version
from typing import cast

from opentelemetry import context as context_api
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue
from typing_extensions import TypeVar

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic.config import Config
from lmnr.sdk.log import get_default_logger

_PYDANTIC_VERSION = version("pydantic")

LMNR_TRACE_CONTENT = "LMNR_TRACE_CONTENT"


logger = get_default_logger(__name__)


T = TypeVar("T")


def set_span_attribute(span: Span, name: str, value: AttributeValue | None):
    if value is not None and value != "":
        span.set_attribute(name, value)


def should_send_prompts() -> bool:
    return (
        os.getenv(LMNR_TRACE_CONTENT) or "true"
    ).lower() == "true" or cast(bool, context_api.get_value("override_enable_content_tracing"))


def dont_throw(func: Callable[..., T | Awaitable[T]]) -> Callable[..., T | Awaitable[T]] | None:  # pyright: ignore[reportExplicitAny]
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.
    Works for both synchronous and asynchronous functions.
    """
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
            "OpenLLMetry failed to trace in %s, error: %s",
            func.__name__,
            traceback.format_exc(),
        )
        if Config.exception_logger:
            Config.exception_logger(e)

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper


async def aextract_response_data(response):
    """Async version of _extract_response_data that can await coroutines."""
    import inspect

    # If we get a coroutine, await it
    if inspect.iscoroutine(response):
        try:
            response = await response
        except Exception:
            logger.debug("Failed to await coroutine response", exc_info=True)
            return {}

    if isinstance(response, dict):
        return response

    # Handle with_raw_response wrapped responses
    if hasattr(response, "parse") and callable(response.parse):
        try:
            # For with_raw_response, parse() gives us the actual response object
            parsed_response = response.parse()
            if not isinstance(parsed_response, dict):
                parsed_response = parsed_response.__dict__
            return parsed_response
        except Exception:
            logger.debug(
                f"Failed to parse response, response type: {type(response)}",
                exc_info=True,
            )

    # Fallback to __dict__ for regular response objects
    if hasattr(response, "__dict__"):
        response_dict = response.__dict__
        return response_dict

    return {}


def extract_response_data(response):
    """Extract the actual response data from both regular and with_raw_response wrapped responses."""
    import inspect

    # If we get a coroutine, we cannot process it in sync context
    if inspect.iscoroutine(response):
        logger.warning(
            f"_extract_response_data received coroutine {response} - response processing skipped"
        )
        return {}

    if isinstance(response, dict):
        return response

    # Handle with_raw_response wrapped responses
    if hasattr(response, "parse") and callable(response.parse):
        try:
            # For with_raw_response, parse() gives us the actual response object
            parsed_response = response.parse()
            if not isinstance(parsed_response, dict):
                parsed_response = parsed_response.__dict__
            return parsed_response
        except Exception:
            logger.debug(
                f"Failed to parse response, response type: {type(response)}",
                exc_info=True,
            )

    # Fallback to __dict__ for regular response objects
    if hasattr(response, "__dict__"):
        response_dict = response.__dict__
        return response_dict

    return {}


def run_async(method):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        thread = threading.Thread(target=lambda: asyncio.run(method))
        thread.start()
        thread.join()
    else:
        asyncio.run(method)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "to_json"):
            return o.to_json()

        if hasattr(o, "model_dump_json"):
            return o.model_dump_json()

        try:
            return str(o)
        except Exception:
            logger = logging.getLogger(__name__)
            logger.debug("Failed to serialize object of type: %s", type(o).__name__)
            return ""


def model_as_dict(model):
    if isinstance(model, dict):
        return model
    if _PYDANTIC_VERSION < "2.0.0" and hasattr(model, "dict"):
        return model.dict()
    if hasattr(model, "model_dump"):
        return model.model_dump()
    else:
        try:
            return dict(model)
        except Exception:
            return model

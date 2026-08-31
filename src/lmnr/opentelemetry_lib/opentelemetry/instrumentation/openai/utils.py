import asyncio
import inspect
import logging
import os
import threading
import traceback
from importlib.metadata import version

from opentelemetry import context as context_api
from packaging.version import parse

import openai

from .shared.config import Config

_OPENAI_VERSION = version("openai")

LMNR_TRACE_CONTENT = "LMNR_TRACE_CONTENT"


def is_openai_v1():
    return parse(_OPENAI_VERSION) >= parse("1.0.0")


def is_reasoning_supported():
    # Reasoning has been introduced in OpenAI API on Dec 17, 2024
    #     as per https://platform.openai.com/docs/changelog.
    # The updated OpenAI library version is 1.58.0
    #     as per https://pypi.org/project/openai/.
    return parse(_OPENAI_VERSION) >= parse("1.58.0")


def is_azure_openai(instance):

    return is_openai_v1() and isinstance(
        instance._client, (openai.AsyncAzureOpenAI, openai.AzureOpenAI)
    )


def is_metrics_enabled() -> bool:
    return False


def should_record_stream_token_usage():
    return Config.enrich_token_usage


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
            "OpenLLMetry failed to trace in %s, error: %s",
            func.__name__,
            traceback.format_exc(),
        )
        if Config.exception_logger:
            Config.exception_logger(e)

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper


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


def should_send_prompts():
    return (
        os.getenv(LMNR_TRACE_CONTENT) or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")

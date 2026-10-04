import asyncio
import os
import threading
from importlib.metadata import version
from types import CoroutineType
from typing import Any, cast

from opentelemetry import context as context_api
from packaging.version import parse
from typing_extensions import TypeVar

import openai

_OPENAI_VERSION = version("openai")

LMNR_TRACE_CONTENT = "LMNR_TRACE_CONTENT"
T = TypeVar("T")

def is_openai_v1():
    return parse(_OPENAI_VERSION) >= parse("1.0.0")


def is_reasoning_supported():
    # Reasoning has been introduced in OpenAI API on Dec 17, 2024
    #     as per https://platform.openai.com/docs/changelog.
    # The updated OpenAI library version is 1.58.0
    #     as per https://pypi.org/project/openai/.
    return parse(_OPENAI_VERSION) >= parse("1.58.0")


def is_azure_openai(instance: Any):

    return is_openai_v1() and isinstance(
        instance._client, (openai.AsyncAzureOpenAI, openai.AzureOpenAI)
    )


def is_metrics_enabled() -> bool:
    return False


def run_async(method: CoroutineType[None, None, Any]):
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


def should_send_prompts() -> bool:
    return (
        os.getenv(LMNR_TRACE_CONTENT) or "true"
    ).lower() == "true" or cast(bool, context_api.get_value("override_enable_content_tracing"))

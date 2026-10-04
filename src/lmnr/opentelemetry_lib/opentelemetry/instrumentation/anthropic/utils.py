import asyncio
import json
import logging
import threading

from typing_extensions import TypeVar

from lmnr.sdk.log import get_default_logger


logger = get_default_logger(__name__)


T = TypeVar("T")


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



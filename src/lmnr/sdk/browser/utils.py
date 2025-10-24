import asyncio
import logging
import time

from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.client.synchronous.sync_client import LaminarClient

logger = logging.getLogger(__name__)


def with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def with_tracer_and_client_wrapper(func):
    """Helper for providing tracer and client for wrapper functions."""

    def _with_tracer_and_client(
        tracer, client: LaminarClient | AsyncLaminarClient, to_wrap
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, client, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer_and_client


def retry_sync(func, retries=5, delay=0.5, error_message="Operation failed"):
    """Utility function for retry logic in synchronous operations"""
    for attempt in range(retries):
        try:
            result = func()
            if result:  # If function returns truthy value, consider it successful
                return result
            if attempt == retries - 1:  # Last attempt
                logger.debug(f"{error_message} after all retries")
                return None
        except Exception as e:
            if attempt == retries - 1:  # Last attempt
                logger.error(f"{error_message}: {e}")
                return None
        time.sleep(delay)
    return None


async def retry_async(func, retries=5, delay=0.5, error_message="Operation failed"):
    """Utility function for retry logic in asynchronous operations"""
    for attempt in range(retries):
        try:
            result = await func()
            if result:  # If function returns truthy value, consider it successful
                return result
            if attempt == retries - 1:  # Last attempt
                logger.error(f"{error_message} after all retries")
                return None
        except Exception as e:
            if attempt == retries - 1:  # Last attempt
                logger.error(f"{error_message}: {e}")
                return None
        await asyncio.sleep(delay)
    return None

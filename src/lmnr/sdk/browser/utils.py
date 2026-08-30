import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_sync(
    func: Callable[[], T],
    retries: int = 5,
    delay: float = 0.5,
    error_message: str = "Operation failed"
) -> T | None:
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
                logger.debug(f"{error_message}: {e}")
                return None
        time.sleep(delay)
    return None


async def retry_async(
    func: Callable[[], Awaitable[T]],
    retries: int = 5,
    delay: float = 0.5,
    error_message: str = "Operation failed"
) -> T | None:
    """Utility function for retry logic in asynchronous operations"""
    for attempt in range(retries):
        try:
            result = await func()
            if result:  # If function returns truthy value, consider it successful
                return result
            if attempt == retries - 1:  # Last attempt
                logger.debug(f"{error_message} after all retries")
                return None
        except Exception as e:
            if attempt == retries - 1:  # Last attempt
                logger.debug(f"{error_message}: {e}")
                return None
        await asyncio.sleep(delay)
    return None

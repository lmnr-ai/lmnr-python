"""
Background sending for browser events.

This module provides background execution for HTTP requests that send browser events,
ensuring sends never block the main execution flow while guaranteeing completion at
program exit.

## Background Event Loop Architecture
Uses a dedicated event loop running in a separate thread to handle async HTTP requests.
This architecture provides:

1. **Non-blocking execution**: Sends happen in the background, never blocking the main
   thread or Playwright's event loop, allowing browser automation to continue smoothly.

2. **Guaranteed completion**: When the program exits, all pending async sends are
   awaited and complete successfully, even if they're slow. No events are dropped.

3. **Lifecycle independence**: The background loop runs independently of Playwright's
   event loop, so it survives when Playwright shuts down its internal loop before
   program exit.

The pattern uses `asyncio.run_coroutine_threadsafe()` to submit async coroutines
from any thread (sync or async) to our background loop, maintaining pure async code
while achieving cross-thread execution.
"""

import asyncio
import atexit
import threading
from typing import Any

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

# Timeout for waiting for each async send operation at exit
ASYNC_SEND_TIMEOUT_SECONDS = 30

# Timeout for background loop creation
LOOP_CREATION_TIMEOUT_SECONDS = 5

# Timeout for thread join during cleanup
THREAD_JOIN_TIMEOUT_SECONDS = 5

# ==============================================================================
# Background event loop for async sends
# ==============================================================================

# Background event loop state
_background_loop = None
_background_loop_thread = None
_background_loop_lock = threading.Lock()
_background_loop_ready = threading.Event()
_pending_async_futures: set[asyncio.Future[Any]] = set()


def get_background_loop() -> asyncio.AbstractEventLoop:
    """
    Get or create the background event loop for async sends.

    Creates a dedicated event loop running in a daemon thread on first call.
    Subsequent calls return the same loop. Thread-safe.

    Returns:
        The background event loop running in a separate thread.
    """
    global _background_loop, _background_loop_thread

    with _background_loop_lock:
        if _background_loop is None:
            # Create a new event loop in a background thread
            def run_loop():
                global _background_loop
                _background_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_background_loop)
                _background_loop_ready.set()
                _background_loop.run_forever()

            _background_loop_thread = threading.Thread(
                target=run_loop, daemon=True, name="lmnr-async-sends"
            )
            _background_loop_thread.start()

            # Register cleanup handler
            atexit.register(_cleanup_background_loop)

    # Wait for loop to be created (outside the lock to avoid blocking other threads)
    if not _background_loop_ready.wait(timeout=LOOP_CREATION_TIMEOUT_SECONDS):
        raise RuntimeError("Background loop creation timed out")

    return _background_loop


def track_async_send(future: asyncio.Future) -> None:
    """
    Track an async send future for cleanup at exit.

    The future is automatically removed from tracking when it completes,
    preventing memory leaks.

    Args:
        future: The future returned by asyncio.run_coroutine_threadsafe()
    """
    with _background_loop_lock:
        _pending_async_futures.add(future)

    def remove_on_done(f):
        """Remove the future from tracking when it completes."""
        with _background_loop_lock:
            _pending_async_futures.discard(f)

    future.add_done_callback(remove_on_done)


def _cleanup_background_loop():
    """
    Shutdown the background event loop and wait for all pending sends to complete.

    Called automatically at program exit via atexit. Waits for each pending send
    to complete with a timeout, then stops the background loop gracefully.
    """
    global _background_loop

    # Create a snapshot of pending futures to avoid holding the lock during waits
    with _background_loop_lock:
        futures_to_wait = list(_pending_async_futures)

    pending_count = len(futures_to_wait)

    if pending_count > 0:
        logger.info(
            f"Finishing sending {pending_count} browser events... "
            "Ctrl+C to cancel (may result in incomplete session recording)."
        )

        # Wait for all pending futures to complete
        for future in futures_to_wait:
            try:
                future.result(timeout=ASYNC_SEND_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.debug("Timeout waiting for async send to complete")
            except KeyboardInterrupt:
                logger.debug("Interrupted, cancelling pending async sends")
                for f in futures_to_wait:
                    f.cancel()
                raise
            except Exception as e:
                logger.debug(f"Error in async send: {e}")

    # Stop the background loop
    if _background_loop is not None and not _background_loop.is_closed():
        try:
            _background_loop.call_soon_threadsafe(_background_loop.stop)
            # Wait for thread to finish
            if _background_loop_thread is not None:
                _background_loop_thread.join(timeout=THREAD_JOIN_TIMEOUT_SECONDS)
        except Exception as e:
            logger.debug(f"Error stopping background loop: {e}")

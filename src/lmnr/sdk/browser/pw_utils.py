from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, Protocol, cast

import orjson
from opentelemetry import trace

from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.opentelemetry_lib.tracing.context import get_current_context
from lmnr.opentelemetry_lib.utils.package_check import is_package_installed
from lmnr.sdk.browser.background_send_events import (
    get_background_loop,
    track_async_send,
)
from lmnr.sdk.browser.chunk_types import ChunkBuffer, ChunkMessage
from lmnr.sdk.browser.utils import retry_async, retry_sync
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.decorators import observe
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import MaskInputOptions

if TYPE_CHECKING:
    from playwright.async_api import Page
    from playwright.sync_api import Page as SyncPage

if not (is_package_installed("playwright") or is_package_installed("patchright")):
    raise ImportError(
        "Attempted to import lmnr.sdk.browser.pw_utils, but neither "
        + "playwright nor patchright is installed. Use `pip install playwright` "
        + "or `pip install patchright` to install one of the supported browsers."
    )

logger = get_default_logger(__name__)

OLD_BUFFER_TIMEOUT = 60


class _ExposesSyncEvents(Protocol):
    """The slice of playwright's sync `Page`/`BrowserContext` API used to wire
    up `lmnrSendEvents`, re-declared with a precise callback type.

    Playwright's own stub declares `expose_function`'s `callback` as a bare
    `typing.Callable` — no parameter/return types — which pyright renders as
    `(...) -> Unknown` and flags on every call, regardless of what we pass.
    Casting to this Protocol at the call site swaps in a signature we own.
    """

    def expose_function(
        self, name: str, callback: Callable[[ChunkMessage], None]
    ) -> None: ...


class _ExposesAsyncEvents(Protocol):
    """Async counterpart of `_ExposesSyncEvents`, see there for why this exists."""

    async def expose_function(
        self,
        name: str,
        callback: Callable[[ChunkMessage], Coroutine[Any, Any, None]],  # pyright: ignore[reportExplicitAny]
    ) -> None: ...


def create_send_events_handler(
    chunk_buffers: dict[str, ChunkBuffer],
    session_id: str,
    trace_id: str,
    client: AsyncLaminarClient,
    background_loop: asyncio.AbstractEventLoop,
) -> Callable[[ChunkMessage], Coroutine[Any, Any, None]]:  # pyright: ignore[reportExplicitAny]
    """
    Create an async event handler for sending browser events.

    This handler reassembles chunked event data and submits it to the background
    loop for async HTTP sending. The handler itself processes chunks synchronously
    but delegates the actual HTTP send to the background loop.

    Args:
        chunk_buffers: Dictionary to store incomplete chunk batches
        session_id: Browser session ID
        trace_id: OpenTelemetry trace ID
        client: Async Laminar client for HTTP requests
        background_loop: Background event loop for async sends

    Returns:
        An async function that handles incoming event chunks from the browser
    """

    async def send_events_from_browser(chunk: ChunkMessage) -> None:
        try:
            # Handle chunked data
            batch_id = chunk["batchId"]
            chunk_index = chunk["chunkIndex"]
            total_chunks = chunk["totalChunks"]
            data = chunk["data"]

            # Initialize buffer for this batch if needed
            if batch_id not in chunk_buffers:
                chunk_buffers[batch_id] = ChunkBuffer(
                    chunks={},
                    total=total_chunks,
                    timestamp=time.time(),
                )

            # Store chunk
            chunk_buffers[batch_id]["chunks"][chunk_index] = data

            # Check if we have all chunks
            if len(chunk_buffers[batch_id]["chunks"]) == total_chunks:
                # Reassemble the full message
                full_data = ""
                for i in range(total_chunks):
                    full_data += chunk_buffers[batch_id]["chunks"][i]

                # Parse the JSON. The individual event shape is rrweb's, not
                # ours, so `Any` here is a genuine external contract.
                events = cast(list[dict[str, Any]], orjson.loads(full_data))  # pyright: ignore[reportExplicitAny]

                # Send to server in background loop (independent of Playwright's loop)
                if events and len(events) > 0:
                    future = asyncio.run_coroutine_threadsafe(
                        # Internal client API, shared across the browser
                        # instrumentation package.
                        client._browser_events.send(  # pyright: ignore[reportPrivateUsage]
                            session_id, trace_id, events
                        ),
                        background_loop,
                    )
                    track_async_send(future)

                # Clean up buffer
                del chunk_buffers[batch_id]

            # Clean up old incomplete buffers
            current_time = time.time()
            to_delete: list[str] = []
            for bid, buffer in chunk_buffers.items():
                if current_time - buffer["timestamp"] > OLD_BUFFER_TIMEOUT:
                    to_delete.append(bid)
            for bid in to_delete:
                logger.debug(f"Cleaning up incomplete chunk buffer: {bid}")
                del chunk_buffers[bid]

        except Exception as e:
            logger.debug(f"Could not send events: {e}")

    return send_events_from_browser


current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "recorder", "record.umd.min.cjs"), "r") as f:
    RRWEB_CONTENT = f"() => {{ {f.read()} }}"

with open(os.path.join(current_dir, "inject_script.js"), "r") as f:
    INJECT_SCRIPT_CONTENT = f.read()


def get_mask_input_setting() -> MaskInputOptions:
    """Get the mask_input setting from session recording configuration."""
    try:
        config = TracerWrapper.get_session_recording_options()
        return config.get("mask_input_options") or MaskInputOptions(
            textarea=False,
            text=False,
            number=False,
            select=False,
            email=False,
            tel=False,
        )
    except (AttributeError, Exception):
        # Fallback to default configuration if TracerWrapper is not initialized
        return MaskInputOptions(
            textarea=False,
            text=False,
            number=False,
            select=False,
            email=False,
            tel=False,
        )


def inject_session_recorder_sync(page: SyncPage):
    try:
        try:
            is_loaded = cast(bool, page.evaluate(
                """() => typeof window.lmnrRrweb !== 'undefined'"""
            ))
        except Exception as e:
            logger.debug(f"Failed to check if session recorder is loaded: {e}")
            is_loaded = False

        if not is_loaded:

            def load_session_recorder():
                try:
                    if page.is_closed():
                        return False
                    page.evaluate(RRWEB_CONTENT)
                    return True
                except Exception as e:
                    logger.debug(f"Failed to load session recorder: {e}")
                    return False

            if not retry_sync(
                load_session_recorder,
                delay=1,
                error_message="Failed to load session recorder",
            ):
                return

            try:
                if not page.is_closed():
                    page.evaluate(
                        f"({INJECT_SCRIPT_CONTENT})({orjson.dumps(get_mask_input_setting()).decode('utf-8')}, false)"
                    )
            except Exception as e:
                logger.debug(f"Failed to inject session recorder: {e}")

    except Exception as e:
        logger.debug(f"Error during session recorder injection: {e}")


async def inject_session_recorder_async(page: Page):
    try:
        try:
            is_loaded = cast(bool, await page.evaluate(
                """() => typeof window.lmnrRrweb !== 'undefined'"""
            ))
        except Exception as e:
            logger.debug(f"Failed to check if session recorder is loaded: {e}")
            is_loaded = False

        if not is_loaded:

            async def load_session_recorder():
                try:
                    if page.is_closed():
                        return False
                    await page.evaluate(RRWEB_CONTENT)
                    return True
                except Exception as e:
                    logger.debug(f"Failed to load session recorder: {e}")
                    return False

            if not await retry_async(
                load_session_recorder,
                delay=1,
                error_message="Failed to load session recorder",
            ):
                return

            try:
                if not page.is_closed():
                    await page.evaluate(
                        f"({INJECT_SCRIPT_CONTENT})({orjson.dumps(get_mask_input_setting()).decode('utf-8')}, false)"
                    )
            except Exception as e:
                logger.debug(f"Failed to inject session recorder placeholder: {e}")

    except Exception as e:
        logger.debug(f"Error during session recorder injection: {e}")


@observe(name="playwright.page", ignore_input=True, ignore_output=True)
def start_recording_events_sync(
    page: SyncPage, session_id: str, client: AsyncLaminarClient
):

    ctx = get_current_context()
    span = trace.get_current_span(ctx)
    trace_id = format(span.get_span_context().trace_id, "032x")
    span.set_attribute("lmnr.internal.has_browser_session", True)

    # Get the background loop for async sends
    background_loop = get_background_loop()

    # Buffer for reassembling chunks
    chunk_buffers: dict[str, ChunkBuffer] = {}

    # Create the async event handler (shared implementation)
    send_events_from_browser = create_send_events_handler(
        chunk_buffers, session_id, trace_id, client, background_loop
    )

    def submit_event(chunk: ChunkMessage) -> None:
        """Sync wrapper that submits async handler to background loop."""
        try:
            # Submit async handler to background loop
            _future = asyncio.run_coroutine_threadsafe(
                send_events_from_browser(chunk),
                background_loop,
            )
        except Exception as e:
            logger.debug(f"Error submitting event: {e}")

    try:
        cast(_ExposesSyncEvents, page).expose_function("lmnrSendEvents", submit_event)
    except Exception as e:
        logger.debug(f"Could not expose function: {e}")

    inject_session_recorder_sync(page)

    def on_load(p: SyncPage) -> None:
        try:
            if not p.is_closed():
                inject_session_recorder_sync(p)
        except Exception as e:
            logger.debug(f"Error in on_load handler: {e}")

    page.on("domcontentloaded", on_load)


@observe(name="playwright.page", ignore_input=True, ignore_output=True)
async def start_recording_events_async(
    page: Page, session_id: str, client: AsyncLaminarClient
):
    ctx = get_current_context()
    span = trace.get_current_span(ctx)
    trace_id = format(span.get_span_context().trace_id, "032x")
    span.set_attribute("lmnr.internal.has_browser_session", True)

    # Get the background loop for async sends (independent of Playwright's loop)
    background_loop = get_background_loop()

    # Buffer for reassembling chunks
    chunk_buffers: dict[str, ChunkBuffer] = {}

    # Create the async event handler (shared implementation)
    send_events_from_browser = create_send_events_handler(
        chunk_buffers, session_id, trace_id, client, background_loop
    )

    try:
        await cast(_ExposesAsyncEvents, page).expose_function(
            "lmnrSendEvents", send_events_from_browser
        )
    except Exception as e:
        logger.debug(f"Could not expose function: {e}")

    await inject_session_recorder_async(page)

    async def on_load(p: Page) -> None:
        try:
            # Check if page is closed before attempting to inject
            if not p.is_closed():
                await inject_session_recorder_async(p)
        except Exception as e:
            logger.debug(f"Error in on_load handler: {e}")

    page.on("domcontentloaded", on_load)


def take_full_snapshot(page: SyncPage) -> bool:
    return cast(bool, page.evaluate(
        """() => {
        if (window.lmnrRrweb) {
            try {
                window.lmnrRrweb.record.takeFullSnapshot();
                return true;
            } catch (e) {
                console.error("Error taking full snapshot:", e);
                return false;
            }
        }
        return false;
    }"""
    ))


async def take_full_snapshot_async(page: Page) -> bool:
    return cast(bool, await page.evaluate(
        """() => {
        if (window.lmnrRrweb) {
            try {
                window.lmnrRrweb.record.takeFullSnapshot();
                return true;
            } catch (e) {
                console.error("Error taking full snapshot:", e);
                return false;
            }
        }
        return false;
    }"""
    ))

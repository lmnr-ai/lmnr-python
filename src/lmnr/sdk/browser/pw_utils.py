import asyncio
import logging
import os
import time
import threading

from opentelemetry import trace

from lmnr.sdk.decorators import observe
from lmnr.sdk.browser.utils import retry_sync, retry_async
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient

try:
    from playwright.async_api import Page
    from playwright.sync_api import Page as SyncPage
except ImportError as e:
    raise ImportError(
        f"Attempted to import {__file__}, but it is designed "
        "to patch Playwright, which is not installed. Use `pip install playwright` "
        "to install Playwright or remove this import."
    ) from e

logger = logging.getLogger(__name__)

# Track pages we've already instrumented to avoid double-instrumentation
instrumented_pages = set()
async_instrumented_pages = set()


current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "rrweb", "rrweb.min.js"), "r") as f:
    RRWEB_CONTENT = f"() => {{ {f.read()} }}"

INJECT_PLACEHOLDER = """
() => {
    const BATCH_SIZE = 1000;  // Maximum events to store in memory
    
    window.lmnrRrwebEventsBatch = new Set();
    
    // Utility function to compress individual event data
    async function compressEventData(data) {
        const jsonString = JSON.stringify(data);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const compressedStream = blob.stream().pipeThrough(new CompressionStream('gzip'));
        const compressedResponse = new Response(compressedStream);
        const compressedData = await compressedResponse.arrayBuffer();
        return Array.from(new Uint8Array(compressedData));
    }
    
    window.lmnrGetAndClearEvents = () => {
        const events = window.lmnrRrwebEventsBatch;
        window.lmnrRrwebEventsBatch = new Set();
        return Array.from(events);
    };

    // Add heartbeat events
    setInterval(async () => {
        const heartbeat = {
            type: 6,
            data: await compressEventData({ source: 'heartbeat' }),
            timestamp: Date.now()
        };
        
        window.lmnrRrwebEventsBatch.add(heartbeat);
        
        // Prevent memory issues by limiting batch size
        if (window.lmnrRrwebEventsBatch.size > BATCH_SIZE) {
            window.lmnrRrwebEventsBatch = new Set(Array.from(window.lmnrRrwebEventsBatch).slice(-BATCH_SIZE));
        }
    }, 1000);

    window.lmnrRrweb.record({
        async emit(event) {
            // Ignore events from all tabs except the current one
            if (document.visibilityState === 'hidden' || document.hidden) {
                return;
            }
            // Compress the data field
            const compressedEvent = {
                ...event,
                data: await compressEventData(event.data)
            };
            window.lmnrRrwebEventsBatch.add(compressedEvent);
        }
    });
}
"""


async def send_events_async(
    page: Page, session_id: str, trace_id: str, client: AsyncLaminarClient
):
    """Fetch events from the page and send them to the server"""
    try:
        # Check if function exists first
        has_function = await page.evaluate(
            """
            () => typeof window.lmnrGetAndClearEvents === 'function'
        """
        )
        if not has_function:
            return

        events = await page.evaluate("window.lmnrGetAndClearEvents()")
        if not events or len(events) == 0:
            return

        await client._browser_events.send(session_id, trace_id, events)

    except Exception as e:
        logger.error(f"Error sending events: {e}")


def send_events_sync(
    page: SyncPage, session_id: str, trace_id: str, client: LaminarClient
):
    """Synchronous version of send_events"""
    try:
        # Check if function exists first
        has_function = page.evaluate(
            """
            () => typeof window.lmnrGetAndClearEvents === 'function'
        """
        )
        if not has_function:
            return

        events = page.evaluate("window.lmnrGetAndClearEvents()")
        if not events or len(events) == 0:
            return

        client._browser_events.send(session_id, trace_id, events)

    except Exception as e:
        logger.error(f"Error sending events: {e}")


def inject_rrweb_sync(page: SyncPage):
    try:
        page.wait_for_load_state("domcontentloaded")

        # Wrap the evaluate call in a try-catch
        try:
            is_loaded = page.evaluate(
                """() => typeof window.lmnrRrweb !== 'undefined'"""
            )
        except Exception as e:
            logger.debug(f"Failed to check if rrweb is loaded: {e}")
            is_loaded = False

        if not is_loaded:

            def load_rrweb():
                try:
                    page.evaluate(RRWEB_CONTENT)
                    page.wait_for_function(
                        """(() => typeof window.lmnrRrweb !== 'undefined')""",
                        timeout=5000,
                    )
                    return True
                except Exception as e:
                    logger.debug(f"Failed to load rrweb: {e}")
                    return False

            if not retry_sync(
                load_rrweb, delay=1, error_message="Failed to load rrweb"
            ):
                return

        try:
            page.evaluate(INJECT_PLACEHOLDER)
        except Exception as e:
            logger.debug(f"Failed to inject rrweb placeholder: {e}")

    except Exception as e:
        logger.error(f"Error during rrweb injection: {e}")


async def inject_rrweb_async(page: Page):
    try:
        await page.wait_for_load_state("domcontentloaded")

        # Wrap the evaluate call in a try-catch
        try:
            is_loaded = await page.evaluate(
                """() => typeof window.lmnrRrweb !== 'undefined'"""
            )
        except Exception as e:
            logger.debug(f"Failed to check if rrweb is loaded: {e}")
            is_loaded = False

        if not is_loaded:

            async def load_rrweb():
                try:
                    await page.evaluate(RRWEB_CONTENT)
                    await page.wait_for_function(
                        """(() => typeof window.lmnrRrweb !== 'undefined')""",
                        timeout=5000,
                    )
                    return True
                except Exception as e:
                    logger.debug(f"Failed to load rrweb: {e}")
                    return False

            if not await retry_async(
                load_rrweb, delay=1, error_message="Failed to load rrweb"
            ):
                return

        try:
            await page.evaluate(INJECT_PLACEHOLDER)
        except Exception as e:
            logger.debug(f"Failed to inject rrweb placeholder: {e}")

    except Exception as e:
        logger.error(f"Error during rrweb injection: {e}")


@observe(name="playwright.page", ignore_input=True, ignore_output=True)
def handle_navigation_sync(
    page: SyncPage, session_id: str, trace_id: str, client: LaminarClient
):
    trace.get_current_span().set_attribute("lmnr.internal.has_browser_session", True)
    # Check if we've already instrumented this page
    page_id = id(page)
    if page_id in instrumented_pages:
        return
    instrumented_pages.add(page_id)

    def on_load():
        try:
            inject_rrweb_sync(page)
        except Exception as e:
            logger.error(f"Error in on_load handler: {e}")

    page.on("load", on_load)
    inject_rrweb_sync(page)

    def collection_loop():
        while not page.is_closed():  # Stop when page closes
            send_events_sync(page, session_id, trace_id, client)
            time.sleep(2)

        # Clean up when page closes
        if page_id in instrumented_pages:
            instrumented_pages.remove(page_id)

    thread = threading.Thread(target=collection_loop, daemon=True)
    thread.start()


@observe(name="playwright.page", ignore_input=True, ignore_output=True)
async def handle_navigation_async(
    page: Page, session_id: str, trace_id: str, client: AsyncLaminarClient
):
    trace.get_current_span().set_attribute("lmnr.internal.has_browser_session", True)
    # Check if we've already instrumented this page
    page_id = id(page)
    if page_id in async_instrumented_pages:
        return
    async_instrumented_pages.add(page_id)

    async def on_load():
        try:
            await inject_rrweb_async(page)
        except Exception as e:
            logger.error(f"Error in on_load handler: {e}")

    page.on("load", lambda: asyncio.create_task(on_load()))
    await inject_rrweb_async(page)

    async def collection_loop():
        try:
            while not page.is_closed():  # Stop when page closes
                await send_events_async(page, session_id, trace_id, client)
                await asyncio.sleep(2)
            # Clean up when page closes
            async_instrumented_pages.remove(page_id)
            logger.info("Event collection stopped")
        except Exception as e:
            logger.error(f"Event collection stopped: {e}")

    # Create and store task
    task = asyncio.create_task(collection_loop())

    # Clean up task when page closes
    page.on("close", lambda: task.cancel())

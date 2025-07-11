import asyncio
import logging
import os
import threading

from opentelemetry import trace

from lmnr.opentelemetry_lib.utils.package_check import is_package_installed
from lmnr.sdk.decorators import observe
from lmnr.sdk.browser.utils import retry_sync, retry_async
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.opentelemetry_lib.tracing import _get_current_context

try:
    if is_package_installed("playwright"):
        from playwright.async_api import Page
        from playwright.sync_api import Page as SyncPage
    elif is_package_installed("patchright"):
        from patchright.async_api import Page
        from patchright.sync_api import Page as SyncPage
    else:
        raise ImportError(
            "Attempted to import lmnr.sdk.browser.pw_utils, but neither "
            "playwright nor patchright is installed. Use `pip install playwright` "
            "or `pip install patchright` to install one of the supported browsers."
        )
except ImportError as e:
    raise ImportError(
        "Attempted to import lmnr.sdk.browser.pw_utils, but neither "
        "playwright nor patchright is installed. Use `pip install playwright` "
        "or `pip install patchright` to install one of the supported browsers."
    ) from e

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "rrweb", "rrweb.umd.min.cjs"), "r") as f:
    RRWEB_CONTENT = f"() => {{ {f.read()} }}"

INJECT_PLACEHOLDER = """
() => {
    const BATCH_SIZE = 1000;  // Maximum events to store in memory
    const BATCH_TIMEOUT = 2000; // Send events after 2 seconds if batch isn't full

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

    // Function to send events when batch is ready
    async function sendBatchIfReady() {
        if (window.lmnrRrwebEventsBatch.size > 0 && typeof window.lmnrSendEvents === 'function') {
            const events = Array.from(window.lmnrRrwebEventsBatch);
            window.lmnrRrwebEventsBatch = new Set();
            
            try {
                await window.lmnrSendEvents(events);
            } catch (error) {
                console.error('Failed to send events:', error);
            }
        }
    }

    // Set up periodic sending every 2 seconds
    setInterval(sendBatchIfReady, BATCH_TIMEOUT);

    // Add heartbeat events
    setInterval(async () => {
        window.lmnrRrweb.record.addCustomEvent('heartbeat', {
            title: document.title,
            url: document.URL,
        })
    }, 1000);

    window.lmnrRrweb.record({
        async emit(event) {    
            // Compress the data field
            const compressedEvent = {
                ...event,
                data: await compressEventData(event.data)
            };
            window.lmnrRrwebEventsBatch.add(compressedEvent);
            // Send immediately if batch is full
            if (window.lmnrRrwebEventsBatch.size >= BATCH_SIZE) {
                await sendBatchIfReady();
            }
        },
        recordCanvas: true,
        collectFonts: true,
        recordCrossOriginIframes: true
    });
}
"""


async def send_events_async(
    page: Page, session_id: str, trace_id: str, client: AsyncLaminarClient
):
    """Fetch events from the page and send them to the server"""
    try:
        # Check if function exists first
        events = await page.evaluate(
            """
        () => {
            if (typeof window.lmnrGetAndClearEvents !== 'function') {
                return [];
            }
            return window.lmnrGetAndClearEvents();
        }
        """
        )

        if not events or len(events) == 0:
            return

        await client._browser_events.send(session_id, trace_id, events)
    except Exception as e:
        if "Page.evaluate: Target page, context or browser has been closed" not in str(
            e
        ):
            logger.debug(f"Could not send events: {e}")


def send_events_sync(
    page: SyncPage, session_id: str, trace_id: str, client: LaminarClient
):
    """Synchronous version of send_events"""
    try:
        events = page.evaluate(
            """
        () => {
            if (typeof window.lmnrGetAndClearEvents !== 'function') {
                return [];
            }
            return window.lmnrGetAndClearEvents();
        }
        """
        )
        if not events or len(events) == 0:
            return

        client._browser_events.send(session_id, trace_id, events)

    except Exception as e:
        if "Page.evaluate: Target page, context or browser has been closed" not in str(
            e
        ):
            logger.debug(f"Could not send events: {e}")


def inject_session_recorder_sync(page: SyncPage):
    try:
        page.wait_for_load_state("domcontentloaded")

        # Wrap the evaluate call in a try-catch
        try:
            is_loaded = page.evaluate(
                """() => typeof window.lmnrRrweb !== 'undefined'"""
            )
        except Exception as e:
            logger.debug(f"Failed to check if session recorder is loaded: {e}")
            is_loaded = False

        if not is_loaded:

            def load_session_recorder():
                try:
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
            page.evaluate(INJECT_PLACEHOLDER)
        except Exception as e:
            logger.debug(f"Failed to inject session recorder: {e}")

    except Exception as e:
        logger.error(f"Error during session recorder injection: {e}")


async def inject_session_recorder_async(page: Page):
    try:
        await page.wait_for_load_state("domcontentloaded")

        # Wrap the evaluate call in a try-catch
        try:
            is_loaded = await page.evaluate(
                """() => typeof window.lmnrRrweb !== 'undefined'"""
            )
        except Exception as e:
            logger.debug(f"Failed to check if session recorder is loaded: {e}")
            is_loaded = False

        if not is_loaded:

            async def load_session_recorder():
                try:
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
            await page.evaluate(INJECT_PLACEHOLDER)
        except Exception as e:
            logger.debug(f"Failed to inject session recorder placeholder: {e}")

    except Exception as e:
        logger.error(f"Error during session recorder injection: {e}")


@observe(name="playwright.page", ignore_input=True, ignore_output=True)
def handle_navigation_sync(page: SyncPage, session_id: str, client: LaminarClient):
    
    ctx = _get_current_context()
    span = trace.get_current_span(ctx)
    trace_id = format(span.get_span_context().trace_id, "032x")
    span.set_attribute("lmnr.internal.has_browser_session", True)
 
    # Expose function to browser so it can call us when events are ready
    def send_events_from_browser(events):
        try:
            if events and len(events) > 0:
                client._browser_events.send(session_id, trace_id, events)
        except Exception as e:
            logger.debug(f"Could not send events: {e}")

    page.expose_function("lmnrSendEvents", send_events_from_browser)

    def on_load():
        try:
            inject_session_recorder_sync(page)
        except Exception as e:
            logger.error(f"Error in on_load handler: {e}")

    def on_close():
        try:
            send_events_sync(page, session_id, trace_id, client)
        except Exception:
            pass

    page.on("load", on_load)
    page.on("close", on_close)
    inject_session_recorder_sync(page)


@observe(name="playwright.page", ignore_input=True, ignore_output=True)
async def handle_navigation_async(
    page: Page, session_id: str, client: AsyncLaminarClient
):

    ctx = _get_current_context()
    span = trace.get_current_span(ctx)
    trace_id = format(span.get_span_context().trace_id, "032x")
    span.set_attribute("lmnr.internal.has_browser_session", True)

    # Expose function to browser so it can call us when events are ready
    async def send_events_from_browser(events):
        try:
            if events and len(events) > 0:
                await client._browser_events.send(session_id, trace_id, events)
        except Exception as e:
            logger.debug(f"Could not send events: {e}")

    await page.expose_function("lmnrSendEvents", send_events_from_browser)

    async def on_load(p):
        try:
            await inject_session_recorder_async(p)
        except Exception as e:
            logger.error(f"Error in on_load handler: {e}")

    async def on_close(p):
        try:
            # Send any remaining events before closing
            await send_events_async(p, session_id, trace_id, client)
        except Exception:
            pass

    page.on("load", on_load)
    page.on("close", on_close)

    await inject_session_recorder_async(page)

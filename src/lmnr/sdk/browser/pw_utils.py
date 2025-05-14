import asyncio
import logging
import os
import time
import threading

from opentelemetry import trace

from lmnr.opentelemetry_lib.utils.package_check import is_package_installed
from lmnr.sdk.decorators import observe
from lmnr.sdk.browser.utils import retry_sync, retry_async
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient


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

    window.lmnrRrwebEventsBatch = new Set();
    
    // Track page focus state
    window.lmnrPageIsFocused = true;
    
    window.addEventListener('blur', () => {
        window.lmnrPageIsFocused = false;
        console.log('Page lost focus');
    });
    
    window.addEventListener('focus', () => {
        window.lmnrPageIsFocused = true;
        console.log('Page gained focus');
    });

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
        if (!window.lmnrPageIsFocused) {
            return;
        }

        window.lmnrRrweb.record.addCustomEvent('heartbeat', {
            title: document.title,
            url: document.URL,
        })

    }, 1000);

    window.lmnrRrweb.record({
        async emit(event) {
            // Ignore events when page is not focused
            if (!window.lmnrPageIsFocused) {
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
        events = await page.evaluate(
            """
        () => {
            if (!window.lmnrPageIsFocused || typeof window.lmnrGetAndClearEvents !== 'function') {
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
        if str(e).startswith("Page.evaluate: Execution context was destroyed"):
            logger.info("Execution context was destroyed, injecting rrweb again")
            await inject_rrweb_async(page)
            await send_events_async(page, session_id, trace_id, client)
        else:
            logger.debug(f"Could not send events: {e}")


def send_events_sync(
    page: SyncPage, session_id: str, trace_id: str, client: LaminarClient
):
    """Synchronous version of send_events"""
    try:
        events = page.evaluate(
            """
        () => {
            if (!window.lmnrPageIsFocused || typeof window.lmnrGetAndClearEvents !== 'function') {
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
        if str(e).startswith("Page.evaluate: Execution context was destroyed"):
            logger.info("Execution context was destroyed, injecting rrweb again")
            inject_rrweb_sync(page)
            send_events_sync(page, session_id, trace_id, client)
        else:
            logger.debug(f"Could not send events: {e}")


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
    original_bring_to_front = page.bring_to_front

    def bring_to_front():
        original_bring_to_front()
        page.evaluate(
            """() => {
            if (window.lmnrRrweb) {
                try {
                    window.lmnrRrweb.record.takeFullSnapshot();
                } catch (e) {
                    console.error("Error taking full snapshot:", e);
                }
            }
        }"""
        )

    page.bring_to_front = bring_to_front

    def on_load():
        try:
            inject_rrweb_sync(page)
        except Exception as e:
            logger.error(f"Error in on_load handler: {e}")

    def collection_loop():
        while not page.is_closed():  # Stop when page closes
            send_events_sync(page, session_id, trace_id, client)
            time.sleep(2)

    thread = threading.Thread(target=collection_loop, daemon=True)
    thread.start()

    def on_close():
        try:
            send_events_sync(page, session_id, trace_id, client)
            thread.join()
        except Exception:
            pass

    page.on("load", on_load)
    page.on("close", on_close)
    inject_rrweb_sync(page)


@observe(name="playwright.page", ignore_input=True, ignore_output=True)
async def handle_navigation_async(
    page: Page, session_id: str, trace_id: str, client: AsyncLaminarClient
):
    trace.get_current_span().set_attribute("lmnr.internal.has_browser_session", True)

    async def collection_loop():
        try:
            while not page.is_closed():  # Stop when page closes
                await send_events_async(page, session_id, trace_id, client)
                await asyncio.sleep(2)
            logger.info("Event collection stopped")
        except Exception as e:
            logger.error(f"Event collection stopped: {e}")

    # Create and store task
    task = asyncio.create_task(collection_loop())

    async def on_load():
        try:
            await inject_rrweb_async(page)
        except Exception as e:
            logger.error(f"Error in on_load handler: {e}")

    async def on_close():
        try:
            task.cancel()
            await send_events_async(page, session_id, trace_id, client)
        except Exception:
            pass

    page.on("load", lambda: asyncio.create_task(on_load()))
    page.on("close", lambda: asyncio.create_task(on_close()))

    original_bring_to_front = page.bring_to_front

    async def bring_to_front():
        await original_bring_to_front()

        await page.evaluate(
            """() => {
            if (window.lmnrRrweb) {
                try {
                    window.lmnrRrweb.record.takeFullSnapshot();
                } catch (e) {
                    console.error("Error taking full snapshot:", e);
                }
            }
        }"""
        )

    page.bring_to_front = bring_to_front
    await inject_rrweb_async(page)

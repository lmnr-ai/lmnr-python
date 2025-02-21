import uuid
import asyncio
import logging
import time
import os
import aiohttp
import requests
import threading
import gzip
import json
from lmnr.version import SDK_VERSION, PYTHON_VERSION
from lmnr import Laminar

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import BrowserContext, Page
    from playwright.sync_api import (
        BrowserContext as SyncBrowserContext,
        Page as SyncPage,
    )
except ImportError as e:
    raise ImportError(
        f"Attempted to import {__file__}, but it is designed "
        "to patch Playwright, which is not installed. Use `pip install playwright` "
        "to install Playwright or remove this import."
    ) from e

_original_new_page = None
_original_new_page_async = None

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "rrweb", "rrweb.min.js"), "r") as f:
    RRWEB_CONTENT = f"() => {{ {f.read()} }}"

INJECT_PLACEHOLDER = """
() => {
    const BATCH_SIZE = 1000;  // Maximum events to store in memory
    
    window.lmnrRrwebEventsBatch = [];
    
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
        window.lmnrRrwebEventsBatch = [];
        return events;
    };

    // Add heartbeat events
    setInterval(async () => {
        const heartbeat = {
            type: 6,
            data: await compressEventData({ source: 'heartbeat' }),
            timestamp: Date.now()
        };
        
        window.lmnrRrwebEventsBatch.push(heartbeat);
        
        // Prevent memory issues by limiting batch size
        if (window.lmnrRrwebEventsBatch.length > BATCH_SIZE) {
            window.lmnrRrwebEventsBatch = window.lmnrRrwebEventsBatch.slice(-BATCH_SIZE);
        }
    }, 1000);

    window.lmnrRrweb.record({
        async emit(event) {
            // Compress the data field
            const compressedEvent = {
                ...event,
                data: await compressEventData(event.data)
            };
            window.lmnrRrwebEventsBatch.push(compressedEvent);
        }
    });
}
"""


def retry_sync(func, retries=5, delay=0.5, error_message="Operation failed"):
    """Utility function for retry logic in synchronous operations"""
    for attempt in range(retries):
        try:
            result = func()
            if result:  # If function returns truthy value, consider it successful
                return result
            if attempt == retries - 1:  # Last attempt
                logger.error(f"{error_message} after all retries")
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


async def send_events_async(
    page: Page, http_url: str, project_api_key: str, session_id: str, trace_id: str
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

        payload = {
            "sessionId": session_id,
            "traceId": trace_id,
            "events": events,
            "source": f"python@{PYTHON_VERSION}",
            "sdkVersion": SDK_VERSION,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {project_api_key}",
            "Accept": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{http_url}/v1/browser-sessions/events",
                json=payload,
                headers=headers,
            ) as response:
                if not response.ok:
                    logger.error(f"Failed to send events: {response.status}")

    except Exception as e:
        logger.error(f"Error sending events: {e}")


def send_events_sync(
    page: SyncPage, http_url: str, project_api_key: str, session_id: str, trace_id: str
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

        payload = {
            "sessionId": session_id,
            "traceId": trace_id,
            "events": events,
            "source": f"python@{PYTHON_VERSION}",
            "sdkVersion": SDK_VERSION,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {project_api_key}",
            "Accept": "application/json",
            "Content-Encoding": "gzip",  # Add Content-Encoding header
        }

        # Compress the payload
        compressed_payload = gzip.compress(json.dumps(payload).encode("utf-8"))

        response = requests.post(
            f"{http_url}/v1/browser-sessions/events",
            data=compressed_payload,  # Use data instead of json for raw bytes
            headers=headers,
        )
        if not response.ok:
            logger.error(f"Failed to send events: {response.status_code}")

    except Exception as e:
        logger.error(f"Error sending events: {e}")


def init_playwright_tracing(http_url: str, project_api_key: str):

    def inject_rrweb(page: SyncPage):
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

    def handle_navigation(page: SyncPage, session_id: str, trace_id: str):
        def on_load():
            try:
                inject_rrweb(page)
            except Exception as e:
                logger.error(f"Error in on_load handler: {e}")

        page.on("load", on_load)
        inject_rrweb(page)

        def collection_loop():
            while not page.is_closed():  # Stop when page closes
                send_events_sync(page, http_url, project_api_key, session_id, trace_id)
                time.sleep(2)

        thread = threading.Thread(target=collection_loop, daemon=True)
        thread.start()

    async def handle_navigation_async(page: Page, session_id: str, trace_id: str):
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
                    await send_events_async(
                        page, http_url, project_api_key, session_id, trace_id
                    )
                    await asyncio.sleep(2)
                logger.info("Event collection stopped")
            except Exception as e:
                logger.error(f"Event collection stopped: {e}")

        # Create and store task
        task = asyncio.create_task(collection_loop())

        # Clean up task when page closes
        page.on("close", lambda: task.cancel())

    def patched_new_page(self: SyncBrowserContext, *args, **kwargs):
        with Laminar.start_as_current_span(name="browser_context.new_page") as span:
            page = _original_new_page(self, *args, **kwargs)

            session_id = str(uuid.uuid4().hex)
            span.set_attribute("lmnr.internal.has_browser_session", True)

            trace_id = format(span.get_span_context().trace_id, "032x")
            session_id = str(uuid.uuid4().hex)

            handle_navigation(page, session_id, trace_id)
            return page

    async def patched_new_page_async(self: BrowserContext, *args, **kwargs):
        with Laminar.start_as_current_span(name="browser_context.new_page") as span:
            page = await _original_new_page_async(self, *args, **kwargs)

            session_id = str(uuid.uuid4().hex)

            span.set_attribute("lmnr.internal.has_browser_session", True)
            trace_id = format(span.get_span_context().trace_id, "032x")
            session_id = str(uuid.uuid4().hex)
            await handle_navigation_async(page, session_id, trace_id)
            return page

    def patch_browser():
        global _original_new_page, _original_new_page_async
        if _original_new_page_async is None:
            _original_new_page_async = BrowserContext.new_page
            BrowserContext.new_page = patched_new_page_async

        if _original_new_page is None:
            _original_new_page = SyncBrowserContext.new_page
            SyncBrowserContext.new_page = patched_new_page

    patch_browser()

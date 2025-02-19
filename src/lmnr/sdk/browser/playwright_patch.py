import opentelemetry
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
from opentelemetry import trace
from contextlib import contextmanager

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
    
    window.lmnrGetAndClearEvents = () => {
        const events = window.lmnrRrwebEventsBatch;
        window.lmnrRrwebEventsBatch = [];
        return events;
    };

    // Add heartbeat events
    setInterval(() => {
        window.lmnrRrwebEventsBatch.push({
            type: 6,
            data: { source: 'heartbeat' },
            timestamp: Date.now()
        });
        
        // Prevent memory issues by limiting batch size
        if (window.lmnrRrwebEventsBatch.length > BATCH_SIZE) {
            window.lmnrRrwebEventsBatch = window.lmnrRrwebEventsBatch.slice(-BATCH_SIZE);
        }
    }, 1000);

    window.lmnrRrweb.record({
        emit(event) {
            window.lmnrRrwebEventsBatch.push(event);
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


async def send_events(page: Page, http_url: str, project_api_key: str, session_id: str, trace_id: str):
    """Fetch events from the page and send them to the server"""
    try:
        # Check if function exists first
        has_function = await page.evaluate("""
            () => typeof window.lmnrGetAndClearEvents === 'function'
        """)
        if not has_function:
            return

        events = await page.evaluate("window.lmnrGetAndClearEvents()")
        if not events or len(events) == 0:
            return

        payload = {
            "sessionId": session_id,
            "traceId": trace_id,
            "events": events
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {project_api_key}',
            'Accept': 'application/json',
            'Content-Encoding': 'gzip'  # Add Content-Encoding header
        }

        compressed_payload = gzip.compress(json.dumps(payload).encode('utf-8'))

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{http_url}/v1/browser-sessions/events",
                data=compressed_payload,  # Use data instead of json for raw bytes
                headers=headers,
            ) as response:
                if not response.ok:
                    logger.error(f"Failed to send events: {response.status}")
                else:
                    logger.info(f"Sent {len(events)} events", events)

    except Exception as e:
        logger.error(f"Error sending events: {e}")


def send_events_sync(page: SyncPage, http_url: str, project_api_key: str, session_id: str, trace_id: str):
    """Synchronous version of send_events"""
    try:
        # Check if function exists first
        has_function = page.evaluate("""
            () => typeof window.lmnrGetAndClearEvents === 'function'
        """)
        if not has_function:
            return

        events = page.evaluate("window.lmnrGetAndClearEvents()")
        if not events or len(events) == 0:
            return

        payload = {
            "sessionId": session_id,
            "traceId": trace_id,
            "events": events,
            "source": "python",
            "sdkVersion": "0.0.1"
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {project_api_key}',
            'Accept': 'application/json',
            'Content-Encoding': 'gzip'  # Add Content-Encoding header
        }

        # Compress the payload
        compressed_payload = gzip.compress(json.dumps(payload).encode('utf-8'))

        response = requests.post(
            f"{http_url}/v1/browser-sessions/events",
            data=compressed_payload,  # Use data instead of json for raw bytes
            headers=headers
        )
        if not response.ok:
            logger.error(f"Failed to send events: {response.status_code}")

    except Exception as e:
        logger.error(f"Error sending events: {e}")


def init_playwright_tracing(http_url: str, project_api_key: str):

    def inject_rrweb(page: SyncPage):
        page.wait_for_load_state("domcontentloaded")

        is_loaded = page.evaluate(
            """() => typeof window.lmnrRrweb !== 'undefined'"""
        )

        if not is_loaded:
            def load_rrweb():
                page.evaluate(RRWEB_CONTENT)
                page.wait_for_function(
                    """(() => typeof window.lmnrRrweb !== 'undefined')""",
                    timeout=5000,
                )
                return True

            if not retry_sync(
                load_rrweb, delay=1, error_message="Failed to load rrweb"
            ):
                return

        page.evaluate(INJECT_PLACEHOLDER)

    async def inject_rrweb_async(page: Page):
        await page.wait_for_load_state("domcontentloaded")

        is_loaded = await page.evaluate(
            """() => typeof window.lmnrRrweb !== 'undefined'"""
        )

        if not is_loaded:
            async def load_rrweb():
                await page.evaluate(RRWEB_CONTENT)
                await page.wait_for_function(
                    """(() => typeof window.lmnrRrweb !== 'undefined')""",
                    timeout=5000,
                )
                return True

            if not await retry_async(
                load_rrweb, delay=1, error_message="Failed to load rrweb"
            ):
                return

        await page.evaluate(INJECT_PLACEHOLDER)

    def handle_navigation(page: SyncPage, session_id: str, trace_id: str):
        def on_load():
            inject_rrweb(page)

        page.on("load", on_load)
        inject_rrweb(page)
        
        def collection_loop():
            while not page.is_closed():  # Stop when page closes
                send_events_sync(page, http_url, project_api_key, session_id, trace_id)
                time.sleep(1)
        
        thread = threading.Thread(target=collection_loop, daemon=True)
        thread.start()

    async def handle_navigation_async(page: Page, session_id: str, trace_id: str):
        async def on_load():
            await inject_rrweb_async(page)

        page.on("load", lambda: asyncio.create_task(on_load()))
        await inject_rrweb_async(page)
        
        async def collection_loop():
            print("started collection_loop")
            try:
                while not page.is_closed():  # Stop when page closes
                    await send_events(page, http_url, project_api_key, session_id, trace_id)
                    await asyncio.sleep(1)
                logger.info("Event collection stopped")
            except Exception as e:
                logger.error(f"Event collection stopped: {e}")
        
        # Create and store task
        task = asyncio.create_task(collection_loop())
        
        # Clean up task when page closes
        page.on("close", lambda: task.cancel())

    def patched_new_page(self: SyncBrowserContext, *args, **kwargs):
        # with browser_session_span() as (session_id, trace_id):
        page = _original_new_page(self, *args, **kwargs)
        
        session_id = str(uuid.uuid4().hex)
        current_span = opentelemetry.trace.get_current_span()
        if current_span.is_recording():
            current_span.set_attribute("lmnr.internal.has_browser_session", True)

        trace_id = format(current_span.get_span_context().trace_id, "032x")
        session_id = str(uuid.uuid4().hex)

        handle_navigation(page, session_id, trace_id)
        return page

    async def patched_new_page_async(self: BrowserContext, *args, **kwargs):
        # with browser_session_span() as (session_id, trace_id):

        page = await _original_new_page_async(self, *args, **kwargs)
        
        session_id = str(uuid.uuid4().hex)
        current_span = opentelemetry.trace.get_current_span()
        if current_span.is_recording():
            current_span.set_attribute("lmnr.internal.has_browser_session", True)

        trace_id = format(current_span.get_span_context().trace_id, "032x")
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

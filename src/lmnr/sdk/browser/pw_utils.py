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

# Track active pages and sessions for cleanup
_active_sessions = {}  # page_id -> session_info
_active_threads = {}   # page_id -> thread/task info

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "rrweb", "rrweb.umd.min.cjs"), "r") as f:
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
    
    # Track this page and session
    page_id = str(id(page))
    _active_sessions[page_id] = {
        'session_id': session_id, 
        'trace_id': trace_id,
        'client': client,
        'page': page
    }
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
            inject_session_recorder_sync(page)
        except Exception as e:
            logger.error(f"Error in on_load handler: {e}")

    stop_event = threading.Event()
    
    def collection_loop():
        while not page.is_closed():  # Stop when page closes
            send_events_sync(page, session_id, trace_id, client)
            time.sleep(2)

    thread = threading.Thread(target=collection_loop, daemon=True)
    thread.start()
    
    # Track the thread for cleanup
    _active_threads[page_id] = {
        'thread': thread,
        'stop_event': stop_event,
        'type': 'sync'
    }

    def on_close():
        try:
            # Signal the thread to stop
            stop_event.set()
            # Send final events
            send_events_sync(page, session_id, trace_id, client)
            # Wait for thread to finish with timeout
            thread.join(timeout=5)
            if thread.is_alive():
                logger.warning("Collection thread did not terminate within timeout")
        except Exception as e:
            logger.debug(f"Error in on_close handler: {e}")
            # Still try to stop the thread
            stop_event.set()
        finally:
            # Clean up tracking
            _active_sessions.pop(page_id, None)
            _active_threads.pop(page_id, None)

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
    
    # Track this page and session
    page_id = str(id(page))
    _active_sessions[page_id] = {
        'session_id': session_id, 
        'trace_id': trace_id,
        'client': client,
        'page': page
    }

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
    
    # Track the task for cleanup
    _active_threads[page_id] = {
        'task': task,
        'type': 'async'
    }

    async def on_load():
        try:
            await inject_session_recorder_async(page)
        except Exception as e:
            logger.error(f"Error in on_load handler: {e}")

    async def on_close():
        try:
            # Cancel the task and wait for it to finish
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5)
                except asyncio.TimeoutError:
                    logger.warning("Collection task did not terminate within timeout")
                except asyncio.CancelledError:
                    pass  # Expected when task is cancelled
            
            # Send final events
            await send_events_async(page, session_id, trace_id, client)
        except Exception as e:
            logger.debug(f"Error in on_close handler: {e}")
            # Still try to cancel the task
            if not task.done():
                task.cancel()
        finally:
            # Clean up tracking
            _active_sessions.pop(page_id, None)
            _active_threads.pop(page_id, None)

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
    await inject_session_recorder_async(page)


def cleanup_all_sessions():
    """Clean up all active sessions and threads - for emergency cleanup"""
    logger.info(f"Cleaning up {len(_active_sessions)} active sessions")
    
    # Clean up all active threads/tasks
    for page_id, thread_info in list(_active_threads.items()):
        try:
            if thread_info['type'] == 'sync':
                # Sync thread cleanup
                stop_event = thread_info.get('stop_event')
                thread = thread_info.get('thread')
                if stop_event:
                    stop_event.set()
                if thread and thread.is_alive():
                    thread.join(timeout=2)
                    if thread.is_alive():
                        logger.warning(f"Thread for page {page_id} did not terminate")
            else:
                # Async task cleanup
                task = thread_info.get('task')
                if task and not task.done():
                    task.cancel()
                    logger.debug(f"Cancelled task for page {page_id}")
        except Exception as e:
            logger.debug(f"Error cleaning up thread for page {page_id}: {e}")
    
    # Clear tracking dictionaries
    _active_sessions.clear()
    _active_threads.clear()
    
    logger.info("Session cleanup completed")

import opentelemetry
import uuid
import asyncio
import logging
import time
import os

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
([baseUrl, projectApiKey]) => {
    const serverUrl = `${baseUrl}/v1/browser-sessions/events`;
    const FLUSH_INTERVAL = 1000;
    const HEARTBEAT_INTERVAL = 1000;

    window.lmnrRrwebEventsBatch = [];
    
    window.lmnrSendRrwebEventsBatch = async () => {
        if (window.lmnrRrwebEventsBatch.length === 0) return;
        
        const eventsPayload = {
            sessionId: window.lmnrRrwebSessionId,
            traceId: window.lmnrTraceId,
            events: window.lmnrRrwebEventsBatch
        };
        
        try {
            const jsonString = JSON.stringify(eventsPayload);
            const uint8Array = new TextEncoder().encode(jsonString);
            
            const cs = new CompressionStream('gzip');
            const compressedStream = await new Response(
                new Response(uint8Array).body.pipeThrough(cs)
            ).arrayBuffer();
            
            const compressedArray = new Uint8Array(compressedStream);
            
            const blob = new Blob([compressedArray], { type: 'application/octet-stream' });
            
            const response = await fetch(serverUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Encoding': 'gzip',
                    'Authorization': `Bearer ${projectApiKey}`,
                    'Accept': 'application/json'
                },
                body: blob,
                mode: 'cors',
                credentials: 'omit'
            });
            
            if (!response.ok) {
                console.error(`HTTP error! status: ${response.status}`);
                if (response.status === 0) {
                    console.error('Possible CORS issue - check network tab for details');
                }
            }

            window.lmnrRrwebEventsBatch = [];
        } catch (error) {
            console.error('Failed to send events:', error);
        }
    };

    setInterval(() => window.lmnrSendRrwebEventsBatch(), FLUSH_INTERVAL);

    setInterval(() => {
        window.lmnrRrwebEventsBatch.push({
            type: 6,
            data: { source: 'heartbeat' },
            timestamp: Date.now()
        });
    }, HEARTBEAT_INTERVAL);

    window.lmnrRrweb.record({
        emit(event) {
            window.lmnrRrwebEventsBatch.push(event);
        }
    });

    window.addEventListener('beforeunload', () => {
        window.lmnrSendRrwebEventsBatch();
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


def init_playwright_tracing(http_url: str, project_api_key: str):

    def inject_rrweb(page: SyncPage):
        # Wait for the page to be in a ready state first
        page.wait_for_load_state("domcontentloaded")

        # First check if rrweb is already loaded
        is_loaded = page.evaluate(
            """
            () => typeof window.lmnrRrweb !== 'undefined'
        """
        )

        if not is_loaded:

            def load_rrweb():
                page.evaluate(RRWEB_CONTENT)
                # Verify script loaded successfully
                page.wait_for_function(
                    """(() => typeof window.lmnrRrweb !== 'undefined')""",
                    timeout=5000,
                )
                return True

            if not retry_sync(
                load_rrweb, delay=1, error_message="Failed to load rrweb"
            ):
                return

        # Get current trace ID from active span
        current_span = opentelemetry.trace.get_current_span()
        if current_span.is_recording():
            current_span.set_attribute("lmnr.internal.has_browser_session", True)

        trace_id = format(current_span.get_span_context().trace_id, "032x")
        session_id = str(uuid.uuid4().hex)

        def set_window_vars():
            page.evaluate(
                """([traceId, sessionId]) => {
                window.lmnrRrwebSessionId = sessionId;
                window.lmnrTraceId = traceId;
            }""",
                [trace_id, session_id],
            )
            return page.evaluate(
                """
                () => window.lmnrRrwebSessionId && window.lmnrTraceId
            """
            )

        if not retry_sync(
            set_window_vars, error_message="Failed to set window variables"
        ):
            return

        # Update the recording setup to include trace ID
        page.evaluate(
            INJECT_PLACEHOLDER,
            [http_url, project_api_key],
        )

    async def inject_rrweb_async(page: Page):
        # Wait for the page to be in a ready state first
        await page.wait_for_load_state("domcontentloaded")

        # First check if rrweb is already loaded
        is_loaded = await page.evaluate(
            """
            () => typeof window.lmnrRrweb !== 'undefined'
        """
        )

        if not is_loaded:

            async def load_rrweb():
                await page.evaluate(RRWEB_CONTENT)
                # Verify script loaded successfully
                await page.wait_for_function(
                    """(() => typeof window.lmnrRrweb !== 'undefined')""",
                    timeout=5000,
                )
                return True

            if not await retry_async(
                load_rrweb, delay=1, error_message="Failed to load rrweb"
            ):
                return

        # Get current trace ID from active span
        current_span = opentelemetry.trace.get_current_span()
        if current_span.is_recording():
            current_span.set_attribute("lmnr.internal.has_browser_session", True)

        trace_id = format(current_span.get_span_context().trace_id, "032x")
        session_id = str(uuid.uuid4().hex)

        async def set_window_vars():
            await page.evaluate(
                """([traceId, sessionId]) => {
                window.lmnrRrwebSessionId = sessionId;
                window.lmnrTraceId = traceId;
            }""",
                [trace_id, session_id],
            )
            return await page.evaluate(
                """
                () => window.lmnrRrwebSessionId && window.lmnrTraceId
            """
            )

        if not await retry_async(
            set_window_vars, error_message="Failed to set window variables"
        ):
            return

        # Update the recording setup to include trace ID
        await page.evaluate(
            INJECT_PLACEHOLDER,
            [http_url, project_api_key],
        )

    def handle_navigation(page: SyncPage):
        def on_load():
            inject_rrweb(page)

        page.on("load", on_load)
        inject_rrweb(page)

    async def handle_navigation_async(page: Page):
        async def on_load():
            await inject_rrweb_async(page)

        page.on("load", lambda: asyncio.create_task(on_load()))
        await inject_rrweb_async(page)

    async def patched_new_page_async(self: BrowserContext, *args, **kwargs):
        # Modify CSP to allow required domains
        async def handle_route(route):
            try:
                response = await route.fetch()
                headers = dict(response.headers)

                # Find and modify CSP header
                for header_name in headers:
                    if header_name.lower() == "content-security-policy":
                        csp = headers[header_name]
                        parts = csp.split(";")
                        for i, part in enumerate(parts):
                            if "connect-src" in part:
                                parts[i] = f"{part.strip()} {http_url}"
                        headers[header_name] = ";".join(parts)

                await route.fulfill(response=response, headers=headers)
            except Exception as e:
                logger.debug(f"Error handling route: {e}")
                await route.continue_()

        # Intercept all navigation requests to modify CSP headers
        await self.route("**/*", handle_route)
        page = await _original_new_page_async(self, *args, **kwargs)
        await handle_navigation_async(page)
        return page

    def patched_new_page(self: SyncBrowserContext, *args, **kwargs):
        # Modify CSP to allow required domains
        def handle_route(route):
            try:
                response = route.fetch()
                headers = dict(response.headers)

                # Find and modify CSP header
                for header_name in headers:
                    if header_name.lower() == "content-security-policy":
                        csp = headers[header_name]
                        parts = csp.split(";")
                        for i, part in enumerate(parts):
                            if "connect-src" in part:
                                parts[i] = f"{part.strip()} {http_url}"
                        if not any("connect-src" in part for part in parts):
                            parts.append(f" connect-src 'self' {http_url}")
                        headers[header_name] = ";".join(parts)

                route.fulfill(response=response, headers=headers)
            except Exception as e:
                logger.debug(f"Error handling route: {e}")
                route.continue_()

        # Intercept all navigation requests to modify CSP headers
        self.route("**/*", handle_route)
        page = _original_new_page(self, *args, **kwargs)
        handle_navigation(page)
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

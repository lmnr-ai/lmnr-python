import opentelemetry
import uuid
import asyncio
import logging
import time

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

INJECT_PLACEHOLDER = """
([baseUrl, projectApiKey]) => {
    const serverUrl = `${baseUrl}/v1/browser-sessions/events`;
    const FLUSH_INTERVAL = 1000;
    const HEARTBEAT_INTERVAL = 1000;

    window.rrwebEventsBatch = [];
    
    window.sendBatch = async () => {
        if (window.rrwebEventsBatch.length === 0) return;
        
        const eventsPayload = {
            sessionId: window.rrwebSessionId,
            traceId: window.traceId,
            events: window.rrwebEventsBatch
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
                    'Authorization': `Bearer ${projectApiKey}`
                },
                body: blob,
                credentials: 'omit',
                mode: 'cors',
                cache: 'no-cache',
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            window.rrwebEventsBatch = [];
        } catch (error) {
            console.error('Failed to send events:', error);
        }
    };

    setInterval(() => window.sendBatch(), FLUSH_INTERVAL);

    setInterval(() => {
        window.rrwebEventsBatch.push({
            type: 6,
            data: { source: 'heartbeat' },
            timestamp: Date.now()
        });
    }, HEARTBEAT_INTERVAL);

    window.rrweb.record({
        emit(event) {
            window.rrwebEventsBatch.push(event);            
        }
    });

    window.addEventListener('beforeunload', () => {
        window.sendBatch();
    });
}
"""


def init_playwright_tracing(http_url: str, project_api_key: str):

    def inject_rrweb(page: SyncPage):
        # Wait for the page to be in a ready state first
        page.wait_for_load_state("domcontentloaded")

        # First check if rrweb is already loaded
        is_loaded = page.evaluate(
            """
            () => typeof window.rrweb !== 'undefined'
        """
        )

        if not is_loaded:
            try:
                # Add retry logic for script loading
                retries = 3
                for attempt in range(retries):
                    try:
                        page.add_script_tag(
                            url="https://cdn.jsdelivr.net/npm/rrweb@latest/dist/rrweb.min.js"
                        )
                        # Verify script loaded successfully
                        page.wait_for_function(
                            """(() => typeof window.rrweb !== 'undefined')""",
                            timeout=5000,
                        )
                        break
                    except Exception:
                        if attempt == retries - 1:  # Last attempt
                            raise
                        time.sleep(0.5)  # Wait before retry
            except Exception as script_error:
                logger.error("Failed to load rrweb after all retries: %s", script_error)
                return

        # Get current trace ID from active span
        current_span = opentelemetry.trace.get_current_span()
        if current_span.is_recording():
            current_span.set_attribute("lmnr.internal.has_browser_session", True)

        trace_id = format(current_span.get_span_context().trace_id, "032x")
        session_id = str(uuid.uuid4().hex)

        # Generate UUID session ID and set trace ID
        page.evaluate(
            """([traceId, sessionId]) => {
            window.rrwebSessionId = sessionId;
            window.traceId = traceId;
        }""",
            [trace_id, session_id],
        )

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
            () => typeof window.rrweb !== 'undefined'
        """
        )

        if not is_loaded:
            try:
                # Add retry logic for script loading
                retries = 3
                for attempt in range(retries):
                    try:
                        await page.add_script_tag(
                            url="https://cdn.jsdelivr.net/npm/rrweb@latest/dist/rrweb.min.js"
                        )
                        # Verify script loaded successfully
                        await page.wait_for_function(
                            """(() => typeof window.rrweb !== 'undefined')""",
                            timeout=5000,
                        )
                        break
                    except Exception:
                        if attempt == retries - 1:  # Last attempt
                            raise
                        await asyncio.sleep(0.5)  # Wait before retry
            except Exception as script_error:
                logger.error("Failed to load rrweb after all retries: %s", script_error)
                return
        # Get current trace ID from active span
        current_span = opentelemetry.trace.get_current_span()
        if current_span.is_recording():
            current_span.set_attribute("lmnr.internal.has_browser_session", True)

        trace_id = format(current_span.get_span_context().trace_id, "032x")
        session_id = str(uuid.uuid4().hex)

        # Generate UUID session ID and set trace ID
        await page.evaluate(
            """([traceId, sessionId]) => {
            window.rrwebSessionId = sessionId;
            window.traceId = traceId;
        }""",
            [trace_id, session_id],
        )

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
                            if "script-src" in part:
                                parts[i] = f"{part.strip()} cdn.jsdelivr.net"
                            elif "connect-src" in part:
                                parts[i] = f"{part.strip()} " + http_url
                        if not any("connect-src" in part for part in parts):
                            parts.append(" connect-src 'self' " + http_url)
                        headers[header_name] = ";".join(parts)

                await route.fulfill(response=response, headers=headers)
            except Exception:
                await route.continue_()

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
                            if "script-src" in part:
                                parts[i] = f"{part.strip()} cdn.jsdelivr.net"
                            elif "connect-src" in part:
                                parts[i] = f"{part.strip()} " + http_url
                        if not any("connect-src" in part for part in parts):
                            parts.append(" connect-src 'self' " + http_url)
                        headers[header_name] = ";".join(parts)

                route.fulfill(response=response, headers=headers)
            except Exception:
                # Continue with the original request without modification
                route.continue_()

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

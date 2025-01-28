import opentelemetry
import uuid

try:
    from playwright.async_api import BrowserContext, Page
    from playwright.sync_api import (
        BrowserContext as SyncBrowserContext,
        Page as SyncPage,
    )
except ImportError as e:
    raise ImportError(
        f"Attempated to import {__file__}, but it is designed "
        "to patch Playwright, which is not installed. Use `pip install playwright` "
        "to install Playwright or remove this import."
    ) from e

_original_new_page = None
_original_goto = None
_original_new_page_async = None
_original_goto_async = None

INJECT_PLACEHOLDER = """
([baseUrl, projectApiKey]) => {
    const serverUrl = `${baseUrl}/v1/browser-sessions/events`;
    const BATCH_SIZE = 16;
    const FLUSH_INTERVAL = 1000;
    const HEARTBEAT_INTERVAL = 1000; // 1 second heartbeat

    window.rrwebEventsBatch = [];
    
    window.sendBatch = async () => {
        if (window.rrwebEventsBatch.length === 0) return;
        
        const eventsPayload = {
            sessionId: window.rrwebSessionId,
            traceId: window.traceId,
            events: window.rrwebEventsBatch
        };

        try {
            await fetch(serverUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${projectApiKey}` },
                body: JSON.stringify(eventsPayload),
            });
            window.rrwebEventsBatch = [];
        } catch (error) {
            console.error('Failed to send events:', error);
        }
    };

    setInterval(() => window.sendBatch(), FLUSH_INTERVAL);

    // Add heartbeat event
    setInterval(() => {
        window.rrwebEventsBatch.push({
            type: 6, // Custom event type
            data: { source: 'heartbeat' },
            timestamp: Date.now()
        });
    }, HEARTBEAT_INTERVAL);

    window.rrweb.record({
        emit(event) {
            window.rrwebEventsBatch.push(event);
            
            if (window.rrwebEventsBatch.length >= BATCH_SIZE) {
                window.sendBatch();
            }
        }
    });

    // Simplified beforeunload handler
    window.addEventListener('beforeunload', () => {
        window.sendBatch();
    });
}
"""


def init_playwright_tracing(http_url: str, project_api_key: str):
    def inject_rrweb(page: SyncPage):
        # Get current trace ID from active span
        current_span = opentelemetry.trace.get_current_span()
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

        # Load rrweb and set up recording
        page.add_script_tag(
            url="https://cdn.jsdelivr.net/npm/rrweb@latest/dist/rrweb.min.js"
        )

        # Update the recording setup to include trace ID
        page.evaluate(
            INJECT_PLACEHOLDER,
            [http_url, project_api_key],
        )

    async def inject_rrweb_async(page: Page):
        # Wait for the page to be in a ready state first
        await page.wait_for_load_state("domcontentloaded")

        # Get current trace ID from active span
        current_span = opentelemetry.trace.get_current_span()
        current_span.set_attribute("lmnr.internal.has_browser_session", True)
        trace_id = format(current_span.get_span_context().trace_id, "032x")
        session_id = str(uuid.uuid4().hex)

        # Wait for any existing script load to complete
        await page.wait_for_load_state("networkidle")

        # Generate UUID session ID and set trace ID
        await page.evaluate(
            """([traceId, sessionId]) => {
            window.rrwebSessionId = sessionId;
            window.traceId = traceId;
        }""",
            [trace_id, session_id],
        )

        # Load rrweb and set up recording
        await page.add_script_tag(
            url="https://cdn.jsdelivr.net/npm/rrweb@latest/dist/rrweb.min.js"
        )

        await page.wait_for_function("""(() => window.rrweb || 'rrweb' in window)""")

        # Update the recording setup to include trace ID
        await page.evaluate(
            INJECT_PLACEHOLDER,
            [http_url, project_api_key],
        )

    async def patched_new_page_async(self: BrowserContext, *args, **kwargs):
        # Call the original new_page (returns a Page object)
        page = await _original_new_page_async(self, *args, **kwargs)
        # Inject rrweb automatically after the page is created
        await inject_rrweb_async(page)
        return page

    async def patched_goto_async(self: Page, *args, **kwargs):
        # Call the original goto
        result = await _original_goto_async(self, *args, **kwargs)
        # Inject rrweb after navigation
        await inject_rrweb_async(self)
        return result

    def patched_new_page(self: SyncBrowserContext, *args, **kwargs):
        # Call the original new_page (returns a Page object)
        page = _original_new_page(self, *args, **kwargs)
        # Inject rrweb automatically after the page is created
        inject_rrweb(page)
        return page

    def patched_goto(self: SyncPage, *args, **kwargs):
        # Call the original goto
        result = _original_goto(self, *args, **kwargs)
        # Inject rrweb after navigation
        inject_rrweb(self)
        return result

    def patch_browser():
        """
        Overrides BrowserContext.new_page with a patched async function
        that injects rrweb into every new page.
        """
        global _original_new_page, _original_goto, _original_new_page_async, _original_goto_async
        if _original_new_page_async is None or _original_goto_async is None:
            _original_new_page_async = BrowserContext.new_page
            BrowserContext.new_page = patched_new_page_async

            _original_goto_async = Page.goto
            Page.goto = patched_goto_async

        if _original_new_page is None or _original_goto is None:
            _original_new_page = SyncBrowserContext.new_page
            SyncBrowserContext.new_page = patched_new_page

            _original_goto = SyncPage.goto
            SyncPage.goto = patched_goto

    patch_browser()

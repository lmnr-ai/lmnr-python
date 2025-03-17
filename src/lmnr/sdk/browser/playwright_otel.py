import asyncio
import logging
import os
import threading
import time
import uuid

from lmnr.sdk.browser.utils import (
    INJECT_PLACEHOLDER,
    _with_tracer_wrapper,
    retry_sync,
    retry_async,
)
from lmnr.sdk.client import LaminarClient
from lmnr.version import PYTHON_VERSION, SDK_VERSION

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer, Tracer, get_current_span
from typing import Collection
from wrapt import wrap_function_wrapper

try:
    from playwright.async_api import Page
    from playwright.sync_api import Page as SyncPage
except ImportError as e:
    raise ImportError(
        f"Attempted to import {__file__}, but it is designed "
        "to patch Playwright, which is not installed. Use `pip install playwright` "
        "to install Playwright or remove this import."
    ) from e

# all available versions at https://pypi.org/project/playwright/#history
_instruments = ("playwright >= 1.9.0",)
logger = logging.getLogger(__name__)

WRAPPED_METHODS = [
    {
        "package": "playwright.sync_api",
        "object": "BrowserContext",
        "method": "new_page",
    }
]

WRAPPED_METHODS_ASYNC = [
    {
        "package": "playwright.async_api",
        "object": "BrowserContext",
        "method": "new_page",
    }
]

_original_new_page = None
_original_new_page_async = None

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "rrweb", "rrweb.min.js"), "r") as f:
    RRWEB_CONTENT = f"() => {{ {f.read()} }}"


async def send_events_async(page: Page, session_id: str, trace_id: str):
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

        await LaminarClient.send_browser_events(
            session_id, trace_id, events, f"python@{PYTHON_VERSION}"
        )

    except Exception as e:
        logger.error(f"Error sending events: {e}")


def send_events_sync(page: SyncPage, session_id: str, trace_id: str):
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

        LaminarClient.send_browser_events_sync(
            session_id, trace_id, events, f"python@{PYTHON_VERSION}"
        )

    except Exception as e:
        logger.error(f"Error sending events: {e}")


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
            send_events_sync(page, session_id, trace_id)
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
                await send_events_async(page, session_id, trace_id)
                await asyncio.sleep(2)
            logger.info("Event collection stopped")
        except Exception as e:
            logger.error(f"Event collection stopped: {e}")

    # Create and store task
    task = asyncio.create_task(collection_loop())

    # Clean up task when page closes
    page.on("close", lambda: task.cancel())


@_with_tracer_wrapper
def _wrap(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(
        f"browser_context.{to_wrap.get('method')}"
    ) as span:
        page = wrapped(*args, **kwargs)
        session_id = str(uuid.uuid4().hex)
        trace_id = format(get_current_span().get_span_context().trace_id, "032x")
        span.set_attribute("lmnr.internal.has_browser_session", True)
        handle_navigation(page, session_id, trace_id)
        return page


@_with_tracer_wrapper
async def _wrap_async(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(
        f"browser_context.{to_wrap.get('method')}"
    ) as span:
        page = await wrapped(*args, **kwargs)
        session_id = str(uuid.uuid4().hex)
        trace_id = format(get_current_span().get_span_context().trace_id, "032x")
        span.set_attribute("lmnr.internal.has_browser_session", True)
        await handle_navigation_async(page, session_id, trace_id)
        return page


class PlaywrightInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, SDK_VERSION, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(
                        tracer,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we're not instrumenting everything

        for wrapped_method in WRAPPED_METHODS_ASYNC:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap_async(
                        tracer,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we're not instrumenting everything

    def _uninstrument(self, **kwargs):
        for wrapped_method in [*WRAPPED_METHODS, *WRAPPED_METHODS_ASYNC]:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            unwrap(wrap_package, f"{wrap_object}.{wrap_method}")

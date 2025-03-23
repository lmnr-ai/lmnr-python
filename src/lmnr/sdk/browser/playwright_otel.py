import asyncio
import logging
import os
import threading
import time
import uuid

from lmnr.sdk.browser.utils import (
    INJECT_PLACEHOLDER,
    with_tracer_and_client_wrapper,
    retry_sync,
    retry_async,
)
from lmnr.sdk.client.async_client import AsyncLaminarClient
from lmnr.sdk.client.sync_client import LaminarClient
from lmnr.version import SDK_VERSION

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
    },
]

WRAPPED_ATTRIBUTES = [
    {
        "package": "playwright.sync_api",
        "object": "BrowserContext",
        "attribute": "pages",
    },
]

WRAPPED_METHODS_ASYNC = [
    {
        "package": "playwright.async_api",
        "object": "BrowserContext",
        "method": "new_page",
    },
]

WRAPPED_ATTRIBUTES_ASYNC = [
    {
        "package": "playwright.async_api",
        "object": "BrowserContext",
        "attribute": "pages",
    },
]

current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "rrweb", "rrweb.min.js"), "r") as f:
    RRWEB_CONTENT = f"() => {{ {f.read()} }}"

# Track pages we've already instrumented to avoid double-instrumentation
instrumented_pages = set()
instrumented_pages_lock = threading.Lock()

# For async pages
async_instrumented_pages = set()
async_instrumented_pages_lock = asyncio.Lock()


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

        await client.send_browser_events(session_id, trace_id, events)

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

        client.send_browser_events(session_id, trace_id, events)

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


def handle_navigation(
    page: SyncPage, session_id: str, trace_id: str, client: LaminarClient
):
    # Check if we've already instrumented this page
    page_id = id(page)
    if page_id in instrumented_pages:
        return
    instrumented_pages.add(page_id)

    def on_load():
        try:
            inject_rrweb(page)
        except Exception as e:
            logger.error(f"Error in on_load handler: {e}")

    page.on("load", on_load)
    inject_rrweb(page)

    def collection_loop():
        while not page.is_closed():  # Stop when page closes
            send_events_sync(page, session_id, trace_id, client)
            time.sleep(2)
        # Clean up when page closes

        if page_id in instrumented_pages:
            instrumented_pages.remove(page_id)

    thread = threading.Thread(target=collection_loop, daemon=True)
    thread.start()


async def handle_navigation_async(
    page: Page, session_id: str, trace_id: str, client: AsyncLaminarClient
):
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


@with_tracer_and_client_wrapper
def _wrap(
    tracer: Tracer, client: LaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    with tracer.start_as_current_span(
        f"browser_context.{to_wrap.get('method')}"
    ) as span:
        page = wrapped(*args, **kwargs)
        session_id = str(uuid.uuid4().hex)
        trace_id = format(get_current_span().get_span_context().trace_id, "032x")
        span.set_attribute("lmnr.internal.has_browser_session", True)
        handle_navigation(page, session_id, trace_id, client)
        return page


@with_tracer_and_client_wrapper
async def _wrap_async(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    with tracer.start_as_current_span(
        f"browser_context.{to_wrap.get('method')}"
    ) as span:
        page = await wrapped(*args, **kwargs)
        session_id = str(uuid.uuid4().hex)
        trace_id = format(get_current_span().get_span_context().trace_id, "032x")
        span.set_attribute("lmnr.internal.has_browser_session", True)
        await handle_navigation_async(page, session_id, trace_id, client)
        return page


class InstrumentedPageList(list):
    """Wrapper around the list of pages that instruments accessed pages"""

    def __init__(self, original_list, tracer, client, *args, **kwargs):
        super().__init__(original_list)
        self._original_list = original_list
        self._tracer = tracer
        self._client = client
        self._is_async = kwargs.get("is_async", False)
        for page in self._original_list:
            with tracer.start_as_current_span("browser_context.page") as span:
                session_id = str(uuid.uuid4().hex)
                trace_id = format(
                    get_current_span().get_span_context().trace_id, "032x"
                )
                span.set_attribute("lmnr.internal.has_browser_session", True)
                if self._is_async:
                    asyncio.create_task(
                        handle_navigation_async(
                            page, session_id, trace_id, self._client
                        )
                    )
                else:
                    handle_navigation(page, session_id, trace_id, self._client)

    def __getitem__(self, idx):
        return self._original_list[idx]

    def __len__(self):
        return len(self._original_list)

    # Forward all other methods/attributes to the original list
    def __getattr__(self, name):
        return getattr(self._original_list, name)


class PlaywrightInstrumentor(BaseInstrumentor):
    def __init__(self, client: LaminarClient, async_client: AsyncLaminarClient):
        super().__init__()
        self.client = client
        self.async_client = async_client

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
                        self.client,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some module is missing

        for wrapped_attr in WRAPPED_ATTRIBUTES:
            package_name = wrapped_attr.get("package")
            object_name = wrapped_attr.get("object")
            attribute_name = wrapped_attr.get("attribute")
            try:
                module = __import__(package_name, fromlist=[object_name])
                cls = getattr(module, object_name)
                original_property = getattr(cls, attribute_name)

                def wrapped_getter(instance, *args, **kwargs):
                    original_value = original_property.__get__(
                        instance, cls, *args, **kwargs
                    )
                    return InstrumentedPageList(
                        original_value, tracer, self.client, is_async=False
                    )

                setattr(cls, attribute_name, property(wrapped_getter))
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some module is missing

        # Wrap async methods
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
                        self.async_client,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some module is missing

        # Wrap async attributes
        for wrapped_attr in WRAPPED_ATTRIBUTES_ASYNC:
            package_name = wrapped_attr.get("package")
            object_name = wrapped_attr.get("object")
            attribute_name = wrapped_attr.get("attribute")
            try:
                module = __import__(package_name, fromlist=[object_name])
                cls = getattr(module, object_name)
                original_property = getattr(cls, attribute_name)

                def wrapped_getter(instance, *args, **kwargs):
                    original_value = original_property.__get__(
                        instance, cls, *args, **kwargs
                    )
                    return InstrumentedPageList(
                        original_value, tracer, self.async_client, is_async=True
                    )

                setattr(cls, attribute_name, property(wrapped_getter))
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some module is missing

    def _uninstrument(self, **kwargs):
        # Unwrap methods
        for wrapped_method in WRAPPED_METHODS + WRAPPED_METHODS_ASYNC:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            unwrap(wrap_package, f"{wrap_object}.{wrap_method}")

        # # Unwrap attributes
        # for wrapped_attr in WRAPPED_ATTRIBUTES + WRAPPED_ATTRIBUTES_ASYNC:
        #     package_name = wrapped_attr.get("package")
        #     object_name = wrapped_attr.get("object")
        #     attribute_name = wrapped_attr.get("attribute")
        #     try:
        #         module = __import__(package_name, fromlist=[object_name])
        #         obj_class = getattr(module, object_name)
        #         unwrap(obj_class, attribute_name)
        #     except (ModuleNotFoundError, AttributeError):
        #         pass  # that's ok, we don't want to fail if some module is missing

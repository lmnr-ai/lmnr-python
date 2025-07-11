import logging
import uuid
from typing import Collection, Dict, List, Any

from lmnr.opentelemetry_lib.utils.package_check import is_package_installed
from lmnr.sdk.browser.pw_utils import handle_navigation_async, handle_navigation_sync, cleanup_all_sessions
from lmnr.sdk.browser.utils import with_tracer_and_client_wrapper
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.version import __version__

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import (
    get_tracer,
    Tracer,
)
from wrapt import wrap_function_wrapper

try:
    if is_package_installed("playwright"):
        from playwright.async_api import Browser, BrowserContext, Page
        from playwright.sync_api import (
            Browser as SyncBrowser,
            BrowserContext as SyncBrowserContext,
            Page as SyncPage,
        )
    elif is_package_installed("patchright"):
        from patchright.async_api import Browser, BrowserContext, Page
        from patchright.sync_api import (
            Browser as SyncBrowser,
            BrowserContext as SyncBrowserContext,
            Page as SyncPage,
        )
    else:
        raise ImportError(
            "Attempted to import lmnr.sdk.browser.playwright_otel, but neither "
            "playwright nor patchright is installed. Use `pip install playwright` "
            "or `pip install patchright` to install one of the supported browsers."
        )
except ImportError as e:
    raise ImportError(
        f"Attempted to import {__file__}, but it is designed "
        "to patch Playwright, which is not installed. Use `pip install playwright` "
        "or `pip install patchright` to install Playwright or remove this import."
    ) from e

# all available versions at https://pypi.org/project/playwright/#history
_instruments = ("playwright >= 1.9.0",)
logger = logging.getLogger(__name__)

# Global registry to track event handlers and cleanup functions
_cleanup_registry: Dict[str, List[Any]] = {}


def _register_cleanup(resource_id: str, cleanup_func: Any):
    """Register a cleanup function for a resource"""
    if resource_id not in _cleanup_registry:
        _cleanup_registry[resource_id] = []
    _cleanup_registry[resource_id].append(cleanup_func)


def _cleanup_resource(resource_id: str):
    """Clean up all registered cleanup functions for a resource"""
    if resource_id in _cleanup_registry:
        for cleanup_func in _cleanup_registry[resource_id]:
            try:
                cleanup_func()
            except Exception as e:
                logger.debug(f"Error during cleanup: {e}")
        del _cleanup_registry[resource_id]


@with_tracer_and_client_wrapper
def _wrap_new_page(
    tracer: Tracer, client: LaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    page = wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    
    # Register cleanup for when page is closed
    page_id = str(id(page))
    
    def cleanup_page():
        _cleanup_resource(page_id)
    
    page.on("close", cleanup_page)
    
    handle_navigation_sync(page, session_id, client)
    return page


@with_tracer_and_client_wrapper
async def _wrap_new_page_async(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    page = await wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    
    # Register cleanup for when page is closed
    page_id = str(id(page))
    
    def cleanup_page():
        _cleanup_resource(page_id)
    
    page.on("close", cleanup_page)
    
    await handle_navigation_async(page, session_id, client)
    return page


@with_tracer_and_client_wrapper
def _wrap_new_browser_sync(
    tracer: Tracer, client: LaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    browser: SyncBrowser = wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    browser_id = str(id(browser))
    
    # Track handlers for cleanup
    handlers = []
    
    def cleanup_browser():
        # Remove all registered handlers
        for context, event_name, handler in handlers:
            try:
                context.off(event_name, handler)
            except Exception as e:
                logger.debug(f"Error removing handler: {e}")
        handlers.clear()
        _cleanup_resource(browser_id)
    
    _register_cleanup(browser_id, cleanup_browser)
    
    for context in browser.contexts:

        def handle_page_navigation(page: SyncPage):
            return handle_navigation_sync(page, session_id, client)

        context.on("page", handle_page_navigation)
        handlers.append((context, "page", handle_page_navigation))
        
        # Register cleanup for context close
        def cleanup_context():
            try:
                context.off("page", handle_page_navigation)
            except Exception as e:
                logger.debug(f"Error removing context handler: {e}")
        
        context.on("close", cleanup_context)
        handlers.append((context, "close", cleanup_context))

        for page in context.pages:
            handle_navigation_sync(page, session_id, client)
    
    # Register cleanup for browser close
    browser.on("disconnected", cleanup_browser)
    
    return browser


@with_tracer_and_client_wrapper
async def _wrap_new_browser_async(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    browser: Browser = await wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    browser_id = str(id(browser))
    
    # Track handlers for cleanup
    handlers = []
    
    def cleanup_browser():
        # Remove all registered handlers
        for context, event_name, handler in handlers:
            try:
                context.off(event_name, handler)
            except Exception as e:
                logger.debug(f"Error removing handler: {e}")
        handlers.clear()
        _cleanup_resource(browser_id)
    
    _register_cleanup(browser_id, cleanup_browser)
    
    for context in browser.contexts:

        async def handle_page_navigation(page: Page):
            return await handle_navigation_async(page, session_id, client)

        context.on("page", handle_page_navigation)
        handlers.append((context, "page", handle_page_navigation))
        
        # Register cleanup for context close
        def cleanup_context():
            try:
                context.off("page", handle_page_navigation)
            except Exception as e:
                logger.debug(f"Error removing context handler: {e}")
        
        context.on("close", cleanup_context)
        handlers.append((context, "close", cleanup_context))

        for page in context.pages:
            await handle_navigation_async(page, session_id, client)
    
    # Register cleanup for browser close
    browser.on("disconnected", cleanup_browser)
    
    return browser


@with_tracer_and_client_wrapper
def _wrap_new_context_sync(
    tracer: Tracer, client: LaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    context: SyncBrowserContext = wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    context_id = str(id(context))
    
    # Track handlers for cleanup
    handlers = []
    
    def cleanup_context():
        # Remove all registered handlers
        for obj, event_name, handler in handlers:
            try:
                obj.off(event_name, handler)
            except Exception as e:
                logger.debug(f"Error removing handler: {e}")
        handlers.clear()
        _cleanup_resource(context_id)
    
    _register_cleanup(context_id, cleanup_context)

    def handle_page_navigation(page: SyncPage):
        return handle_navigation_sync(page, session_id, client)

    context.on("page", handle_page_navigation)
    handlers.append((context, "page", handle_page_navigation))
    
    # Register cleanup for context close
    context.on("close", cleanup_context)
    handlers.append((context, "close", cleanup_context))
    
    for page in context.pages:
        handle_navigation_sync(page, session_id, client)
    
    return context


@with_tracer_and_client_wrapper
async def _wrap_new_context_async(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    context: BrowserContext = await wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    context_id = str(id(context))
    
    # Track handlers for cleanup
    handlers = []
    
    def cleanup_context():
        # Remove all registered handlers
        for obj, event_name, handler in handlers:
            try:
                obj.off(event_name, handler)
            except Exception as e:
                logger.debug(f"Error removing handler: {e}")
        handlers.clear()
        _cleanup_resource(context_id)
    
    _register_cleanup(context_id, cleanup_context)

    async def handle_page_navigation(page):
        return await handle_navigation_async(page, session_id, client)

    context.on("page", handle_page_navigation)
    handlers.append((context, "page", handle_page_navigation))
    
    # Register cleanup for context close
    context.on("close", cleanup_context)
    handlers.append((context, "close", cleanup_context))
    
    for page in context.pages:
        await handle_navigation_async(page, session_id, client)
    
    return context


WRAPPED_METHODS = [
    {
        "package": "playwright.sync_api",
        "object": "BrowserContext",
        "method": "new_page",
        "wrapper": _wrap_new_page,
    },
    {
        "package": "playwright.sync_api",
        "object": "Browser",
        "method": "new_page",
        "wrapper": _wrap_new_page,
    },
    {
        "package": "playwright.sync_api",
        "object": "BrowserType",
        "method": "launch",
        "wrapper": _wrap_new_browser_sync,
    },
    {
        "package": "playwright.sync_api",
        "object": "BrowserType",
        "method": "connect",
        "wrapper": _wrap_new_browser_sync,
    },
    {
        "package": "playwright.sync_api",
        "object": "BrowserType",
        "method": "connect_over_cdp",
        "wrapper": _wrap_new_browser_sync,
    },
    {
        "package": "playwright.sync_api",
        "object": "Browser",
        "method": "new_context",
        "wrapper": _wrap_new_context_sync,
    },
    {
        "package": "playwright.sync_api",
        "object": "BrowserType",
        "method": "launch_persistent_context",
        "wrapper": _wrap_new_context_sync,
    },
]

WRAPPED_METHODS_ASYNC = [
    {
        "package": "playwright.async_api",
        "object": "BrowserContext",
        "method": "new_page",
        "wrapper": _wrap_new_page_async,
    },
    {
        "package": "playwright.async_api",
        "object": "Browser",
        "method": "new_page",
        "wrapper": _wrap_new_page_async,
    },
    {
        "package": "playwright.async_api",
        "object": "BrowserType",
        "method": "launch",
        "wrapper": _wrap_new_browser_async,
    },
    {
        "package": "playwright.async_api",
        "object": "BrowserType",
        "method": "connect",
        "wrapper": _wrap_new_browser_async,
    },
    {
        "package": "playwright.async_api",
        "object": "BrowserType",
        "method": "connect_over_cdp",
        "wrapper": _wrap_new_browser_async,
    },
    {
        "package": "playwright.async_api",
        "object": "Browser",
        "method": "new_context",
        "wrapper": _wrap_new_context_async,
    },
    {
        "package": "playwright.async_api",
        "object": "BrowserType",
        "method": "launch_persistent_context",
        "wrapper": _wrap_new_context_async,
    },
]


class PlaywrightInstrumentor(BaseInstrumentor):
    def __init__(self, client: LaminarClient, async_client: AsyncLaminarClient):
        super().__init__()
        self.client = client
        self.async_client = async_client

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    wrapped_method.get("wrapper")(
                        tracer,
                        self.client,
                        wrapped_method,
                    ),
                )
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
                    wrapped_method.get("wrapper")(
                        tracer,
                        self.async_client,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some module is missing

    def _uninstrument(self, **kwargs):
        # Clean up all active sessions and threads
        cleanup_all_sessions()
        
        # Clean up all registered cleanup functions
        global _cleanup_registry
        for resource_id in list(_cleanup_registry.keys()):
            _cleanup_resource(resource_id)
        
        # Unwrap methods
        for wrapped_method in WRAPPED_METHODS + WRAPPED_METHODS_ASYNC:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            unwrap(wrap_package, f"{wrap_object}.{wrap_method}")

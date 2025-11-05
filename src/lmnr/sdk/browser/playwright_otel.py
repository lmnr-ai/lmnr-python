import logging
import uuid

from lmnr.opentelemetry_lib.utils.package_check import is_package_installed
from lmnr.sdk.browser.pw_utils import (
    start_recording_events_async,
    start_recording_events_sync,
    take_full_snapshot,
    take_full_snapshot_async,
)
from lmnr.sdk.browser.utils import with_tracer_and_client_wrapper
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.version import __version__

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import (
    get_tracer,
    Tracer,
)
from typing import Collection
from wrapt import wrap_function_wrapper

try:
    if is_package_installed("playwright"):
        from playwright.async_api import Browser, BrowserContext
        from playwright.sync_api import (
            Browser as SyncBrowser,
            BrowserContext as SyncBrowserContext,
        )
    elif is_package_installed("patchright"):
        from patchright.async_api import Browser, BrowserContext
        from patchright.sync_api import (
            Browser as SyncBrowser,
            BrowserContext as SyncBrowserContext,
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


@with_tracer_and_client_wrapper
def _wrap_new_browser_sync(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    browser: SyncBrowser = wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)

    def create_page_handler(session_id, client):
        def page_handler(page):
            start_recording_events_sync(page, session_id, client)

        return page_handler

    for context in browser.contexts:
        page_handler = create_page_handler(session_id, client)
        context.on("page", page_handler)
        for page in context.pages:
            start_recording_events_sync(page, session_id, client)

    return browser


@with_tracer_and_client_wrapper
async def _wrap_new_browser_async(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    browser: Browser = await wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)

    def create_page_handler(session_id, client):
        async def page_handler(page):
            await start_recording_events_async(page, session_id, client)

        return page_handler

    for context in browser.contexts:
        page_handler = create_page_handler(session_id, client)
        context.on("page", page_handler)
        for page in context.pages:
            await start_recording_events_async(page, session_id, client)
    return browser


@with_tracer_and_client_wrapper
def _wrap_new_context_sync(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    context: SyncBrowserContext = wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)

    def create_page_handler(session_id, client):
        def page_handler(page):
            start_recording_events_sync(page, session_id, client)

        return page_handler

    page_handler = create_page_handler(session_id, client)
    context.on("page", page_handler)
    for page in context.pages:
        start_recording_events_sync(page, session_id, client)

    return context


@with_tracer_and_client_wrapper
async def _wrap_new_context_async(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    context: BrowserContext = await wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)

    def create_page_handler(session_id, client):
        async def page_handler(page):
            await start_recording_events_async(page, session_id, client)

        return page_handler

    page_handler = create_page_handler(session_id, client)
    context.on("page", page_handler)
    for page in context.pages:
        await start_recording_events_async(page, session_id, client)

    return context


@with_tracer_and_client_wrapper
def _wrap_bring_to_front_sync(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    wrapped(*args, **kwargs)
    take_full_snapshot(instance)


@with_tracer_and_client_wrapper
async def _wrap_bring_to_front_async(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    await wrapped(*args, **kwargs)
    await take_full_snapshot_async(instance)


@with_tracer_and_client_wrapper
def _wrap_browser_new_page_sync(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    page = wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    start_recording_events_sync(page, session_id, client)
    return page


@with_tracer_and_client_wrapper
async def _wrap_browser_new_page_async(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    page = await wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    await start_recording_events_async(page, session_id, client)
    return page


WRAPPED_METHODS = [
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
    {
        "package": "playwright.sync_api",
        "object": "Page",
        "method": "bring_to_front",
        "wrapper": _wrap_bring_to_front_sync,
    },
    {
        "package": "playwright.sync_api",
        "object": "Browser",
        "method": "new_page",
        "wrapper": _wrap_browser_new_page_sync,
    },
]

WRAPPED_METHODS_ASYNC = [
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
    {
        "package": "playwright.async_api",
        "object": "Page",
        "method": "bring_to_front",
        "wrapper": _wrap_bring_to_front_async,
    },
    {
        "package": "playwright.async_api",
        "object": "Browser",
        "method": "new_page",
        "wrapper": _wrap_browser_new_page_async,
    },
]


class PlaywrightInstrumentor(BaseInstrumentor):
    def __init__(self, async_client: AsyncLaminarClient):
        super().__init__()
        self.async_client = async_client

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Both sync and async methods use async_client because we are using
        # a background asyncio loop for async sends
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
                        self.async_client,
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
        # Unwrap methods
        for wrapped_method in WRAPPED_METHODS + WRAPPED_METHODS_ASYNC:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            unwrap(wrap_package, f"{wrap_object}.{wrap_method}")

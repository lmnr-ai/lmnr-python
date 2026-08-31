import logging
import uuid

from lmnr.opentelemetry_lib.utils.package_check import is_package_installed
from lmnr.sdk.browser.pw_utils import (
    start_recording_events_async,
    start_recording_events_sync,
    take_full_snapshot,
    take_full_snapshot_async,
)
from importlib.metadata import version
from typing import Any, Collection, Sequence

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient

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


def _wrap_new_browser_sync(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    client: AsyncLaminarClient,
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


async def _wrap_new_browser_async(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    client: AsyncLaminarClient,
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


def _wrap_new_context_sync(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    client: AsyncLaminarClient,
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


async def _wrap_new_context_async(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    client: AsyncLaminarClient,
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


def _wrap_bring_to_front_sync(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    client: AsyncLaminarClient,
):
    wrapped(*args, **kwargs)
    take_full_snapshot(instance)


async def _wrap_bring_to_front_async(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    client: AsyncLaminarClient,
):
    await wrapped(*args, **kwargs)
    await take_full_snapshot_async(instance)


def _wrap_browser_new_page_sync(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    client: AsyncLaminarClient,
):
    page = wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    start_recording_events_sync(page, session_id, client)
    return page


async def _wrap_browser_new_page_async(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    client: AsyncLaminarClient,
):
    page = await wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    await start_recording_events_async(page, session_id, client)
    return page




WRAPPED_FUNCTIONS: list[WrappedFunctionSpec] = [
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="BrowserType",
        method_name="launch",
        is_async=False,
        wrapper_function=_wrap_new_browser_sync,
    ),
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="BrowserType",
        method_name="connect",
        is_async=False,
        wrapper_function=_wrap_new_browser_sync,
    ),
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="BrowserType",
        method_name="connect_over_cdp",
        is_async=False,
        wrapper_function=_wrap_new_browser_sync,
    ),
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="Browser",
        method_name="new_context",
        is_async=False,
        wrapper_function=_wrap_new_context_sync,
    ),
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="BrowserType",
        method_name="launch_persistent_context",
        is_async=False,
        wrapper_function=_wrap_new_context_sync,
    ),
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="Page",
        method_name="bring_to_front",
        is_async=False,
        wrapper_function=_wrap_bring_to_front_sync,
    ),
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="Browser",
        method_name="new_page",
        is_async=False,
        wrapper_function=_wrap_browser_new_page_sync,
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="BrowserType",
        method_name="launch",
        is_async=True,
        wrapper_function=_wrap_new_browser_async,
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="BrowserType",
        method_name="connect",
        is_async=True,
        wrapper_function=_wrap_new_browser_async,
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="BrowserType",
        method_name="connect_over_cdp",
        is_async=True,
        wrapper_function=_wrap_new_browser_async,
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="Browser",
        method_name="new_context",
        is_async=True,
        wrapper_function=_wrap_new_context_async,
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="BrowserType",
        method_name="launch_persistent_context",
        is_async=True,
        wrapper_function=_wrap_new_context_async,
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="Page",
        method_name="bring_to_front",
        is_async=True,
        wrapper_function=_wrap_bring_to_front_async,
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="Browser",
        method_name="new_page",
        is_async=True,
        wrapper_function=_wrap_browser_new_page_async,
    ),
]


class PlaywrightInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    def __init__(self, async_client: AsyncLaminarClient):
        super().__init__()
        self.async_client = async_client
        self.instrumentor_config = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                pw_version = version("playwright")
            except Exception as e:
                logger.debug(f"Failed to get playwright version {e}")
                pw_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="playwright",
                version=pw_version,
            )
        return self._scope

    def wrapper_kwargs(self) -> dict[str, Any]:
        # Both sync and async wrappers get the ASYNC client on purpose: sends go
        # through a background asyncio loop either way.
        return {"client": self.async_client}

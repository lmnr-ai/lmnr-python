from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Collection, Coroutine, Sequence
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.utils.package_check import is_package_installed
from lmnr.sdk.browser.pw_utils import (
    start_recording_events_async,
    start_recording_events_sync,
    take_full_snapshot,
    take_full_snapshot_async,
)
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient

if TYPE_CHECKING:
    # `from __future__ import annotations` makes every annotation in this file a
    # deferred string, so this import (unlike the patchright fallback below) is
    # never evaluated at runtime — it exists purely so pyright can resolve the
    # types below. playwright and patchright are drop-in forks of one another
    # (same API shape), so typing against playwright's stubs is accurate
    # regardless of which package the user actually has installed.
    from playwright.async_api import Browser, BrowserContext, BrowserType, Page
    from playwright.sync_api import Browser as SyncBrowser
    from playwright.sync_api import BrowserContext as SyncBrowserContext
    from playwright.sync_api import BrowserType as SyncBrowserType
    from playwright.sync_api import Page as SyncPage

if not (is_package_installed("playwright") or is_package_installed("patchright")):
    raise ImportError(
        f"Attempted to import {__file__}, but it is designed "
        + "to patch Playwright, which is not installed. Use `pip install playwright` "
        + "or `pip install patchright` to install Playwright or remove this import."
    )

# all available versions at https://pypi.org/project/playwright/#history
_instruments = ("playwright >= 1.9.0",)
logger = logging.getLogger(__name__)


def _wrap_new_browser_sync(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., SyncBrowser],
    _instance: SyncBrowserType,
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    *,
    client: AsyncLaminarClient,
) -> SyncBrowser:
    browser = wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)

    def create_page_handler(
        session_id: str, client: AsyncLaminarClient
    ) -> Callable[[SyncPage], None]:
        def page_handler(page: SyncPage) -> None:
            start_recording_events_sync(page, session_id, client)

        return page_handler

    for context in browser.contexts:
        page_handler = create_page_handler(session_id, client)
        _page_listener = context.on("page", page_handler)
        for page in context.pages:
            start_recording_events_sync(page, session_id, client)

    return browser


async def _wrap_new_browser_async(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Coroutine[Any, Any, Browser]],  # pyright: ignore[reportExplicitAny]
    _instance: BrowserType,
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    *,
    client: AsyncLaminarClient,
) -> Browser:
    browser = await wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)

    def create_page_handler(
        session_id: str, client: AsyncLaminarClient
    ) -> Callable[[Page], Coroutine[Any, Any, None]]:  # pyright: ignore[reportExplicitAny]
        async def page_handler(page: Page) -> None:
            await start_recording_events_async(page, session_id, client)

        return page_handler

    for context in browser.contexts:
        page_handler = create_page_handler(session_id, client)
        _page_listener = context.on("page", page_handler)
        for page in context.pages:
            await start_recording_events_async(page, session_id, client)
    return browser


def _wrap_new_context_sync(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., SyncBrowserContext],
    _instance: SyncBrowser | SyncBrowserType,
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    *,
    client: AsyncLaminarClient,
) -> SyncBrowserContext:
    context = wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)

    def create_page_handler(
        session_id: str, client: AsyncLaminarClient
    ) -> Callable[[SyncPage], None]:
        def page_handler(page: SyncPage) -> None:
            start_recording_events_sync(page, session_id, client)

        return page_handler

    page_handler = create_page_handler(session_id, client)
    _page_listener = context.on("page", page_handler)
    for page in context.pages:
        start_recording_events_sync(page, session_id, client)

    return context


async def _wrap_new_context_async(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Coroutine[Any, Any, BrowserContext]],  # pyright: ignore[reportExplicitAny]
    _instance: Browser | BrowserType,
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    *,
    client: AsyncLaminarClient,
) -> BrowserContext:
    context = await wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)

    def create_page_handler(
        session_id: str, client: AsyncLaminarClient
    ) -> Callable[[Page], Coroutine[Any, Any, None]]:  # pyright: ignore[reportExplicitAny]
        async def page_handler(page: Page) -> None:
            await start_recording_events_async(page, session_id, client)

        return page_handler

    page_handler = create_page_handler(session_id, client)
    _page_listener = context.on("page", page_handler)
    for page in context.pages:
        await start_recording_events_async(page, session_id, client)

    return context


def _wrap_bring_to_front_sync(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., None],
    instance: SyncPage,
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    *,
    _client: AsyncLaminarClient,
) -> None:
    wrapped(*args, **kwargs)
    _snapshot_taken = take_full_snapshot(instance)


async def _wrap_bring_to_front_async(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Coroutine[Any, Any, None]],  # pyright: ignore[reportExplicitAny]
    instance: Page,
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    *,
    _client: AsyncLaminarClient,
) -> None:
    await wrapped(*args, **kwargs)
    _snapshot_taken = await take_full_snapshot_async(instance)


def _wrap_browser_new_page_sync(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., SyncPage],
    _instance: SyncBrowser,
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    *,
    client: AsyncLaminarClient,
) -> SyncPage:
    page = wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    start_recording_events_sync(page, session_id, client)
    return page


async def _wrap_browser_new_page_async(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Coroutine[Any, Any, Page]],  # pyright: ignore[reportExplicitAny]
    _instance: Browser,
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    *,
    client: AsyncLaminarClient,
) -> Page:
    page = await wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    await start_recording_events_async(page, session_id, client)
    return page




# Every wrapper below takes a keyword-only `client`, forwarded via
# `wrapper_kwargs()` on `PlaywrightInstrumentor` (see
# wrapper_helpers.add_spec_wrapper) — WrapperHandler can't express that extra
# parameter, so the `reportArgumentType` ignores below are a known-safe mismatch.
WRAPPED_FUNCTIONS: list[WrappedFunctionSpec] = [
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="BrowserType",
        method_name="launch",
        is_async=False,
        wrapper_function=_wrap_new_browser_sync,  # pyright: ignore[reportArgumentType]
    ),
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="BrowserType",
        method_name="connect",
        is_async=False,
        wrapper_function=_wrap_new_browser_sync,  # pyright: ignore[reportArgumentType]
    ),
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="BrowserType",
        method_name="connect_over_cdp",
        is_async=False,
        wrapper_function=_wrap_new_browser_sync,  # pyright: ignore[reportArgumentType]
    ),
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="Browser",
        method_name="new_context",
        is_async=False,
        wrapper_function=_wrap_new_context_sync,  # pyright: ignore[reportArgumentType]
    ),
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="BrowserType",
        method_name="launch_persistent_context",
        is_async=False,
        wrapper_function=_wrap_new_context_sync,  # pyright: ignore[reportArgumentType]
    ),
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="Page",
        method_name="bring_to_front",
        is_async=False,
        wrapper_function=_wrap_bring_to_front_sync,  # pyright: ignore[reportArgumentType]
    ),
    WrappedFunctionSpec(
        package_name="playwright.sync_api",
        object_name="Browser",
        method_name="new_page",
        is_async=False,
        wrapper_function=_wrap_browser_new_page_sync,  # pyright: ignore[reportArgumentType]
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="BrowserType",
        method_name="launch",
        is_async=True,
        wrapper_function=_wrap_new_browser_async,  # pyright: ignore[reportArgumentType]
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="BrowserType",
        method_name="connect",
        is_async=True,
        wrapper_function=_wrap_new_browser_async,  # pyright: ignore[reportArgumentType]
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="BrowserType",
        method_name="connect_over_cdp",
        is_async=True,
        wrapper_function=_wrap_new_browser_async,  # pyright: ignore[reportArgumentType]
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="Browser",
        method_name="new_context",
        is_async=True,
        wrapper_function=_wrap_new_context_async,  # pyright: ignore[reportArgumentType]
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="BrowserType",
        method_name="launch_persistent_context",
        is_async=True,
        wrapper_function=_wrap_new_context_async,  # pyright: ignore[reportArgumentType]
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="Page",
        method_name="bring_to_front",
        is_async=True,
        wrapper_function=_wrap_bring_to_front_async,  # pyright: ignore[reportArgumentType]
    ),
    WrappedFunctionSpec(
        package_name="playwright.async_api",
        object_name="Browser",
        method_name="new_page",
        is_async=True,
        wrapper_function=_wrap_browser_new_page_async,  # pyright: ignore[reportArgumentType]
    ),
]


class PlaywrightInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    def __init__(self, async_client: AsyncLaminarClient):
        super().__init__()
        self.async_client: AsyncLaminarClient = async_client
        self.instrumentor_config: LaminarInstrumentorConfig = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
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

    @override
    def wrapper_kwargs(self) -> dict[str, AsyncLaminarClient]:
        # Both sync and async wrappers get the ASYNC client on purpose: sends go
        # through a background asyncio loop either way.
        return {"client": self.async_client}

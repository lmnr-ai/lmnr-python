import logging
import uuid

from lmnr.sdk.browser.pw_utils import handle_navigation_async, handle_navigation_sync
from lmnr.sdk.browser.utils import with_tracer_and_client_wrapper
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.client.synchronous.sync_client import LaminarClient
from lmnr.version import __version__

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer, Tracer, get_current_span, Span, use_span
from typing import Collection
from wrapt import wrap_function_wrapper

try:
    from playwright.async_api import Browser
    from playwright.sync_api import Browser as SyncBrowser
except ImportError as e:
    raise ImportError(
        f"Attempted to import {__file__}, but it is designed "
        "to patch Playwright, which is not installed. Use `pip install playwright` "
        "to install Playwright or remove this import."
    ) from e

# all available versions at https://pypi.org/project/playwright/#history
_instruments = ("playwright >= 1.9.0",)
logger = logging.getLogger(__name__)

_context_spans: dict[str, Span] = {}


@with_tracer_and_client_wrapper
def _wrap_new_page(
    tracer: Tracer, client: LaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    with tracer.start_as_current_span(
        f"{to_wrap.get('object')}.{to_wrap.get('method')}"
    ) as span:
        page = wrapped(*args, **kwargs)
        session_id = str(uuid.uuid4().hex)
        trace_id = format(get_current_span().get_span_context().trace_id, "032x")
        span.set_attribute("lmnr.internal.has_browser_session", True)
        handle_navigation_sync(page, session_id, trace_id, client)
        return page


@with_tracer_and_client_wrapper
async def _wrap_new_page_async(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    with tracer.start_as_current_span(
        f"{to_wrap.get('object')}.{to_wrap.get('method')}"
    ) as span:
        page = await wrapped(*args, **kwargs)
        session_id = str(uuid.uuid4().hex)
        trace_id = format(get_current_span().get_span_context().trace_id, "032x")
        span.set_attribute("lmnr.internal.has_browser_session", True)
        await handle_navigation_async(page, session_id, trace_id, client)
        return page


@with_tracer_and_client_wrapper
def _wrap_new_browser_sync(
    tracer: Tracer, client: LaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    global _context_spans
    browser: SyncBrowser = wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    for context in browser.contexts:
        span = tracer.start_span(
            name=f"{to_wrap.get('object')}.{to_wrap.get('method')}"
        )
        span.set_attribute("lmnr.internal.has_browser_session", True)
        _context_spans[id(context)] = span
        with use_span(span, end_on_exit=False):
            for page in context.pages:
                trace_id = format(
                    get_current_span().get_span_context().trace_id, "032x"
                )
                handle_navigation_sync(page, session_id, trace_id, client)
    return browser


@with_tracer_and_client_wrapper
async def _wrap_new_browser_async(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    global _context_spans
    browser: Browser = await wrapped(*args, **kwargs)
    session_id = str(uuid.uuid4().hex)
    for context in browser.contexts:
        span = tracer.start_span(
            name=f"{to_wrap.get('object')}.{to_wrap.get('method')}"
        )
        span.set_attribute("lmnr.internal.has_browser_session", True)
        _context_spans[id(context)] = span
        with use_span(span, end_on_exit=False):
            for page in context.pages:
                trace_id = format(
                    get_current_span().get_span_context().trace_id, "032x"
                )
                await handle_navigation_async(page, session_id, trace_id, client)
    return browser


@with_tracer_and_client_wrapper
def _wrap_close_context_sync(
    tracer: Tracer,
    client: LaminarClient,
    to_wrap,
    wrapped,
    instance: SyncBrowser,
    args,
    kwargs,
):
    global _context_spans
    key = id(instance)
    span = _context_spans.get(key)
    if span:
        if span.is_recording():
            span.end()
        _context_spans.pop(key)
    return wrapped(*args, **kwargs)


@with_tracer_and_client_wrapper
async def _wrap_close_context_async(
    tracer: Tracer,
    client: AsyncLaminarClient,
    to_wrap,
    wrapped,
    instance: Browser,
    args,
    kwargs,
):
    global _context_spans
    key = id(instance)
    span = _context_spans.get(key)
    if span:
        if span.is_recording():
            span.end()
        _context_spans.pop(key)
    return await wrapped(*args, **kwargs)


@with_tracer_and_client_wrapper
def _wrap_close_browser_sync(
    tracer: Tracer,
    client: LaminarClient,
    to_wrap,
    wrapped,
    instance: SyncBrowser,
    args,
    kwargs,
):
    global _context_spans
    for context in instance.contexts:
        key = id(context)
        span = _context_spans.get(key)
        if span:
            if span.is_recording():
                span.end()
            _context_spans.pop(key)
    return wrapped(*args, **kwargs)


@with_tracer_and_client_wrapper
async def _wrap_close_browser_async(
    tracer: Tracer,
    client: AsyncLaminarClient,
    to_wrap,
    wrapped,
    instance: Browser,
    args,
    kwargs,
):
    global _context_spans
    for context in instance.contexts:
        key = id(context)
        span = _context_spans.get(key)
        if span:
            if span.is_recording():
                span.end()
            _context_spans.pop(key)
    return await wrapped(*args, **kwargs)


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
        "method": "close",
        "wrapper": _wrap_close_browser_sync,
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
        "method": "close",
        "wrapper": _wrap_close_browser_async,
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
        # Unwrap methods
        global _context_spans
        for wrapped_method in WRAPPED_METHODS + WRAPPED_METHODS_ASYNC:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            unwrap(wrap_package, f"{wrap_object}.{wrap_method}")
            for span in _context_spans.values():
                if span.is_recording():
                    span.end()
            _context_spans = {}

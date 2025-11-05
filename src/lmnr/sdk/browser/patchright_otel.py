from lmnr.sdk.browser.playwright_otel import (
    _wrap_bring_to_front_async,
    _wrap_bring_to_front_sync,
    _wrap_new_browser_sync,
    _wrap_new_browser_async,
    _wrap_new_context_sync,
    _wrap_new_context_async,
)
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from lmnr.version import __version__
from typing import Collection
from wrapt import wrap_function_wrapper

_instruments = ("patchright >= 1.9.0",)

WRAPPED_METHODS = [
    {
        "package": "patchright.sync_api",
        "object": "BrowserType",
        "method": "launch",
        "wrapper": _wrap_new_browser_sync,
    },
    {
        "package": "patchright.sync_api",
        "object": "BrowserType",
        "method": "connect",
        "wrapper": _wrap_new_browser_sync,
    },
    {
        "package": "patchright.sync_api",
        "object": "BrowserType",
        "method": "connect_over_cdp",
        "wrapper": _wrap_new_browser_sync,
    },
    {
        "package": "patchright.sync_api",
        "object": "Browser",
        "method": "new_context",
        "wrapper": _wrap_new_context_sync,
    },
    {
        "package": "patchright.sync_api",
        "object": "BrowserType",
        "method": "launch_persistent_context",
        "wrapper": _wrap_new_context_sync,
    },
    {
        "package": "patchright.sync_api",
        "object": "Page",
        "method": "bring_to_front",
        "wrapper": _wrap_bring_to_front_sync,
    },
]

WRAPPED_METHODS_ASYNC = [
    {
        "package": "patchright.async_api",
        "object": "BrowserType",
        "method": "launch",
        "wrapper": _wrap_new_browser_async,
    },
    {
        "package": "patchright.async_api",
        "object": "BrowserType",
        "method": "connect",
        "wrapper": _wrap_new_browser_async,
    },
    {
        "package": "patchright.async_api",
        "object": "BrowserType",
        "method": "connect_over_cdp",
        "wrapper": _wrap_new_browser_async,
    },
    {
        "package": "patchright.async_api",
        "object": "Browser",
        "method": "new_context",
        "wrapper": _wrap_new_context_async,
    },
    {
        "package": "patchright.async_api",
        "object": "BrowserType",
        "method": "launch_persistent_context",
        "wrapper": _wrap_new_context_async,
    },
    {
        "package": "patchright.async_api",
        "object": "Page",
        "method": "bring_to_front",
        "wrapper": _wrap_bring_to_front_async,
    },
]


class PatchrightInstrumentor(BaseInstrumentor):
    def __init__(self, async_client: AsyncLaminarClient):
        super().__init__()
        self.async_client = async_client

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Both sync and async methods use async_client
        # because we are using a background asyncio loop for async sends
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
                pass

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
                pass

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS + WRAPPED_METHODS_ASYNC:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            unwrap(wrap_package, f"{wrap_object}.{wrap_method}")

from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.browser.utils import with_tracer_and_client_wrapper
from lmnr.version import __version__

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer, Tracer
from typing import Collection
from wrapt import wrap_function_wrapper
import uuid

# Stable versions, e.g. 1.0.0, satisfy this condition too
_instruments = ("browser-use >= 1.0.0rc1",)

WRAPPED_METHODS = [
    {
        "package": "browser_use.browser.session",
        "object": "BrowserSession",
        "method": "get_or_create_cdp_session",
    },
]


@with_tracer_and_client_wrapper
async def _wrap(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    from lmnr.sdk.browser.cdp_utils import (
        is_rrweb_present,
        start_recording_events,
    )

    result = await wrapped(*args, **kwargs)

    cdp_session = result
    is_registered = await is_rrweb_present(cdp_session)
    if not is_registered:
        await start_recording_events(cdp_session, str(uuid.uuid4()), client)
    return result


class BrowserUseInstrumentor(BaseInstrumentor):
    def __init__(self, async_client: AsyncLaminarClient):
        super().__init__()
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
                    _wrap(
                        tracer,
                        self.async_client,
                        wrapped_method,
                    ),
                )
            except (ModuleNotFoundError, ImportError):
                pass  # that's ok, we're not instrumenting everything

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            unwrap(
                f"{wrap_package}.{wrap_object}" if wrap_object else wrap_package,
                wrap_method,
            )

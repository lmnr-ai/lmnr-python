import asyncio
import uuid

from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.browser.utils import with_tracer_and_client_wrapper
from lmnr.version import __version__
from lmnr.sdk.browser.cdp_utils import (
    is_recorder_present,
    start_recording_events,
    take_full_snapshot,
)

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer, Tracer
from typing import Collection
from wrapt import wrap_function_wrapper

# Stable versions, e.g. 0.6.0, satisfy this condition too
_instruments = ("browser-use >= 0.6.0rc1",)

WRAPPED_METHODS = [
    {
        "package": "browser_use.browser.session",
        "object": "BrowserSession",
        "method": "get_or_create_cdp_session",
        "action": "inject_session_recorder",
    },
    {
        "package": "browser_use.browser.session",
        "object": "BrowserSession",
        "method": "on_SwitchTabEvent",
        "action": "take_full_snapshot",
    },
]


async def process_wrapped_result(result, instance, client, to_wrap):
    if to_wrap.get("action") == "inject_session_recorder":
        is_registered = await is_recorder_present(result)
        if not is_registered:
            await start_recording_events(result, str(uuid.uuid4()), client)

    if to_wrap.get("action") == "take_full_snapshot":
        target_id = result
        if target_id:
            cdp_session = await instance.get_or_create_cdp_session(target_id)
            await take_full_snapshot(cdp_session)


@with_tracer_and_client_wrapper
async def _wrap(
    tracer: Tracer, client: AsyncLaminarClient, to_wrap, wrapped, instance, args, kwargs
):
    result = await wrapped(*args, **kwargs)
    asyncio.create_task(process_wrapped_result(result, instance, client, to_wrap))

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

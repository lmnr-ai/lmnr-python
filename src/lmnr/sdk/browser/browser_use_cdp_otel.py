import asyncio
import uuid
from collections.abc import Collection, Sequence
from importlib.metadata import version
from typing import Any

from typing_extensions import override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.sdk.browser.cdp_utils import (
    start_recording_events,
    take_full_snapshot,
)
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class BrowserUseCdpSpec(WrappedFunctionSpec, total=False):
    """browser-use CDP's per-method extra.

    `action` selects what `process_wrapped_result` does with the return value —
    these wrappers open no spans, they bootstrap session recording.
    """

    action: str

# Stable versions, e.g. 0.6.0, satisfy this condition too
_instruments = ("browser-use >= 0.6.0rc1",)

# Track CDP sessions that already have recording initialized.
# Checked on every get_or_create_cdp_session call (which is very frequent),
# so this must be a fast O(1) lookup instead of a CDP evaluate call.
_initialized_sessions: set[str] = set()


async def process_wrapped_result(
    result: Any,
    instance: Any,
    client: AsyncLaminarClient,
    to_wrap: BrowserUseCdpSpec,
):
    if to_wrap.get("action") == "inject_session_recorder":
        session_id = result.session_id
        if session_id in _initialized_sessions:
            return
        # Add eagerly to prevent parallel calls from double-initializing
        _initialized_sessions.add(session_id)
        try:
            is_recording = await start_recording_events(
                result, str(uuid.uuid4()), client
            )
            if not is_recording:
                _initialized_sessions.discard(session_id)
        except Exception:
            _initialized_sessions.discard(session_id)

    if to_wrap.get("action") == "take_full_snapshot":
        target_id = result
        if target_id:
            cdp_session = await instance.get_or_create_cdp_session(target_id)
            _ = await take_full_snapshot(cdp_session)


async def _wrap(
    to_wrap: BrowserUseCdpSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    client: AsyncLaminarClient,
):
    result = await wrapped(*args, **kwargs)
    _ = asyncio.create_task(process_wrapped_result(result, instance, client, to_wrap))

    return result


WRAPPED_FUNCTIONS: list[BrowserUseCdpSpec] = [
    BrowserUseCdpSpec(
        package_name="browser_use.browser.session",
        object_name="BrowserSession",
        method_name="get_or_create_cdp_session",
        is_async=True,
        action="inject_session_recorder",
        # `_wrap` takes a keyword-only `client`, forwarded via `wrapper_kwargs()`
        # below (see wrapper_helpers.add_spec_wrapper) — WrapperHandler can't
        # express that extra parameter, so this is a known-safe mismatch.
        wrapper_function=_wrap,  # pyright: ignore[reportArgumentType]
    ),
    BrowserUseCdpSpec(
        package_name="browser_use.browser.session",
        object_name="BrowserSession",
        method_name="on_SwitchTabEvent",
        is_async=True,
        action="take_full_snapshot",
        wrapper_function=_wrap,  # pyright: ignore[reportArgumentType]
    ),
]


class BrowserUseInstrumentor(BaseLaminarInstrumentor):
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
                bu_version = version("browser-use")
            except Exception as e:
                logger.debug(f"Failed to get browser-use version {e}")
                bu_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="browser-use",
                version=bu_version,
            )
        return self._scope

    @override
    def wrapper_kwargs(self) -> dict[str, Any]:
        # The client is instrumentor-level, not per-method, so it rides the
        # base's handler-kwargs channel rather than being stuffed into each spec.
        return {"client": self.async_client}

    @override
    def _uninstrument(self, **kwargs: Any):
        super()._uninstrument(**kwargs)
        # Session ids must not outlive the instrumentation: a stale entry would
        # make a later run skip recorder injection for a reused session id.
        _initialized_sessions.clear()

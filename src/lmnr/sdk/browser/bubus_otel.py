from collections.abc import Callable, Collection, Coroutine, Sequence
from importlib.metadata import version
from typing import Any, TypeVar

from opentelemetry.trace import NonRecordingSpan, SpanContext, get_current_span
from typing_extensions import override

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.tracing.context import get_current_context
from lmnr.sdk.log import get_default_logger

_instruments = ("bubus >= 1.3.0",)
event_id_to_span_context: dict[str, SpanContext] = {}
logger = get_default_logger(__name__)

#: Both wrappers below return exactly what `wrapped` returns, whatever that is
#: per call site — bounding it lets pyright track that identity instead of
#: widening to `Any`.
T = TypeVar("T")


def wrap_dispatch(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    _instance: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
) -> T:
    event = args[0] if args and len(args) > 0 else kwargs.get("event", None)
    if event and hasattr(event, "event_id"):
        event_id = str(event.event_id)
        if event_id:
            span = get_current_span(get_current_context())
            event_id_to_span_context[event_id] = span.get_span_context()
    return wrapped(*args, **kwargs)


async def wrap_process_event(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Coroutine[Any, Any, T]],  # pyright: ignore[reportExplicitAny]
    _instance: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
) -> T:
    event = args[0] if args and len(args) > 0 else kwargs.get("event", None)
    span_context = None
    if event and hasattr(event, "event_id"):
        event_id = str(event.event_id)
        if event_id:
            span_context = event_id_to_span_context.get(event_id)
    if not span_context:
        return await wrapped(*args, **kwargs)
    if not Laminar.is_initialized():
        return await wrapped(*args, **kwargs)
    with Laminar.use_span(NonRecordingSpan(span_context)):
        return await wrapped(*args, **kwargs)


WRAPPED_FUNCTIONS: list[WrappedFunctionSpec] = [
    WrappedFunctionSpec(
        package_name="bubus.service",
        object_name="EventBus",
        method_name="dispatch",
        is_async=False,
        wrapper_function=wrap_dispatch,
    ),
    WrappedFunctionSpec(
        package_name="bubus.service",
        object_name="EventBus",
        method_name="process_event",
        is_async=True,
        wrapper_function=wrap_process_event,
    ),
]


class BubusInstrumentor(BaseLaminarInstrumentor):
    """Context-propagation shim, not telemetry.

    These wrappers open no spans: `dispatch` stashes the current span context
    against the event id, and `process_event` re-attaches it so work done on the
    bus's own task still nests under the dispatching trace.
    """

    _scope: LaminarInstrumentationScopeAttributes | None = None

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                bubus_version = version("bubus")
            except Exception as e:
                logger.debug(f"Failed to get bubus version {e}")
                bubus_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="bubus",
                version=bubus_version,
            )
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config: LaminarInstrumentorConfig = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

    @override
    def _uninstrument(self, **kwargs: dict[str, Any]):  # pyright: ignore[reportExplicitAny]
        super()._uninstrument(**kwargs)
        # This map is the whole point of the instrumentation, so it must not
        # outlive it — a stale entry would re-parent a later event onto a span
        # from a previous run.
        event_id_to_span_context.clear()

from typing import Collection

from lmnr import Laminar
from lmnr.opentelemetry_lib.tracing.context import get_current_context
from lmnr.sdk.log import get_default_logger

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import NonRecordingSpan, get_current_span
from wrapt import wrap_function_wrapper


_instruments = ("bubus >= 1.3.0",)
event_id_to_span_context = {}
logger = get_default_logger(__name__)


def wrap_dispatch(wrapped, instance, args, kwargs):
    event = args[0] if args and len(args) > 0 else kwargs.get("event", None)
    if event and hasattr(event, "event_id"):
        event_id = event.event_id
        if event_id:
            span = get_current_span(get_current_context())
            event_id_to_span_context[event_id] = span.get_span_context()
    return wrapped(*args, **kwargs)


async def wrap_process_event(wrapped, instance, args, kwargs):
    event = args[0] if args and len(args) > 0 else kwargs.get("event", None)
    span_context = None
    if event and hasattr(event, "event_id"):
        event_id = event.event_id
        if event_id:
            span_context = event_id_to_span_context.get(event_id)
    if not span_context:
        return await wrapped(*args, **kwargs)
    if not Laminar.is_initialized():
        return await wrapped(*args, **kwargs)
    with Laminar.use_span(NonRecordingSpan(span_context)):
        return await wrapped(*args, **kwargs)


class BubusInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        try:
            wrap_function_wrapper("bubus.service", "EventBus.dispatch", wrap_dispatch)
        except (ModuleNotFoundError, ImportError):
            pass
        try:
            wrap_function_wrapper(
                "bubus.service", "EventBus.process_event", wrap_process_event
            )
        except (ModuleNotFoundError, ImportError):
            pass

    def _uninstrument(self, **kwargs):
        try:
            unwrap("bubus.service", "EventBus.dispatch")
        except (ModuleNotFoundError, ImportError):
            pass
        try:
            unwrap("bubus.service", "EventBus.process_event")
        except (ModuleNotFoundError, ImportError):
            pass
        event_id_to_span_context.clear()

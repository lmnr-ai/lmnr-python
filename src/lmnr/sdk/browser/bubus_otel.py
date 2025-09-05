from typing import Collection

from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.opentelemetry_lib.tracing.context import get_current_context

from opentelemetry import context as context_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_current_span
from wrapt import wrap_function_wrapper

from lmnr.sdk.log import get_default_logger

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
    if not TracerWrapper.verify_initialized():
        return await wrapped(*args, **kwargs)
    wrapper = None
    context = None
    context_token = None
    try:
        wrapper = TracerWrapper()
        context = wrapper.push_raw_span_context(span_context)
        # Some auto-instrumentations are not under our control, so they
        # don't have access to our isolated context. We attach the context
        # to the OTEL global context, so that spans know their parent
        # span and trace_id.
        context_token = context_api.attach(context)
    except Exception as e:
        logger.debug("Error pushing span context: %s", e)
    try:
        return await wrapped(*args, **kwargs)
    finally:
        if context_token:
            context_api.detach(context_token)
        if wrapper and context:
            wrapper.pop_span_context()


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

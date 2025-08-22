"""OpenTelemetry OpenHands AI instrumentation"""

import sys
from typing import Collection

from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SESSION_ID,
    USER_ID,
)
from lmnr.opentelemetry_lib.utils.wrappers import _with_tracer_wrapper
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import get_input_from_func_args
from lmnr import Laminar
from lmnr.version import __version__

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer, Tracer
from wrapt import wrap_function_wrapper

logger = get_default_logger(__name__)

_instruments = ("openhands-ai >= 0.9.0", "openhands-aci >= 0.1.0")
parent_spans = {}


WRAPPED_METHODS = [
    {
        "package": "openhands.agenthub.browsing_agent.browsing_agent",
        "object": "BrowsingAgent",
        "methods": [
            {"method": "step"},
        ],
    },
    {
        "package": "openhands.agenthub.codeact_agent.codeact_agent",
        "object": "CodeActAgent",
        "methods": [
            {"method": "step"},
            {"method": "response_to_actions"},
        ],
    },
    {
        "package": "openhands.agenthub.loc_agent.loc_agent",
        "object": "LocAgent",
        "methods": [
            {"method": "response_to_actions"},
        ],
    },
    {
        "package": "openhands.agenthub.readonly_agent.readonly_agent",
        "object": "ReadOnlyAgent",
        "methods": [
            {"method": "step"},
            {"method": "response_to_actions"},
        ],
    },
    {
        "package": "openhands.agenthub.visualbrowsing_agent.visualbrowsing_agent",
        "object": "VisualBrowsingAgent",
        "methods": [
            {"method": "step"},
        ],
    },
    {
        "package": "openhands.controller.agent_controller",
        "object": "AgentController",
        "methods": [
            {"method": "step"},
            {"method": "_handle_action", "async": True},
            {"method": "_handle_observation", "async": True},
            {"method": "_handle_message_action"},
            {"method": "on_event"},
            {"method": "save_state"},
            {"method": "get_trajectory"},
            {"method": "start_delegate"},
            {"method": "end_delegate"},
            {"method": "_is_stuck"},
        ],
    },
    {
        "package": "openhands.runtime.browser.browser_env",
        "object": "BrowserEnv",
        "methods": [
            {"method": "step"},
            {"method": "init_browser"},
            {"method": "browser_process"},
            {"method": "check_alive"},
            {"method": "close"},
        ],
    },
]


@_with_tracer_wrapper
def _wrap_on_event(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    """Wrapper for on_event."""
    id = instance.id
    user_id = instance.user_id
    event = kwargs.get("event", args[0] if len(args) > 0 else None)
    start_event = False
    finish_event = False
    if event and hasattr(event, "action") and event.action == "system":
        return wrapped(*args, **kwargs)

    event_type = None
    subtype = None
    if event and hasattr(event, "action"):
        event_type = "action"
        try:
            subtype = event.action.value
        except Exception:
            subtype = event.action
    elif event and hasattr(event, "observation"):
        event_type = "observation"
        try:
            subtype = event.observation.value
        except Exception:
            subtype = event.observation
    if event_type and subtype:
        to_wrap["span_name"] = f"{event_type}.{subtype}"

    # start trace on user message
    if (
        event
        and hasattr(event, "action")
        and event.action == "message"
        and hasattr(event, "source")
        and event.source == "user"
    ):
        start_event = True
    # end trace on agent state change to finished or error
    if (
        event
        and hasattr(event, "observation")
        and event.observation == "agent_state_changed"
        and hasattr(event, "agent_state")
        and event.agent_state in ["stopped", "awaiting_user_input"]
    ):
        finish_event = True

    if start_event:
        if id in parent_spans:
            logger.debug(
                "Received a message, but already have a span for this trace. Resetting span."
            )
            parent_spans[id].end()
            del parent_spans[id]
        parent_span = Laminar.start_span("conversation.turn", span_type="DEFAULT")
        if user_id:
            parent_span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{USER_ID}", user_id)
        parent_span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{SESSION_ID}", id)
        parent_spans[id] = parent_span

    if id in parent_spans:
        with Laminar.use_span(parent_spans[id]):
            result = _wrap_sync_method_inner(
                tracer, to_wrap, wrapped, instance, args, kwargs
            )
            if finish_event:
                parent_spans[id].end()
                del parent_spans[id]
            return result

    return wrapped(*args, **kwargs)


@_with_tracer_wrapper
def _wrap_handle_action(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    """Wrapper for on_event."""
    event = kwargs.get("event", args[0] if len(args) > 0 else None)
    if event and hasattr(event, "action"):
        if event.action == "system":
            return wrapped(*args, **kwargs)
    id = instance.id
    if id not in parent_spans:
        return wrapped(*args, **kwargs)
    return _wrap_sync_method_inner(tracer, to_wrap, wrapped, instance, args, kwargs)


def _wrap_sync_method_inner(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    """Wrapper for synchronous methods."""
    span_name = to_wrap.get("span_name")

    with Laminar.start_as_current_span(
        span_name,
        span_type=to_wrap.get("span_type", "DEFAULT"),
        input=json_dumps(get_input_from_func_args(wrapped, True, args, kwargs)),
    ) as span:
        try:
            result = wrapped(*args, **kwargs)

            # Capture output
            span.set_attribute("lmnr.span.output", json_dumps(result))
            return result

        except Exception as e:
            span.record_exception(e)
            raise


@_with_tracer_wrapper
def _wrap_sync_method(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    id = None
    if to_wrap.get("object") == "AgentController":
        id = instance.id
    if to_wrap.get("object") == "ActionExecutionClient" and hasattr(instance, "sid"):
        id = instance.sid
    if id is not None and id not in parent_spans:
        return wrapped(*args, **kwargs)
    return _wrap_sync_method_inner(tracer, to_wrap, wrapped, instance, args, kwargs)


@_with_tracer_wrapper
async def _wrap_async_method(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    """Wrapper for asynchronous methods."""
    span_name = to_wrap.get("span_name")
    id = None
    if to_wrap.get("object") == "AgentController":
        id = instance.id
    if to_wrap.get("object") == "ActionExecutionClient" and hasattr(instance, "sid"):
        id = instance.sid
    if id is not None and id not in parent_spans:
        return await wrapped(*args, **kwargs)

    with Laminar.start_as_current_span(
        span_name,
        span_type=to_wrap.get("span_type", "DEFAULT"),
        input=json_dumps(
            get_input_from_func_args(
                wrapped, to_wrap.get("object") is not None, args, kwargs
            )
        ),
    ) as span:
        try:
            result = await wrapped(*args, **kwargs)

            # Capture output
            span.set_attribute("lmnr.span.output", json_dumps(result))
            return result

        except Exception as e:
            span.record_exception(e)
            raise


class OpenHandsInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenHands AI."""

    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        """Instrument OpenHands AI methods."""
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for wrapped_config in WRAPPED_METHODS:
            wrap_package = wrapped_config.get("package")

            wrap_object = wrapped_config.get("object")
            methods = wrapped_config.get("methods", [])

            for method_config in methods:

                wrap_method = method_config.get("method")
                async_wrap = method_config.get("async", False)
                windows_only = method_config.get("windows_only", False)
                if windows_only and sys.platform != "win32":
                    continue

                # Create the method configuration for the wrapper
                method_wrapper_config = {
                    "package": wrap_package,
                    "object": wrap_object,
                    "method": wrap_method,
                    "span_name": method_config.get(
                        "span_name",
                        f"{wrap_object}.{wrap_method}" if wrap_object else wrap_method,
                    ),
                    "span_type": method_config.get("span_type", "DEFAULT"),
                    "async": async_wrap,
                }

                # Determine the target for wrapping
                if wrap_object:
                    target = f"{wrap_object}.{wrap_method}"
                else:
                    target = wrap_method

                if wrap_object == "AgentController" and wrap_method == "on_event":
                    wrap_function_wrapper(
                        wrap_package,
                        target,
                        _wrap_on_event(tracer, method_wrapper_config),
                    )
                    continue
                if wrap_object == "AgentController" and wrap_method == "_handle_action":
                    wrap_function_wrapper(
                        wrap_package,
                        target,
                        _wrap_handle_action(tracer, method_wrapper_config),
                    )
                    continue

                try:
                    if async_wrap:
                        wrap_function_wrapper(
                            wrap_package,
                            target,
                            _wrap_async_method(tracer, method_wrapper_config),
                        )
                    else:
                        wrap_function_wrapper(
                            wrap_package,
                            target,
                            _wrap_sync_method(tracer, method_wrapper_config),
                        )
                except (ModuleNotFoundError, AttributeError) as e:
                    logger.debug(f"Could not instrument {wrap_package}.{target}: {e}")

    def _uninstrument(self, **kwargs):
        """Remove OpenHands AI instrumentation."""
        for wrapped_config in WRAPPED_METHODS:
            wrap_package = wrapped_config.get("package")
            wrap_object = wrapped_config.get("object")
            methods = wrapped_config.get("methods", [])

            for method_config in methods:
                wrap_method = method_config.get("method")

                # Determine the module path for unwrapping
                if wrap_object:
                    module_path = f"{wrap_package}.{wrap_object}"
                else:
                    module_path = wrap_package

                try:
                    unwrap(module_path, wrap_method)
                except (AttributeError, ValueError) as e:
                    logger.debug(
                        f"Could not uninstrument {module_path}.{wrap_method}: {e}"
                    )

"""OpenTelemetry Kernel instrumentation"""

import functools
from typing import Collection

from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.kernel.utils import (
    process_tool_output_formatter,
    screenshot_tool_output_formatter,
)
from lmnr.sdk.decorators import observe
from lmnr.sdk.utils import get_input_from_func_args, is_async
from lmnr import Laminar
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

_instruments = ("kernel >= 0.2.0",)


WRAPPED_METHODS = [
    {
        "package": "kernel.resources.browsers",
        "object": "BrowsersResource",
        "method": "create",
        "class_name": "Browser",
    },
    {
        "package": "kernel.resources.browsers",
        "object": "BrowsersResource",
        "method": "retrieve",
        "class_name": "Browser",
    },
    {
        "package": "kernel.resources.browsers",
        "object": "BrowsersResource",
        "method": "list",
        "class_name": "Browser",
    },
    {
        "package": "kernel.resources.browsers",
        "object": "BrowsersResource",
        "method": "delete",
        "class_name": "Browser",
    },
    {
        "package": "kernel.resources.browsers",
        "object": "BrowsersResource",
        "method": "delete_by_id",
        "class_name": "Browser",
    },
    {
        "package": "kernel.resources.browsers",
        "object": "BrowsersResource",
        "method": "load_extensions",
        "class_name": "Browser",
    },
    {
        "package": "kernel.resources.browsers.computer",
        "object": "ComputerResource",
        "method": "capture_screenshot",
        "class_name": "Computer",
        "span_type": "TOOL",
        "output_formatter": screenshot_tool_output_formatter,
    },
    {
        "package": "kernel.resources.browsers.computer",
        "object": "ComputerResource",
        "method": "click_mouse",
        "class_name": "Computer",
        "span_type": "TOOL",
    },
    {
        "package": "kernel.resources.browsers.computer",
        "object": "ComputerResource",
        "method": "drag_mouse",
        "class_name": "Computer",
        "span_type": "TOOL",
    },
    {
        "package": "kernel.resources.browsers.computer",
        "object": "ComputerResource",
        "method": "move_mouse",
        "class_name": "Computer",
        "span_type": "TOOL",
    },
    {
        "package": "kernel.resources.browsers.computer",
        "object": "ComputerResource",
        "method": "press_key",
        "class_name": "Computer",
        "span_type": "TOOL",
    },
    {
        "package": "kernel.resources.browsers.computer",
        "object": "ComputerResource",
        "method": "scroll",
        "class_name": "Computer",
        "span_type": "TOOL",
    },
    {
        "package": "kernel.resources.browsers.computer",
        "object": "ComputerResource",
        "method": "type_text",
        "class_name": "Computer",
        "span_type": "TOOL",
    },
    {
        "package": "kernel.resources.browsers.playwright",
        "object": "PlaywrightResource",
        "method": "execute",
        "class_name": "Playwright",
    },
    {
        "package": "kernel.resources.browsers.process",
        "object": "ProcessResource",
        "method": "exec",
        "class_name": "Process",
        "span_type": "TOOL",
        "output_formatter": process_tool_output_formatter,
    },
    {
        "package": "kernel.resources.browsers.process",
        "object": "ProcessResource",
        "method": "kill",
        "class_name": "Process",
        "span_type": "TOOL",
        "output_formatter": process_tool_output_formatter,
    },
    {
        "package": "kernel.resources.browsers.process",
        "object": "ProcessResource",
        "method": "spawn",
        "class_name": "Process",
        "span_type": "TOOL",
        "output_formatter": process_tool_output_formatter,
    },
    {
        "package": "kernel.resources.browsers.process",
        "object": "ProcessResource",
        "method": "status",
        "class_name": "Process",
        "span_type": "TOOL",
        "output_formatter": process_tool_output_formatter,
    },
    {
        "package": "kernel.resources.browsers.process",
        "object": "ProcessResource",
        "method": "stdin",
        "class_name": "Process",
        "span_type": "TOOL",
        "output_formatter": process_tool_output_formatter,
    },
    {
        "package": "kernel.resources.browsers.process",
        "object": "ProcessResource",
        "method": "stdout_stream",
        "class_name": "Process",
        "span_type": "TOOL",
        "output_formatter": process_tool_output_formatter,
    },
]


def _with_wrapper(func):
    """Helper for providing tracer for wrapper functions. Includes metric collectors."""

    def wrapper(
        to_wrap,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return wrapper


@_with_wrapper
def _wrap(
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    with Laminar.start_as_current_span(
        f"{to_wrap.get('class_name')}.{to_wrap.get('method')}",
        span_type=to_wrap.get("span_type", "DEFAULT"),
    ) as span:
        input_kv = get_input_from_func_args(wrapped, True, args, kwargs)
        if "id" in input_kv:
            input_kv["session_id"] = input_kv.get("id")
            input_kv.pop("id")
        span.set_attribute(
            "lmnr.span.input",
            json_dumps(input_kv),
        )
        try:
            result = wrapped(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            raise
        output_formatter = to_wrap.get("output_formatter") or (lambda x: json_dumps(x))
        span.set_attribute("lmnr.span.output", output_formatter(result))
        return result


@_with_wrapper
async def _wrap_async(
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    with Laminar.start_as_current_span(
        f"{to_wrap.get('class_name')}.{to_wrap.get('method')}",
        span_type=to_wrap.get("span_type", "DEFAULT"),
    ) as span:
        input_kv = get_input_from_func_args(wrapped, True, args, kwargs)
        if "id" in input_kv:
            input_kv["session_id"] = input_kv.get("id")
            input_kv.pop("id")
        span.set_attribute(
            "lmnr.span.input",
            json_dumps(input_kv),
        )
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            raise
        output_formatter = to_wrap.get("output_formatter") or (lambda x: json_dumps(x))
        span.set_attribute("lmnr.span.output", output_formatter(result))
        return result


@_with_wrapper
def _wrap_app_action(
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """
    Wraps app.action() decorator factory to add tracing to action handlers.

    wrapped: the original `action` method
    args: (name,) - the action name
    kwargs: potentially {'name': ...}

    Returns a decorator that wraps handlers with tracing before registering them.
    """

    # Call the original action method to get the decorator
    original_decorator = wrapped(*args, **kwargs)

    # Get the action name from args
    action_name = args[0] if args else kwargs.get("name", "unknown")

    # Create a wrapper for the decorator that intercepts the handler
    def tracing_decorator(handler):
        # Apply the observe decorator to add tracing
        observed_handler = observe(
            name=f"action.{action_name}",
            span_type="DEFAULT",
        )(handler)

        # Create an additional wrapper to add post-execution logic
        if is_async(handler):

            @functools.wraps(handler)
            async def async_wrapper_with_flush(*handler_args, **handler_kwargs):
                # Execute the observed handler (tracing happens here)
                result = await observed_handler(*handler_args, **handler_kwargs)

                Laminar.flush()

                return result

            # Register the wrapper with the original decorator
            return original_decorator(async_wrapper_with_flush)
        else:

            @functools.wraps(handler)
            def sync_wrapper_with_flush(*handler_args, **handler_kwargs):
                # Execute the observed handler (tracing happens here)
                result = observed_handler(*handler_args, **handler_kwargs)

                Laminar.flush()

                return result

            # Register the wrapper with the original decorator
            return original_decorator(sync_wrapper_with_flush)

    return tracing_decorator


class KernelInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(wrapped_method),
                )
            except (ModuleNotFoundError, AttributeError):
                pass  # that's ok, we don't want to fail if some methods do not exist

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = f"Async{wrapped_method.get('object')}"
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap_async(wrapped_method),
                )
            except (ModuleNotFoundError, AttributeError):
                pass  # that's ok, we don't want to fail if some methods do not exist

        try:
            wrap_function_wrapper(
                "kernel.app_framework",
                "KernelApp.action",
                _wrap_app_action({}),
            )
        except (ModuleNotFoundError, AttributeError):
            pass

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            try:
                unwrap(
                    f"{wrap_package}.{wrap_object}",
                    wrapped_method.get("method"),
                )
            except (ModuleNotFoundError, AttributeError):
                pass  # that's ok, we don't want to fail if some methods do not exist

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = f"Async{wrapped_method.get('object')}"
            try:
                unwrap(
                    f"{wrap_package}.{wrap_object}",
                    wrapped_method.get("method"),
                )
            except (ModuleNotFoundError, AttributeError):
                pass  # that's ok, we don't want to fail if some methods do not exist

        try:
            unwrap("kernel.app_framework.KernelApp", "action")
        except (ModuleNotFoundError, AttributeError):
            pass

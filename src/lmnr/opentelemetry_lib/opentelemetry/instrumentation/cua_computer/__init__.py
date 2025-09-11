"""OpenTelemetry CUA instrumentation"""

import logging
from typing import Collection

from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.sdk.utils import get_input_from_func_args
from lmnr import Laminar
from lmnr.opentelemetry_lib.tracing.context import get_current_context
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry import trace
from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

from .utils import payload_to_placeholder

logger = logging.getLogger(__name__)

_instruments = ("cua-computer >= 0.4.0",)


WRAPPED_METHODS = [
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "close",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "force_close",
    },
]
WRAPPED_AMETHODS = [
    {
        "package": "computer.computer",
        "object": "Computer",
        "method": "__aenter__",
        "action": "start_parent_span",
    },
    {
        "package": "computer.computer",
        "object": "Computer",
        "method": "__aexit__",
        "action": "end_parent_span",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "mouse_down",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "mouse_up",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "left_click",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "right_click",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "double_click",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "move_cursor",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "drag_to",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "drag",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "key_down",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "key_up",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "type_text",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "press",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "hotkey",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "scroll",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "scroll_down",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "scroll_up",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "screenshot",
        "output_formatter": payload_to_placeholder,
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "get_screen_size",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "get_cursor_position",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "copy_to_clipboard",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "set_clipboard",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "file_exists",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "directory_exists",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "list_dir",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "read_text",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "write_text",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "read_bytes",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "write_bytes",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "delete_file",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "create_dir",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "delete_dir",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "get_file_size",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "run_command",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "get_accessibility_tree",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "to_screen_coordinates",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "get_active_window_bounds",
    },
    {
        "package": "computer.interface.generic",
        "object": "GenericComputerInterface",
        "method": "to_screenshot_coordinates",
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


def add_input_to_parent_span(span, instance):
    # api_key is skipped on purpose
    params = {}
    if hasattr(instance, "display"):
        params["display"] = instance.display
    if hasattr(instance, "memory"):
        params["memory"] = instance.memory
    if hasattr(instance, "cpu"):
        params["cpu"] = instance.cpu
    if hasattr(instance, "os_type"):
        params["os_type"] = instance.os_type
    if hasattr(instance, "name"):
        params["name"] = instance.name
    if hasattr(instance, "image"):
        params["image"] = instance.image
    if hasattr(instance, "shared_directories"):
        params["shared_directories"] = instance.shared_directories
    if hasattr(instance, "use_host_computer_server"):
        params["use_host_computer_server"] = instance.use_host_computer_server
    if hasattr(instance, "verbosity"):
        if (
            isinstance(instance.verbosity, int)
            and instance.verbosity in logging._levelToName
        ):
            params["verbosity"] = logging._levelToName[instance.verbosity]
        else:
            params["verbosity"] = instance.verbosity
    if hasattr(instance, "telemetry_enabled"):
        params["telemetry_enabled"] = instance.telemetry_enabled
    if hasattr(instance, "provider_type"):
        params["provider_type"] = instance.provider_type
    if hasattr(instance, "port"):
        params["port"] = instance.port
    if hasattr(instance, "noVNC_port"):
        params["noVNC_port"] = instance.noVNC_port
    if hasattr(instance, "host"):
        params["host"] = instance.host
    if hasattr(instance, "storage"):
        params["storage"] = instance.storage
    if hasattr(instance, "ephemeral"):
        params["ephemeral"] = instance.ephemeral
    if hasattr(instance, "experiments"):
        params["experiments"] = instance.experiments
    span.set_attribute("lmnr.span.input", json_dumps(params))


@_with_wrapper
def _wrap(
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    if to_wrap.get("action") == "start_parent_span":
        parent_span = Laminar.start_span("computer.run")
        add_input_to_parent_span(parent_span, instance)
        result = wrapped(*args, **kwargs)
        try:
            instance._interface._lmnr_parent_span = parent_span
        except Exception:
            pass
        return result
    elif to_wrap.get("action") == "end_parent_span":
        result = wrapped(*args, **kwargs)
        try:
            parent_span: Span = instance._interface._lmnr_parent_span
            if parent_span and parent_span.is_recording():
                parent_span.end()
        except Exception:
            pass
        return result

    # if there's no parent span, use
    parent_span = trace.get_current_span(context=get_current_context())
    try:
        if instance._lmnr_parent_span:
            parent_span: Span = instance._lmnr_parent_span
    except Exception:
        pass

    with Laminar.use_span(parent_span):
        instance_name = "interface"
        with Laminar.start_as_current_span(
            f"{instance_name}.{to_wrap.get('method')}", span_type="TOOL"
        ) as span:
            span.set_attribute(
                "lmnr.span.input",
                json_dumps(get_input_from_func_args(wrapped, True, args, kwargs)),
            )
            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
                span.end()
                raise
            output_formatter = to_wrap.get("output_formatter") or (
                lambda x: json_dumps(x)
            )
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
    if to_wrap.get("action") == "start_parent_span":
        parent_span = Laminar.start_span("computer.run")
        add_input_to_parent_span(parent_span, instance)
        result = await wrapped(*args, **kwargs)
        try:
            instance._interface._lmnr_parent_span = parent_span
        except Exception:
            pass
        return result
    elif to_wrap.get("action") == "end_parent_span":
        result = await wrapped(*args, **kwargs)
        try:
            parent_span: Span = instance._interface._lmnr_parent_span
            if parent_span and parent_span.is_recording():
                parent_span.end()
        except Exception:
            pass
        return result

    # if there's no parent span, use
    parent_span = trace.get_current_span(context=get_current_context())
    try:
        parent_span: Span = instance._lmnr_parent_span
    except Exception:
        pass

    with Laminar.use_span(parent_span):
        instance_name = "interface"
        with Laminar.start_as_current_span(
            f"{instance_name}.{to_wrap.get('method')}",
            span_type="TOOL",
        ) as span:
            span.set_attribute(
                "lmnr.span.input",
                json_dumps(get_input_from_func_args(wrapped, True, args, kwargs)),
            )
            try:
                result = await wrapped(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
                span.end()
                raise
            output_formatter = to_wrap.get("output_formatter") or (
                lambda x: json_dumps(x)
            )
            span.set_attribute("lmnr.span.output", output_formatter(result))
            return result


class CuaComputerInstrumentor(BaseInstrumentor):
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
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some methods do not exist

        for wrapped_method in WRAPPED_AMETHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap_async(wrapped_method),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some methods do not exist

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            try:
                unwrap(
                    f"{wrap_package}.{wrap_object}",
                    wrapped_method.get("method"),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some methods do not exist

        for wrapped_method in WRAPPED_AMETHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            try:
                unwrap(
                    f"{wrap_package}.{wrap_object}",
                    wrapped_method.get("method"),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some methods do not exist

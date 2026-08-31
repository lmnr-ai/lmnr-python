"""OpenTelemetry CUA instrumentation"""

import logging
from importlib.metadata import version
from typing import Any, Callable, Collection, Sequence

from lmnr.sdk.utils import get_input_from_func_args, json_dumps
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

from opentelemetry import trace
from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode

from .utils import payload_to_placeholder

logger = logging.getLogger(__name__)

_instruments = ("cua-computer >= 0.4.0",)


class CuaComputerSpec(WrappedFunctionSpec, total=False):
    """cua-computer's per-method extras.

    `action` tags the two lifecycle methods that open/close the parent
    `computer.run` span rather than tracing a call; `output_formatter` replaces
    a large payload (e.g. a screenshot) with a placeholder before it is recorded.
    """

    action: str
    output_formatter: Callable[[Any], Any]




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


def _wrap(
    to_wrap: CuaComputerSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
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
            f"{instance_name}.{to_wrap['method_name']}", span_type="TOOL"
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


async def _wrap_async(
    to_wrap: CuaComputerSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
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
            f"{instance_name}.{to_wrap['method_name']}",
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


WRAPPED_FUNCTIONS: list[CuaComputerSpec] = [
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="close",
        is_async=False,
        wrapper_function=_wrap,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="force_close",
        is_async=False,
        wrapper_function=_wrap,
    ),
    CuaComputerSpec(
        package_name="computer.computer",
        object_name="Computer",
        method_name="__aenter__",
        is_async=True,
        action="start_parent_span",
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.computer",
        object_name="Computer",
        method_name="__aexit__",
        is_async=True,
        action="end_parent_span",
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="mouse_down",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="mouse_up",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="left_click",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="right_click",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="double_click",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="move_cursor",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="drag_to",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="drag",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="key_down",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="key_up",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="type_text",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="press",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="hotkey",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="scroll",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="scroll_down",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="scroll_up",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="screenshot",
        is_async=True,
        output_formatter=payload_to_placeholder,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="get_screen_size",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="get_cursor_position",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="copy_to_clipboard",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="set_clipboard",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="file_exists",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="directory_exists",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="list_dir",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="read_text",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="write_text",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="read_bytes",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="write_bytes",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="delete_file",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="create_dir",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="delete_dir",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="get_file_size",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="run_command",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="get_accessibility_tree",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="to_screen_coordinates",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="get_active_window_bounds",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
    CuaComputerSpec(
        package_name="computer.interface.generic",
        object_name="GenericComputerInterface",
        method_name="to_screenshot_coordinates",
        is_async=True,
        wrapper_function=_wrap_async,
    ),
]


class CuaComputerInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                cua_version = version("cua-computer")
            except Exception as e:
                logger.debug(f"Failed to get cua-computer version {e}")
                cua_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="cua-computer",
                version=cua_version,
            )
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

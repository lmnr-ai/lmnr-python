"""OpenTelemetry Kernel instrumentation"""

import functools
import logging
from importlib.metadata import version
from typing import Any, Callable, Collection, Sequence

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.kernel.utils import (
    process_tool_output_formatter,
    screenshot_tool_output_formatter,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    stamp_instrumentation_scope,
)
from lmnr.sdk.decorators import observe
from lmnr.sdk.utils import get_input_from_func_args, is_async, json_dumps
from lmnr import Laminar

from opentelemetry.trace.status import Status, StatusCode

logger = logging.getLogger(__name__)

_instruments = ("kernel >= 0.2.0",)


class KernelSpec(WrappedFunctionSpec, total=False):
    """kernel's per-method extras.

    `class_name` names the span after the user-facing resource (e.g. "Browser")
    rather than the SDK class; `output_formatter` condenses a large tool result
    (a screenshot, a process payload) before it is recorded.
    """

    class_name: str
    output_formatter: Callable[[Any], Any]




def _wrap(
    to_wrap: KernelSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    with Laminar.start_as_current_span(
        f"{to_wrap.get('class_name')}.{to_wrap['method_name']}",
        span_type=to_wrap.get("span_type", "DEFAULT"),
    ) as span:
        stamp_instrumentation_scope(span, to_wrap)
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


async def _wrap_async(
    to_wrap: KernelSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    with Laminar.start_as_current_span(
        f"{to_wrap.get('class_name')}.{to_wrap['method_name']}",
        span_type=to_wrap.get("span_type", "DEFAULT"),
    ) as span:
        stamp_instrumentation_scope(span, to_wrap)
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


def _wrap_app_action(
    to_wrap: KernelSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
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


WRAPPED_FUNCTIONS: list[KernelSpec] = [
    KernelSpec(
        package_name="kernel.resources.browsers",
        object_name="BrowsersResource",
        method_name="create",
        is_async=False,
        class_name="Browser",
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers",
        object_name="BrowsersResource",
        method_name="retrieve",
        is_async=False,
        class_name="Browser",
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers",
        object_name="BrowsersResource",
        method_name="list",
        is_async=False,
        class_name="Browser",
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers",
        object_name="BrowsersResource",
        method_name="delete",
        is_async=False,
        class_name="Browser",
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers",
        object_name="BrowsersResource",
        method_name="delete_by_id",
        is_async=False,
        class_name="Browser",
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers",
        object_name="BrowsersResource",
        method_name="load_extensions",
        is_async=False,
        class_name="Browser",
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="ComputerResource",
        method_name="capture_screenshot",
        is_async=False,
        class_name="Computer",
        span_type="TOOL",
        output_formatter=screenshot_tool_output_formatter,
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="ComputerResource",
        method_name="click_mouse",
        is_async=False,
        class_name="Computer",
        span_type="TOOL",
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="ComputerResource",
        method_name="drag_mouse",
        is_async=False,
        class_name="Computer",
        span_type="TOOL",
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="ComputerResource",
        method_name="move_mouse",
        is_async=False,
        class_name="Computer",
        span_type="TOOL",
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="ComputerResource",
        method_name="press_key",
        is_async=False,
        class_name="Computer",
        span_type="TOOL",
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="ComputerResource",
        method_name="scroll",
        is_async=False,
        class_name="Computer",
        span_type="TOOL",
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="ComputerResource",
        method_name="type_text",
        is_async=False,
        class_name="Computer",
        span_type="TOOL",
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.playwright",
        object_name="PlaywrightResource",
        method_name="execute",
        is_async=False,
        class_name="Playwright",
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.process",
        object_name="ProcessResource",
        method_name="exec",
        is_async=False,
        class_name="Process",
        span_type="TOOL",
        output_formatter=process_tool_output_formatter,
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.process",
        object_name="ProcessResource",
        method_name="kill",
        is_async=False,
        class_name="Process",
        span_type="TOOL",
        output_formatter=process_tool_output_formatter,
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.process",
        object_name="ProcessResource",
        method_name="spawn",
        is_async=False,
        class_name="Process",
        span_type="TOOL",
        output_formatter=process_tool_output_formatter,
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.process",
        object_name="ProcessResource",
        method_name="status",
        is_async=False,
        class_name="Process",
        span_type="TOOL",
        output_formatter=process_tool_output_formatter,
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.process",
        object_name="ProcessResource",
        method_name="stdin",
        is_async=False,
        class_name="Process",
        span_type="TOOL",
        output_formatter=process_tool_output_formatter,
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.process",
        object_name="ProcessResource",
        method_name="stdout_stream",
        is_async=False,
        class_name="Process",
        span_type="TOOL",
        output_formatter=process_tool_output_formatter,
        wrapper_function=_wrap,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers",
        object_name="AsyncBrowsersResource",
        method_name="create",
        is_async=True,
        class_name="Browser",
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers",
        object_name="AsyncBrowsersResource",
        method_name="retrieve",
        is_async=True,
        class_name="Browser",
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers",
        object_name="AsyncBrowsersResource",
        method_name="list",
        is_async=True,
        class_name="Browser",
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers",
        object_name="AsyncBrowsersResource",
        method_name="delete",
        is_async=True,
        class_name="Browser",
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers",
        object_name="AsyncBrowsersResource",
        method_name="delete_by_id",
        is_async=True,
        class_name="Browser",
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers",
        object_name="AsyncBrowsersResource",
        method_name="load_extensions",
        is_async=True,
        class_name="Browser",
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="AsyncComputerResource",
        method_name="capture_screenshot",
        is_async=True,
        class_name="Computer",
        span_type="TOOL",
        output_formatter=screenshot_tool_output_formatter,
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="AsyncComputerResource",
        method_name="click_mouse",
        is_async=True,
        class_name="Computer",
        span_type="TOOL",
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="AsyncComputerResource",
        method_name="drag_mouse",
        is_async=True,
        class_name="Computer",
        span_type="TOOL",
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="AsyncComputerResource",
        method_name="move_mouse",
        is_async=True,
        class_name="Computer",
        span_type="TOOL",
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="AsyncComputerResource",
        method_name="press_key",
        is_async=True,
        class_name="Computer",
        span_type="TOOL",
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="AsyncComputerResource",
        method_name="scroll",
        is_async=True,
        class_name="Computer",
        span_type="TOOL",
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.computer",
        object_name="AsyncComputerResource",
        method_name="type_text",
        is_async=True,
        class_name="Computer",
        span_type="TOOL",
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.playwright",
        object_name="AsyncPlaywrightResource",
        method_name="execute",
        is_async=True,
        class_name="Playwright",
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.process",
        object_name="AsyncProcessResource",
        method_name="exec",
        is_async=True,
        class_name="Process",
        span_type="TOOL",
        output_formatter=process_tool_output_formatter,
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.process",
        object_name="AsyncProcessResource",
        method_name="kill",
        is_async=True,
        class_name="Process",
        span_type="TOOL",
        output_formatter=process_tool_output_formatter,
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.process",
        object_name="AsyncProcessResource",
        method_name="spawn",
        is_async=True,
        class_name="Process",
        span_type="TOOL",
        output_formatter=process_tool_output_formatter,
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.process",
        object_name="AsyncProcessResource",
        method_name="status",
        is_async=True,
        class_name="Process",
        span_type="TOOL",
        output_formatter=process_tool_output_formatter,
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.process",
        object_name="AsyncProcessResource",
        method_name="stdin",
        is_async=True,
        class_name="Process",
        span_type="TOOL",
        output_formatter=process_tool_output_formatter,
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.resources.browsers.process",
        object_name="AsyncProcessResource",
        method_name="stdout_stream",
        is_async=True,
        class_name="Process",
        span_type="TOOL",
        output_formatter=process_tool_output_formatter,
        wrapper_function=_wrap_async,
    ),
    KernelSpec(
        package_name="kernel.app_framework",
        object_name="KernelApp",
        method_name="action",
        is_async=False,
        wrapper_function=_wrap_app_action,
    ),
]


class KernelInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                kernel_version = version("kernel")
            except Exception as e:
                logger.debug(f"Failed to get kernel version {e}")
                kernel_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="kernel",
                version=kernel_version,
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

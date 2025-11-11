from lmnr import Laminar
from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.sdk.utils import get_input_from_func_args
from opentelemetry.trace import Status, StatusCode
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from typing import Collection
from wrapt import wrap_function_wrapper

# TODO: confirm if correct
_instruments = ("claude-agent-sdk >= 0.1.0",)

WRAPPED_METHODS = [
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "query",
        "class_name": "ClaudeSDKClient",
    },
    {
        "package": "claude_agent_sdk.query",
        "object": "",
        "method": "query",
        "class_name": "",
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
        f"{to_wrap.get('class_name')}.{to_wrap.get('method')}" if to_wrap.get('class_name') else to_wrap.get('method'),
        span_type=to_wrap.get("span_type", "DEFAULT"),
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
            raise
        # TODO: 
        output_formatter = to_wrap.get("output_formatter") or (lambda x: json_dumps(x))
        span.set_attribute("lmnr.span.output", output_formatter(result))
        return result

class ClaudeAgentInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            
            wrap_name = f"{wrap_object}.{wrap_method}" if wrap_object else wrap_method
            try:
                wrap_function_wrapper(
                    wrap_package,
                    wrap_name,
                    _wrap(wrapped_method),
                )
            except (ModuleNotFoundError, AttributeError) as e:
                pass  # that's ok, we don't want to fail if some methods do not exist

        # TODO: add async wrapping here

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                unwrap(f"{wrap_package}.{wrap_object}", wrap_method)
            except (ModuleNotFoundError, AttributeError):
                pass  # that's ok, we don't want to fail if some methods do not exist

from lmnr.openllmetry_sdk.decorators.base import json_dumps
from lmnr.sdk.browser.utils import with_tracer_wrapper
from lmnr.sdk.utils import get_input_from_func_args
from lmnr.version import __version__

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer, Tracer
from typing import Collection
from wrapt import wrap_function_wrapper

_instruments = ("browser-use >= 0.1.0",)

WRAPPED_METHODS = [
    {
        "package": "browser_use.agent.service",
        "object": "Agent",
        "method": "run",
        "span_name": "agent.run",
        "ignore_input": False,
        "ignore_output": True,
        "span_type": "DEFAULT",
    },
    {
        "package": "browser_use.agent.service",
        "object": "Agent",
        "method": "step",
        "span_name": "agent.step",
        "ignore_input": True,
        "ignore_output": True,
        "span_type": "DEFAULT",
    },
    {
        "package": "browser_use.controller.service",
        "object": "Controller",
        "method": "act",
        "span_name": "controller.act",
        "ignore_input": True,
        "ignore_output": False,
        "span_type": "DEFAULT",
    },
    {
        "package": "browser_use.controller.registry.service",
        "object": "Registry",
        "method": "execute_action",
        "ignore_input": False,
        "ignore_output": False,
        "span_type": "TOOL",
    },
]


@with_tracer_wrapper
async def _wrap(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    span_name = to_wrap.get("span_name")
    attributes = {
        "lmnr.span.type": to_wrap.get("span_type"),
    }
    if to_wrap.get("method") == "execute_action":
        span_name = args[0] if len(args) > 0 else kwargs.get("action_name", "action")
        attributes["lmnr.span.input"] = json_dumps(
            {
                "action": span_name,
                "params": args[1] if len(args) > 1 else kwargs.get("params", {}),
            }
        )
    else:
        if not to_wrap.get("ignore_input"):
            attributes["lmnr.span.input"] = json_dumps(
                get_input_from_func_args(wrapped, True, args, kwargs)
            )
    with tracer.start_as_current_span(span_name, attributes=attributes) as span:
        span.set_attributes(attributes)
        result = await wrapped(*args, **kwargs)
        if not to_wrap.get("ignore_output"):
            span.set_attribute("lmnr.span.output", json_dumps(result))
        return result


class BrowserUseInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(
                        tracer,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we're not instrumenting everything

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            unwrap(
                f"{wrap_package}.{wrap_object}" if wrap_object else wrap_package,
                wrap_method,
            )

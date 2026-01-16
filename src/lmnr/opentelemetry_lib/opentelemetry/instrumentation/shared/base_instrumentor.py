from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from .types import LaminarInstrumentorConfig
from .wrapper_helpers import add_spec_wrapper


class BaseLaminarInstrumentor(BaseInstrumentor):
    instrumentor_config: LaminarInstrumentorConfig

    # default implementation, can be overridden by subclasses
    def _instrument(self, **kwargs):
        for wrapped_function_spec in self.instrumentor_config["wrapped_functions"]:
            target = (
                f"{wrapped_function_spec['object_name']}.{wrapped_function_spec['method_name']}"
                if wrapped_function_spec["object_name"]
                else wrapped_function_spec["method_name"]
            )
            if wrapped_function_spec["package_name"] == "litellm":
                print(
                    f" Instrumenting {wrapped_function_spec['package_name']}.{target}"
                )
            wrap_function_wrapper(
                wrapped_function_spec["package_name"],
                target,
                add_spec_wrapper(
                    wrapped_function_spec["wrapper_function"], wrapped_function_spec
                ),
            )

    # default implementation, can be overridden by subclasses
    def _uninstrument(self, **kwargs):
        for wrapped_function_spec in self.instrumentor_config["wrapped_functions"]:
            target = (
                f"{wrapped_function_spec['object_name']}.{wrapped_function_spec['method_name']}"
                if wrapped_function_spec["object_name"]
                else wrapped_function_spec["method_name"]
            )
            unwrap(wrapped_function_spec["package_name"], target)

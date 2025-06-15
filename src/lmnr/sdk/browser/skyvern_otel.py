from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.sdk.browser.utils import with_tracer_wrapper
from lmnr.sdk.utils import get_input_from_func_args
from lmnr.version import __version__

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer, Tracer
from typing import Collection
from wrapt import wrap_function_wrapper
import pydantic

try:
    from skyvern import Skyvern
except ImportError as e:
    raise ImportError(
        f"Attempted to import {__file__}, but it is designed "
        "to patch Skyvern, which is not installed. Use `pip install skyvern` "
        "to install Skyvern or remove this import."
    ) from e

_instruments = ("skyvern >= 0.1.0",)

WRAPPED_METHODS = [
    {
        "package": "skyvern.library.skyvern",
        "object": "Skyvern",
        "method": "run_task",
        "span_name": "Skyvern.run_task",
        "ignore_input": False,
        "ignore_output": False,
        "span_type": "DEFAULT",
    },
    {
        "package": "skyvern.webeye.scraper",
        "method": "get_interactable_element_tree",
        "span_name": "get_interactable_element_tree",
        "ignore_input": False,
        "ignore_output": False,
        "span_type": "DEFAULT",
    },
]


@with_tracer_wrapper
async def _wrap(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    span_name = to_wrap.get("span_name")
    attributes = {
        "lmnr.span.type": to_wrap.get("span_type"),
    }
    
    if not to_wrap.get("ignore_input"):
        attributes["lmnr.span.input"] = json_dumps(
            get_input_from_func_args(wrapped, True, args, kwargs)
        )
    
    with tracer.start_as_current_span(span_name, attributes=attributes) as span:
        span.set_attributes(attributes)
        result = await wrapped(*args, **kwargs)
        if not to_wrap.get("ignore_output"):
            to_serialize = result
            serialized = (
                to_serialize.model_dump_json()
                if isinstance(to_serialize, pydantic.BaseModel)
                else json_dumps(to_serialize)
            )
            span.set_attribute("lmnr.span.output", serialized)
        return result


class SkyvernInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            span_name = wrapped_method.get("span_name")

            try:
                wrap_function_wrapper(
                    wrap_package,
                    span_name,
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
            span_name = wrapped_method.get("span_name")
            wrap_method = wrapped_method.get("method")

            unwrap(
                f"{wrap_package}.{span_name}" if span_name else wrap_package,
                wrap_method,
            )


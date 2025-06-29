from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer

from wrapt import wrap_function_wrapper

from .wrappers import (
    async_responses_cancel_wrapper,
    async_responses_get_or_create_wrapper,
    responses_cancel_wrapper,
    responses_get_or_create_wrapper,
)

_instruments = ("openai >= 1.66.0",)

WRAPPED_METHODS = [
    {
        "module": "openai.resources.responses",
        "class_name": "Responses",
        "method_name": "create",
        "wrapper": responses_get_or_create_wrapper,
    },
    {
        "module": "openai.resources.responses",
        "class_name": "Responses",
        "method_name": "retrieve",
        "wrapper": responses_get_or_create_wrapper,
    },
    {
        "module": "openai.resources.responses",
        "class_name": "Responses",
        "method_name": "cancel",
        "wrapper": responses_cancel_wrapper,
    },
    {
        "module": "openai.resources.responses",
        "class_name": "AsyncResponses",
        "method_name": "create",
        "wrapper": async_responses_get_or_create_wrapper,
    },
    {
        "module": "openai.resources.responses",
        "class_name": "AsyncResponses",
        "method_name": "retrieve",
        "wrapper": async_responses_get_or_create_wrapper,
    },
    {
        "module": "openai.resources.responses",
        "class_name": "AsyncResponses",
        "method_name": "cancel",
        "wrapper": async_responses_cancel_wrapper,
    },
]


class OpenAIResponsesInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, "0.0.1a1", tracer_provider)

        for method in WRAPPED_METHODS:
            try:
                wrap_function_wrapper(
                    method["module"],
                    f"{method['class_name']}.{method['method_name']}",
                    method["wrapper"](tracer),
                )
            except (AttributeError, ModuleNotFoundError):
                pass

    def _uninstrument(self, **kwargs):
        for method in WRAPPED_METHODS:
            try:
                unwrap(
                    method["module"],
                    f"{method['class_name']}.{method['method_name']}",
                )
            except (AttributeError, ModuleNotFoundError):
                pass

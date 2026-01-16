from typing import Collection

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from .wrappers import wrap_completion, awrap_completion, wrap_responses, awrap_responses


_instruments = ("litellm >= 1.0.0",)


class LitellmInstrumentor(BaseLaminarInstrumentor):

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def __init__(self):
        super().__init__()
        self.instrumentor_config = {
            "wrapped_functions": [
                {
                    "package_name": "litellm",
                    "object_name": None,
                    "method_name": "completion",
                    "is_async": False,
                    "is_streaming": False,
                    "span_name": "litellm.completion",
                    "span_type": "LLM",
                    "wrapper_function": wrap_completion,
                },
                {
                    "package_name": "litellm",
                    "object_name": None,
                    "method_name": "acompletion",
                    "is_async": True,
                    "is_streaming": False,
                    "span_name": "litellm.acompletion",
                    "span_type": "LLM",
                    "wrapper_function": awrap_completion,
                },
                {
                    "package_name": "litellm",
                    "object_name": None,
                    "method_name": "responses",
                    "is_async": False,
                    "is_streaming": True,
                    "span_name": "litellm.responses",
                    "span_type": "LLM",
                    "wrapper_function": wrap_responses,
                },
                {
                    "package_name": "litellm",
                    "object_name": None,
                    "method_name": "aresponses",
                    "is_async": True,
                    "is_streaming": True,
                    "span_name": "litellm.aresponses",
                    "span_type": "LLM",
                    "wrapper_function": awrap_responses,
                },
            ]
        }

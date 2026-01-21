from importlib.metadata import version
from typing import Collection

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
)
from .wrappers import wrap_completion, wrap_responses


_instruments = ("litellm >= 1.0.0",)


class LitellmInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        litellm_version = "unknown"
        try:
            litellm_version = version("litellm")
        except Exception:
            pass
        return LaminarInstrumentationScopeAttributes(
            name="litellm",
            version=litellm_version,
        )

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is not None:
            return self._scope
        self._scope = self._instrumentation_scope()
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config = {
            "wrapped_functions": [
                # we are not wrapping `acompletion`, and `aresponses`,
                # because they call `completion` and `responses` internally respectively
                {
                    "package_name": "litellm",
                    "object_name": None,
                    "method_name": "completion",
                    "is_async": False,
                    "is_streaming": False,
                    "span_name": "litellm.completion",
                    "span_type": "LLM",
                    "wrapper_function": wrap_completion,
                    "replace_aliases": True,  # Enable alias replacement for module-level function
                    "instrumentation_scope": self.instrumentation_scope(),
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
                    "replace_aliases": True,  # Enable alias replacement for module-level function
                    "instrumentation_scope": self.instrumentation_scope(),
                },
            ],
        }

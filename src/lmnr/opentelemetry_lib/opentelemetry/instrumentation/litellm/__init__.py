from collections.abc import Collection
from importlib.metadata import version

from typing_extensions import override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.litellm.wrappers import (
    wrap_completion,
    wrap_responses,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.sdk.log import get_default_logger

instruments = ("litellm >= 1.0.0",)
logger = get_default_logger(__name__)


class LitellmInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return instruments

    def _instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        litellm_version = "unknown"
        try:
            litellm_version = version("litellm")
        except Exception as e:
            logger.debug(f"Failed to get litellm version {e}")

        return LaminarInstrumentationScopeAttributes(
            name="litellm",
            version=litellm_version,
        )

    @override
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is not None:
            return self._scope
        self._scope = self._instrumentation_scope()
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config = LaminarInstrumentorConfig(
            wrapped_functions=[
                # we are not wrapping `acompletion`, and `aresponses`,
                # because they call `completion` and `responses` internally respectively
                WrappedFunctionSpec(
                    package_name="litellm",
                    object_name=None,
                    method_name="completion",
                    is_async=False,
                    is_streaming=False,
                    span_name="litellm.completion",
                    span_type="LLM",
                    wrapper_function=wrap_completion,
                    # Enable alias replacement for module-level function
                    replace_aliases=True,
                    instrumentation_scope=self.instrumentation_scope(),
                ),
                WrappedFunctionSpec(
                    package_name="litellm",
                    object_name=None,
                    method_name="responses",
                    is_async=False,
                    is_streaming=True,
                    span_name="litellm.responses",
                    span_type="LLM",
                    wrapper_function=wrap_responses,
                    # Enable alias replacement for module-level function
                    replace_aliases=True,
                    instrumentation_scope=self.instrumentation_scope(),
                ),
            ],
        )

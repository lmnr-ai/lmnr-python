from collections.abc import Collection
from importlib.metadata import version

from typing_extensions import override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    achat_wrapper,
    chat_wrapper,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.shared.completion_wrappers import (
    acompletion_wrapper,
    completion_wrapper,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.shared.embeddings_wrappers import (
    aembeddings_wrapper,
    embeddings_wrapper,
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

_instruments = ("openai >= 0.27.0", "openai < 1.0.0")
logger = get_default_logger(__name__)


WRAPPED_FUNCTIONS: list[WrappedFunctionSpec] = [
    WrappedFunctionSpec(
        package_name="openai",
        object_name="Completion",
        method_name="create",
        span_name="openai.completion",
        is_async=False,
        wrapper_function=completion_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai",
        object_name="Completion",
        method_name="acreate",
        span_name="openai.completion",
        is_async=True,
        wrapper_function=acompletion_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai",
        object_name="ChatCompletion",
        method_name="create",
        span_name="openai.chat",
        is_async=False,
        wrapper_function=chat_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai",
        object_name="ChatCompletion",
        method_name="acreate",
        span_name="openai.chat",
        is_async=True,
        wrapper_function=achat_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai",
        object_name="Embedding",
        method_name="create",
        span_name="openai.embeddings",
        is_async=False,
        wrapper_function=embeddings_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai",
        object_name="Embedding",
        method_name="acreate",
        span_name="openai.embeddings",
        is_async=True,
        wrapper_function=aembeddings_wrapper,
    ),
]


class OpenAIV0Instrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    def __init__(self):
        super().__init__()
        self.instrumentor_config: LaminarInstrumentorConfig = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                openai_version = version("openai")
            except Exception:
                logger.debug("Failed to get openai version", exc_info=True)
                openai_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="openai",
                version=openai_version,
            )
        return self._scope

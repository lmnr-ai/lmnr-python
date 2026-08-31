from importlib.metadata import version
from typing import Collection

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.sdk.log import get_default_logger
from ..shared.chat_wrappers import (
    achat_wrapper,
    chat_wrapper,
)
from ..shared.completion_wrappers import (
    acompletion_wrapper,
    completion_wrapper,
)
from ..shared.embeddings_wrappers import (
    aembeddings_wrapper,
    embeddings_wrapper,
)
from .assistant_wrappers import (
    assistants_create_wrapper,
    messages_list_wrapper,
    runs_create_and_stream_wrapper,
    runs_create_wrapper,
    runs_retrieve_wrapper,
)

from .responses_wrappers import (
    async_responses_cancel_wrapper,
    async_responses_get_or_create_wrapper,
    responses_cancel_wrapper,
    responses_get_or_create_wrapper,
)


_instruments = ("openai >= 1.0.0",)
logger = get_default_logger(__name__)


WRAPPED_FUNCTIONS: list[WrappedFunctionSpec] = [
    WrappedFunctionSpec(
        package_name="openai.resources.chat.completions",
        object_name="Completions",
        method_name="create",
        span_name="openai.chat",
        is_async=False,
        wrapper_function=chat_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.completions",
        object_name="Completions",
        method_name="create",
        span_name="openai.completion",
        is_async=False,
        wrapper_function=completion_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.embeddings",
        object_name="Embeddings",
        method_name="create",
        span_name="openai.embeddings",
        is_async=False,
        wrapper_function=embeddings_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.chat.completions",
        object_name="AsyncCompletions",
        method_name="create",
        span_name="openai.chat",
        is_async=True,
        wrapper_function=achat_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.completions",
        object_name="AsyncCompletions",
        method_name="create",
        span_name="openai.completion",
        is_async=True,
        wrapper_function=acompletion_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.embeddings",
        object_name="AsyncEmbeddings",
        method_name="create",
        span_name="openai.embeddings",
        is_async=True,
        wrapper_function=aembeddings_wrapper,
    ),
    # in newer versions, Completions.parse are out of beta
    WrappedFunctionSpec(
        package_name="openai.resources.chat.completions",
        object_name="Completions",
        method_name="parse",
        span_name="openai.chat",
        is_async=False,
        wrapper_function=chat_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.chat.completions",
        object_name="AsyncCompletions",
        method_name="parse",
        span_name="openai.chat",
        is_async=True,
        wrapper_function=achat_wrapper,
    ),
    # Beta APIs may not be available consistently in all versions. The base
    # instrumentor swallows AttributeError / ModuleNotFoundError / ImportError
    # per row, which is what the old `_try_wrap` helper did by hand.
    WrappedFunctionSpec(
        package_name="openai.resources.beta.assistants",
        object_name="Assistants",
        method_name="create",
        is_async=False,
        wrapper_function=assistants_create_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.beta.chat.completions",
        object_name="Completions",
        method_name="parse",
        span_name="openai.chat",
        is_async=False,
        wrapper_function=chat_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.beta.chat.completions",
        object_name="AsyncCompletions",
        method_name="parse",
        span_name="openai.chat",
        is_async=True,
        wrapper_function=achat_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.beta.threads.runs",
        object_name="Runs",
        method_name="create",
        is_async=False,
        wrapper_function=runs_create_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.beta.threads.runs",
        object_name="Runs",
        method_name="retrieve",
        is_async=False,
        wrapper_function=runs_retrieve_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.beta.threads.runs",
        object_name="Runs",
        method_name="create_and_stream",
        span_name="openai.assistant.run_stream",
        is_async=False,
        wrapper_function=runs_create_and_stream_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.beta.threads.messages",
        object_name="Messages",
        method_name="list",
        span_name="openai.assistant.run",
        is_async=False,
        wrapper_function=messages_list_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.responses",
        object_name="Responses",
        method_name="create",
        span_name="openai.response",
        is_async=False,
        wrapper_function=responses_get_or_create_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.responses",
        object_name="Responses",
        method_name="retrieve",
        span_name="openai.response",
        is_async=False,
        wrapper_function=responses_get_or_create_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.responses",
        object_name="Responses",
        method_name="cancel",
        span_name="openai.response",
        is_async=False,
        wrapper_function=responses_cancel_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.responses",
        object_name="AsyncResponses",
        method_name="create",
        span_name="openai.response",
        is_async=True,
        wrapper_function=async_responses_get_or_create_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.responses",
        object_name="AsyncResponses",
        method_name="retrieve",
        span_name="openai.response",
        is_async=True,
        wrapper_function=async_responses_get_or_create_wrapper,
    ),
    WrappedFunctionSpec(
        package_name="openai.resources.responses",
        object_name="AsyncResponses",
        method_name="cancel",
        span_name="openai.response",
        is_async=True,
        wrapper_function=async_responses_cancel_wrapper,
    ),
]


class OpenAIV1Instrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    def __init__(self):
        super().__init__()
        self.instrumentor_config = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                openai_version = version("openai")
            except Exception as e:
                logger.debug(f"Failed to get openai version {e}")
                openai_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="openai",
                version=openai_version,
            )
        return self._scope

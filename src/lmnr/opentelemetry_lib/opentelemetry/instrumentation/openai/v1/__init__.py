from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

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

from ..version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper


_instruments = ("openai >= 1.0.0",)
logger = get_default_logger(__name__)


class OpenAIV1Instrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _try_wrap(self, module, function, wrapper):
        """
        Wrap a function if it exists, otherwise do nothing.
        This is useful for handling cases where the function is not available in
        the older versions of the library.

        Args:
            module (str): The module to wrap, e.g. "openai.resources.chat.completions"
            function (str): "Object.function" to wrap, e.g. "Completions.parse"
            wrapper (callable): The wrapper to apply to the function.
        """
        try:
            wrap_function_wrapper(module, function, wrapper)
        except (AttributeError, ModuleNotFoundError, ImportError):
            logger.debug(f"Failed to wrap {module}.{function}")
            pass

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        wrap_function_wrapper(
            "openai.resources.chat.completions",
            "Completions.create",
            chat_wrapper(
                tracer,
            ),
        )

        wrap_function_wrapper(
            "openai.resources.completions",
            "Completions.create",
            completion_wrapper(tracer),
        )

        wrap_function_wrapper(
            "openai.resources.embeddings",
            "Embeddings.create",
            embeddings_wrapper(
                tracer,
            ),
        )

        wrap_function_wrapper(
            "openai.resources.chat.completions",
            "AsyncCompletions.create",
            achat_wrapper(
                tracer,
            ),
        )
        wrap_function_wrapper(
            "openai.resources.completions",
            "AsyncCompletions.create",
            acompletion_wrapper(tracer),
        )
        wrap_function_wrapper(
            "openai.resources.embeddings",
            "AsyncEmbeddings.create",
            aembeddings_wrapper(
                tracer,
            ),
        )
        # in newer versions, Completions.parse are out of beta
        self._try_wrap(
            "openai.resources.chat.completions",
            "Completions.parse",
            chat_wrapper(
                tracer,
            ),
        )
        self._try_wrap(
            "openai.resources.chat.completions",
            "AsyncCompletions.parse",
            achat_wrapper(
                tracer,
            ),
        )

        # Beta APIs may not be available consistently in all versions
        self._try_wrap(
            "openai.resources.beta.assistants",
            "Assistants.create",
            assistants_create_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.beta.chat.completions",
            "Completions.parse",
            chat_wrapper(
                tracer,
            ),
        )
        self._try_wrap(
            "openai.resources.beta.chat.completions",
            "AsyncCompletions.parse",
            achat_wrapper(
                tracer,
            ),
        )
        self._try_wrap(
            "openai.resources.beta.threads.runs",
            "Runs.create",
            runs_create_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.beta.threads.runs",
            "Runs.retrieve",
            runs_retrieve_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.beta.threads.runs",
            "Runs.create_and_stream",
            runs_create_and_stream_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.beta.threads.messages",
            "Messages.list",
            messages_list_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.responses",
            "Responses.create",
            responses_get_or_create_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.responses",
            "Responses.retrieve",
            responses_get_or_create_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.responses",
            "Responses.cancel",
            responses_cancel_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.responses",
            "AsyncResponses.create",
            async_responses_get_or_create_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.responses",
            "AsyncResponses.retrieve",
            async_responses_get_or_create_wrapper(tracer),
        )
        self._try_wrap(
            "openai.resources.responses",
            "AsyncResponses.cancel",
            async_responses_cancel_wrapper(tracer),
        )

    def _uninstrument(self, **kwargs):
        self.try_unwrap("openai.resources.chat.completions.Completions", "create")
        self.try_unwrap("openai.resources.completions.Completions", "create")
        self.try_unwrap("openai.resources.embeddings.Embeddings", "create")
        self.try_unwrap("openai.resources.chat.completions.AsyncCompletions", "create")
        self.try_unwrap("openai.resources.completions.AsyncCompletions", "create")
        self.try_unwrap("openai.resources.embeddings.AsyncEmbeddings", "create")
        self.try_unwrap("openai.resources.images.Images", "generate")
        self.try_unwrap("openai.resources.chat.completions.Completions", "parse")
        self.try_unwrap("openai.resources.chat.completions.AsyncCompletions", "parse")
        self.try_unwrap("openai.resources.beta.assistants.Assistants", "create")
        self.try_unwrap("openai.resources.beta.chat.completions.Completions", "parse")
        self.try_unwrap(
            "openai.resources.beta.chat.completions.AsyncCompletions", "parse"
        )
        self.try_unwrap("openai.resources.beta.threads.runs.Runs", "create")
        self.try_unwrap("openai.resources.beta.threads.runs.Runs", "retrieve")
        self.try_unwrap("openai.resources.beta.threads.runs.Runs", "create_and_stream")
        self.try_unwrap("openai.resources.beta.threads.messages.Messages", "list")
        self.try_unwrap("openai.resources.responses.Responses", "create")
        self.try_unwrap("openai.resources.responses.Responses", "retrieve")
        self.try_unwrap("openai.resources.responses.Responses", "cancel")
        self.try_unwrap("openai.resources.responses.AsyncResponses", "create")
        self.try_unwrap("openai.resources.responses.AsyncResponses", "retrieve")
        self.try_unwrap("openai.resources.responses.AsyncResponses", "cancel")

    def try_unwrap(self, module, function):
        try:
            unwrap(module, function)
        except (AttributeError, ModuleNotFoundError, ImportError):
            logger.debug(f"Failed to unwrap {module}.{function}")
            pass

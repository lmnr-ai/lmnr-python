from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from ..shared.chat_wrappers import (
    achat_wrapper,
    chat_wrapper,
)
from ..shared.completion_wrappers import (
    acompletion_wrapper,
    completion_wrapper,
)
from ..shared.config import Config
from ..shared.embeddings_wrappers import (
    aembeddings_wrapper,
    embeddings_wrapper,
)
from ..version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

_instruments = ("openai >= 0.27.0", "openai < 1.0.0")


class OpenAIV0Instrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        wrap_function_wrapper(
            "openai",
            "Completion.create",
            completion_wrapper(tracer),
        )

        wrap_function_wrapper(
            "openai",
            "Completion.acreate",
            acompletion_wrapper(tracer),
        )
        wrap_function_wrapper(
            "openai",
            "ChatCompletion.create",
            chat_wrapper(
                tracer,
            ),
        )
        wrap_function_wrapper(
            "openai",
            "ChatCompletion.acreate",
            achat_wrapper(
                tracer,
            ),
        )
        wrap_function_wrapper(
            "openai",
            "Embedding.create",
            embeddings_wrapper(
                tracer,
            ),
        )
        wrap_function_wrapper(
            "openai",
            "Embedding.acreate",
            aembeddings_wrapper(
                tracer,
            ),
        )

    def _uninstrument(self, **kwargs):
        unwrap("openai", "Completion.create")
        unwrap("openai", "Completion.acreate")
        unwrap("openai", "ChatCompletion.create")
        unwrap("openai", "ChatCompletion.acreate")
        unwrap("openai", "Embedding.create")
        unwrap("openai", "Embedding.acreate")

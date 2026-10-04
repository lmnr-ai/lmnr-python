"""
Initially copied over from openllmetry, commit
b3a18c9f7e6ff2368c8fb0bc35fd9123f11121c4
"""

from collections.abc import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from .utils import is_openai_v1

_instruments = ("openai >= 0.27.0",)


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        if is_openai_v1():
            from .v1 import OpenAIV1Instrumentor

            OpenAIV1Instrumentor().instrument(**kwargs)
        else:
            from .v0 import OpenAIV0Instrumentor

            OpenAIV0Instrumentor().instrument(**kwargs)

    def _uninstrument(self, **kwargs):
        if is_openai_v1():
            from .v1 import OpenAIV1Instrumentor

            OpenAIV1Instrumentor().uninstrument(**kwargs)
        else:
            from .v0 import OpenAIV0Instrumentor

            OpenAIV0Instrumentor().uninstrument(**kwargs)

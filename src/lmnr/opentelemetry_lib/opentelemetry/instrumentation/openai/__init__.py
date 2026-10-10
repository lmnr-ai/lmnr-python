"""
Initially copied over from openllmetry, commit
b3a18c9f7e6ff2368c8fb0bc35fd9123f11121c4
"""

from collections.abc import Collection
from typing import Any

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from typing_extensions import override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.utils import (
    is_openai_v1,
)

_instruments = ("openai >= 0.27.0",)


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def __init__(self):
        super().__init__()

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
    def _instrument(self, **kwargs: Any):
        if is_openai_v1():
            from .v1 import OpenAIV1Instrumentor

            OpenAIV1Instrumentor().instrument(**kwargs)
        else:
            from .v0 import OpenAIV0Instrumentor

            OpenAIV0Instrumentor().instrument(**kwargs)

    @override
    def _uninstrument(self, **kwargs: Any):
        if is_openai_v1():
            from .v1 import OpenAIV1Instrumentor

            OpenAIV1Instrumentor().uninstrument(**kwargs)
        else:
            from .v0 import OpenAIV0Instrumentor

            OpenAIV0Instrumentor().uninstrument(**kwargs)

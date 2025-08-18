"""
Initially copied over from openllmetry, commit
b3a18c9f7e6ff2368c8fb0bc35fd9123f11121c4
"""

from typing import Callable, Collection, Optional

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from .shared.config import Config
from .utils import is_openai_v1
from typing_extensions import Coroutine

_instruments = ("openai >= 0.27.0",)


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def __init__(
        self,
        enrich_assistant: bool = False,
        enrich_token_usage: bool = False,
        exception_logger=None,
        get_common_metrics_attributes: Callable[[], dict] = lambda: {},
        upload_base64_image: Optional[
            Callable[[str, str, str, str], Coroutine[None, None, str]]
        ] = lambda *args: "",
        enable_trace_context_propagation: bool = True,
        use_legacy_attributes: bool = True,
    ):
        super().__init__()
        Config.enrich_assistant = enrich_assistant
        Config.enrich_token_usage = enrich_token_usage
        Config.exception_logger = exception_logger
        Config.get_common_metrics_attributes = get_common_metrics_attributes
        Config.upload_base64_image = upload_base64_image
        Config.enable_trace_context_propagation = enable_trace_context_propagation
        Config.use_legacy_attributes = use_legacy_attributes

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

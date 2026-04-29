"""OpenTelemetry Opencode SDK instrumentation.

Injects a synthetic `text` part containing the current Laminar span context
into the ``parts`` argument of ``opencode_ai.resources.session.SessionResource.chat``
and its async counterpart. The Opencode server forwards the metadata back to
Laminar so the downstream model call is parented under the active observe span.

Mirrors the TS ``OpencodeInstrumentation`` (``@opencode-ai/sdk`` ``Session.prompt``
/ ``Session.promptAsync``) in ``lmnr-ts``.
"""

from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from lmnr.sdk.log import get_default_logger

from .wrappers import wrap_chat, wrap_chat_async

logger = get_default_logger(__name__)

_instruments = ("opencode-ai >= 0.1.0a0",)

WRAPPED_METHODS = [
    {
        "package": "opencode_ai.resources.session",
        "object": "SessionResource",
        "method": "chat",
        "wrapper": wrap_chat,
    },
    {
        "package": "opencode_ai.resources.session",
        "object": "AsyncSessionResource",
        "method": "chat",
        "wrapper": wrap_chat_async,
    },
]


class OpencodeInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            package = wrapped_method["package"]
            obj = wrapped_method["object"]
            method = wrapped_method["method"]
            wrapper = wrapped_method["wrapper"]
            try:
                wrap_function_wrapper(package, f"{obj}.{method}", wrapper)
            except (ModuleNotFoundError, AttributeError):
                pass

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            package = wrapped_method["package"]
            obj = wrapped_method["object"]
            method = wrapped_method["method"]
            try:
                unwrap(f"{package}.{obj}", method)
            except (ModuleNotFoundError, AttributeError):
                pass

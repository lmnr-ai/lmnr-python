"""OpenTelemetry pydantic_ai instrumentation.

Wires pydantic_ai's built-in OpenTelemetry GenAI semconv instrumentation to the
Laminar tracer provider. pydantic_ai already emits `chat {model}`,
`execute_tool {name}`, and `invoke_agent {name}` spans following the OTel
GenAI semantic conventions, so all we need is to route them through the
tracer provider Laminar already set up and pin the conventions version.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor


class PydanticAIInstrumentor(BaseInstrumentor):
    _previous_instrument_default: object = None
    _enabled: bool = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return ("pydantic-ai-slim >= 1.0.0",)

    def _instrument(self, **kwargs):
        try:
            pkg_version = version("pydantic-ai-slim")
        except PackageNotFoundError:
            try:
                pkg_version = version("pydantic-ai")
            except PackageNotFoundError:
                pkg_version = "0.0.0"

        from packaging.version import InvalidVersion, parse

        try:
            if parse(pkg_version) < parse("1.0.0"):
                return
        except InvalidVersion:
            pass

        from pydantic_ai import Agent
        from pydantic_ai.models.instrumented import InstrumentationSettings

        tracer_provider = kwargs.get("tracer_provider")
        settings = InstrumentationSettings(
            tracer_provider=tracer_provider,
            version=5,
        )

        self._previous_instrument_default = Agent._instrument_default
        Agent.instrument_all(settings)
        self._enabled = True

    def _uninstrument(self, **kwargs):
        if not self._enabled:
            return

        from pydantic_ai import Agent

        Agent._instrument_default = self._previous_instrument_default
        self._previous_instrument_default = None
        self._enabled = False

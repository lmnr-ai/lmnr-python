"""OpenTelemetry pydantic_ai instrumentation.

Wires pydantic_ai's built-in OpenTelemetry GenAI semconv instrumentation to the
Laminar tracer provider. pydantic_ai already emits `chat {model}`,
`execute_tool {name}`, and `invoke_agent {name}` spans following the OTel
GenAI semantic conventions, so all we need is to route them through the
tracer provider Laminar already set up and pick a reasonable default for the
conventions version.

We also wrap `InstrumentationSettings.__init__` so that every code path that
builds one — `Agent.instrument_all(True)`, `Agent(instrument=True)`, or
a user directly constructing `InstrumentationSettings(...)` — ends up on our
tracer provider, and defaults to semconv `version=5` when the caller did not
pick a version explicitly (or picked the now-legacy `version=1`). Callers that
pass any `version >= 2` keep whatever they asked for.
"""

from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

_DEFAULT_SEMCONV_VERSION = 5
_SETTINGS_MODULE = "pydantic_ai.models.instrumented"
_SETTINGS_INIT = "InstrumentationSettings.__init__"


class PydanticAIInstrumentor(BaseInstrumentor):
    _previous_instrument_default: object = None
    _enabled: bool = False

    def instrumentation_dependencies(self) -> Collection[str]:
        # Declared for `BaseInstrumentor.instrument()`'s dependency check.
        # We only list `pydantic-ai-slim` (the core package shared by both the
        # `pydantic-ai-slim` and `pydantic-ai` distributions) — the "full"
        # `pydantic-ai` package re-exports from it, so this single constraint
        # covers both install flavors. The upper bound guards against silent
        # breakage from a 2.x release that could change `InstrumentationSettings`
        # (which we instantiate eagerly during `_instrument`); when 2.x ships,
        # validate compatibility and bump the cap.
        return ("pydantic-ai-slim >= 1.0.0, < 2.0.0",)

    def _instrument(self, **kwargs: Any):
        from pydantic_ai import Agent
        from pydantic_ai.models.instrumented import InstrumentationSettings

        default_tracer_provider = kwargs.get("tracer_provider")

        def _wrap_settings_init(wrapped, instance, args, call_kwargs):
            if call_kwargs.get("tracer_provider") is None:
                call_kwargs["tracer_provider"] = default_tracer_provider
            if call_kwargs.get("version") in (None, 1):
                call_kwargs["version"] = _DEFAULT_SEMCONV_VERSION
            return wrapped(*args, **call_kwargs)

        wrap_function_wrapper(_SETTINGS_MODULE, _SETTINGS_INIT, _wrap_settings_init)

        settings = InstrumentationSettings()

        self._previous_instrument_default = Agent._instrument_default
        Agent.instrument_all(settings)
        self._enabled = True

    def _uninstrument(self, **kwargs: Any):
        if not self._enabled:
            return

        from pydantic_ai import Agent
        from pydantic_ai.models.instrumented import InstrumentationSettings

        Agent._instrument_default = self._previous_instrument_default
        self._previous_instrument_default = None

        unwrap(InstrumentationSettings, "__init__")

        self._enabled = False

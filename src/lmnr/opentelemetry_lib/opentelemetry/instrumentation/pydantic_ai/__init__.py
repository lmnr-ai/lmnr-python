"""OpenTelemetry pydantic_ai instrumentation.

Wires pydantic_ai's built-in OpenTelemetry GenAI semconv instrumentation to the
Laminar tracer provider. pydantic_ai already emits `chat {model}`,
`execute_tool {name}`, and `invoke_agent {name}` spans following the OTel
GenAI semantic conventions, so all we need is to route them through the
tracer provider Laminar already set up and pick a reasonable default for the
conventions version.

We also patch `InstrumentationSettings.__init__` so that every code path that
builds one — `Agent.instrument_all(True)`, `Agent(instrument=True)`, or
a user directly constructing `InstrumentationSettings(...)` — ends up on our
tracer provider, and defaults to semconv `version=5` when the caller did not
pick a version explicitly (or picked the now-legacy `version=1`). Callers that
pass any `version >= 2` keep whatever they asked for.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

_DEFAULT_SEMCONV_VERSION = 5


class PydanticAIInstrumentor(BaseInstrumentor):
    _previous_instrument_default: object = None
    _previous_settings_init: Callable[..., None] | None = None
    _enabled: bool = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return ("pydantic-ai-slim >= 1.0.0",)

    def _instrument(self, **kwargs: Any):
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
            # Bail out rather than patch against an unknown version: the
            # monkey-patch assumes the `>=1.0.0` API shape, and an unparseable
            # version string means we can't confirm that assumption.
            return

        from pydantic_ai import Agent
        from pydantic_ai.models.instrumented import InstrumentationSettings

        tracer_provider = kwargs.get("tracer_provider")

        # Patch InstrumentationSettings.__init__ so that *every* construction
        # path ends up on our tracer provider, and picks a reasonable default
        # semconv version — regardless of whether it's Laminar, user code, or
        # pydantic_ai itself doing the construction (e.g. `instrument_model`
        # does `InstrumentationSettings()` when given `instrument=True`).
        #
        # Version policy: semconv versions >= 2 are broadly compatible with
        # Laminar's parsing, so callers who pick one explicitly get exactly
        # that. If the caller did not pass a version (or passed the
        # legacy `version=1`, which pydantic_ai's own default maps to),
        # upgrade them to v5 so we don't ship v1 spans by accident.
        original_settings_init = InstrumentationSettings.__init__
        default_tracer_provider = tracer_provider

        def patched_settings_init(
            self,
            *,
            tracer_provider=default_tracer_provider,
            **kw,
        ):
            user_version = kw.pop("version", None)
            if user_version is None or user_version == 1:
                effective_version = _DEFAULT_SEMCONV_VERSION
            else:
                effective_version = user_version
            original_settings_init(
                self,
                tracer_provider=tracer_provider,
                version=effective_version,
                **kw,
            )

        InstrumentationSettings.__init__ = patched_settings_init
        self._previous_settings_init = original_settings_init

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

        if self._previous_settings_init is not None:
            InstrumentationSettings.__init__ = self._previous_settings_init
            self._previous_settings_init = None

        self._enabled = False

"""OpenTelemetry pydantic_ai instrumentation.

Wires pydantic_ai's built-in OpenTelemetry GenAI semconv instrumentation to the
Laminar tracer provider. pydantic_ai already emits `chat {model}`,
`execute_tool {name}`, and `invoke_agent {name}` spans following the OTel
GenAI semantic conventions, so all we need is to route them through the
tracer provider Laminar already set up and pin the conventions version.

We also patch `InstrumentationSettings.__init__` so that every code path that
builds one — `Agent.instrument_all(True)`, `Agent(instrument=True)`, or
a user directly constructing `InstrumentationSettings(...)` — ends up with
`version=5` and our tracer provider. Without this, a user calling
`Agent.instrument_all()` after Laminar initialization would silently downgrade
the semconv version to pydantic_ai's current default and swap the tracer
provider back to the global one.
"""

import warnings

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

_FORCED_SEMCONV_VERSION = 5


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
        # path ends up with version=5 and our tracer provider — regardless of
        # whether it's Laminar, user code, or pydantic_ai itself doing the
        # construction (e.g. `instrument_model` does
        # `InstrumentationSettings()` when given `instrument=True`).
        original_settings_init = InstrumentationSettings.__init__
        default_tracer_provider = tracer_provider

        def patched_settings_init(
            self,
            *,
            tracer_provider=default_tracer_provider,
            **kw,
        ):
            # The semconv version is intentionally forced: Laminar's span
            # parsing assumes v5. If a caller passes `version=`, warn loudly
            # and drop it rather than silently honoring a version we don't
            # support.
            user_version = kw.pop("version", None)
            if (
                user_version is not None
                and user_version != _FORCED_SEMCONV_VERSION
            ):
                warnings.warn(
                    (
                        "Laminar's pydantic_ai instrumentor forces "
                        f"InstrumentationSettings(version={_FORCED_SEMCONV_VERSION}); "
                        f"ignoring caller-supplied version={user_version}. "
                        "Uninstrument Laminar's pydantic_ai integration if "
                        "you need a different semconv version."
                    ),
                    stacklevel=2,
                )
            original_settings_init(
                self,
                tracer_provider=tracer_provider,
                version=_FORCED_SEMCONV_VERSION,
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

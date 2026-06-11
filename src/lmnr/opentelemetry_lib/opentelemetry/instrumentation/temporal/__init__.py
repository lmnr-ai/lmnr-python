"""Laminar Temporal instrumentation.

Unlike the TypeScript integration — which needs explicit ``instrumentModules``
wiring because of how Node/ESM resolves modules — the Python SDK hooks in
automatically: when ``temporalio`` is installed and Temporal isn't in the
disabled instrument set, :class:`TemporalInstrumentor` patches
``temporalio.client.Client`` so a :class:`LaminarTracingInterceptor` is injected
into every client. The Temporal worker auto-inherits client interceptors
(``temporalio.worker._worker`` prepends client interceptors to worker ones), so
one injection covers the client, activity, and workflow paths from a single
patch — no separate worker/bundle patching like TS.

The options that gate activity span recording (``create_activity_span`` /
``record_activity_args`` / ``record_activity_output``) are read from the
process-wide :data:`_interceptor_options` so they can be set via
``Laminar.initialize(..., instrument_modules=...)``-style configuration before
the instrumentor runs; they default to the dataclass defaults otherwise.
"""

from __future__ import annotations

from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.utils import unwrap

from lmnr.sdk.log import get_default_logger

from .interceptors import (
    LaminarTemporalInterceptorOptions,
    LaminarTracingInterceptor,
)

logger = get_default_logger(__name__)

_instruments = ("temporalio >= 1.7.0",)

#: Process-wide options applied to the auto-injected interceptor. Overridden by
#: the instrumentor when explicit options are passed at initialization.
_interceptor_options = LaminarTemporalInterceptorOptions()


def _has_laminar_interceptor(interceptors: Any) -> bool:
    return any(
        isinstance(i, LaminarTracingInterceptor) for i in (interceptors or [])
    )


def _wrap_client_init(wrapped, instance, args, kwargs):
    """Inject a Laminar interceptor into every ``Client`` construction.

    ``Client.connect`` builds the config then calls ``cls(...)``, so patching
    ``__init__`` covers both the high-level ``connect`` path and direct
    ``Client(service_client, ...)`` construction. Idempotent — never injects a
    second interceptor if one is already present (e.g. user passed it
    explicitly, or a re-entrant construction).
    """
    interceptors = list(kwargs.get("interceptors") or [])
    if not _has_laminar_interceptor(interceptors):
        interceptors.append(LaminarTracingInterceptor(_interceptor_options))
        kwargs["interceptors"] = interceptors
    return wrapped(*args, **kwargs)


class TemporalInstrumentor(BaseInstrumentor):
    def __init__(self, options: LaminarTemporalInterceptorOptions | None = None):
        super().__init__()
        self._options = options

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        if self._options is not None:
            global _interceptor_options
            _interceptor_options = self._options
        try:
            wrap_function_wrapper(
                "temporalio.client",
                "Client.__init__",
                _wrap_client_init,
            )
        except (ModuleNotFoundError, AttributeError) as e:
            logger.debug(f"failed to instrument temporalio client: {e}")

    def _uninstrument(self, **kwargs):
        try:
            unwrap("temporalio.client.Client", "__init__")
        except (ModuleNotFoundError, AttributeError):
            pass


__all__ = [
    "TemporalInstrumentor",
    "LaminarTracingInterceptor",
    "LaminarTemporalInterceptorOptions",
]

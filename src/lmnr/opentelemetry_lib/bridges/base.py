"""The shape every competitor bridge implements.

A bridge sends what a competitor's SDK records (Langfuse, LangSmith,
Braintrust, ...) to Laminar as well, without changing what the competitor's
own backend receives. Each one is enabled through
`Laminar.initialize(bridges=[...])`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from opentelemetry.sdk.trace import SpanProcessor, TracerProvider

from lmnr.opentelemetry_lib.tracing.instruments import Instruments


class Bridge(str, Enum):
    """Competitor SDKs that Laminar can bridge from."""

    LANGFUSE = "langfuse"


class LaminarBridge(ABC):
    """A bridge from one competitor SDK into Laminar.

    Instances are process-wide singletons (see `bridges/__init__.py`):
    `install` runs on every `Laminar.initialize`, `uninstall` on every
    `Laminar.shutdown`.
    """

    bridge: Bridge

    #: What `is_available` checks for, shown in the warning when it fails,
    #: e.g. "langfuse >= 3.0 (`pip install -U langfuse`)".
    requirement: str = ""

    #: Laminar instruments that would record the calls this bridge already
    #: forwards. They are left out of the default instrument set while the
    #: bridge is enabled.
    conflicting_instruments: frozenset[Instruments] = frozenset()

    #: Whether Laminar's tracing must stay off the global OTel tracer provider
    #: while this bridge is enabled.
    requires_private_tracer_provider: bool = False

    #: Whether Laminar spans should nest under spans of the competitor that are
    #: active in the global OTel context.
    inherits_global_context: bool = False

    @abstractmethod
    def is_available(self) -> bool:
        """Whether a supported version of the competitor SDK is installed."""

    @abstractmethod
    def install(
        self, lmnr_tracer_provider: TracerProvider, lmnr_span_processor: SpanProcessor
    ) -> bool:
        """Start forwarding to `lmnr_span_processor`. Returns whether the
        bridge is now active. Must never raise."""

    @abstractmethod
    def uninstall(self) -> None:
        """Undo `install`. Idempotent; must never raise."""

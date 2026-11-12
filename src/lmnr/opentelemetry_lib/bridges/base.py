"""The shape every bridge implements.

A bridge sends what a third-party tracing SDK (the source SDK: Langfuse,
LangSmith, Braintrust, ...) records to Laminar as well, without changing
what the source SDK's own backend receives. Each one is enabled through
`Laminar.initialize(bridges=[...])`.

The contract assumes nothing about how a bridge hooks the source SDK.
Langfuse is built on OpenTelemetry, so its bridge adds a span processor to
Langfuse's tracer provider; a bridge for an SDK that isn't (LangSmith,
Braintrust) can hook it at the network level and record spans with Laminar's
tracer instead. Either way the spans reach Laminar through the
`BridgeContext` passed to `install`. Only the attributes in the last section
of `LaminarBridge` are specific to OpenTelemetry-based SDKs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from opentelemetry.sdk.trace import SpanProcessor, TracerProvider

from lmnr.opentelemetry_lib.tracing.instruments import Instruments


class Bridge(str, Enum):
    """Source SDKs that Laminar can bridge from."""

    LANGFUSE = "langfuse"


@dataclass(frozen=True)
class BridgeContext:
    """Laminar's tracing pipeline, as built by `Laminar.initialize`."""

    #: Laminar's span processor. A bridge that receives finished spans from
    #: the source SDK hands Laminar's copies to it directly.
    span_processor: SpanProcessor
    #: Laminar's tracer provider. A bridge that builds spans itself (from
    #: intercepted requests, say) creates them with
    #: `tracer_provider.get_tracer(...)`, which routes them through
    #: `span_processor` like any other Laminar span.
    tracer_provider: TracerProvider


class LaminarBridge(ABC):
    """A bridge from one source SDK into Laminar.

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

    @abstractmethod
    def is_available(self) -> bool:
        """Whether a supported version of the source SDK is installed."""

    @abstractmethod
    def install(self, context: BridgeContext) -> bool:
        """Start sending to Laminar through `context`. Returns whether the
        bridge is now active. Must never raise."""

    @abstractmethod
    def uninstall(self) -> None:
        """Undo `install`. Idempotent; must never raise."""

    # Only for source SDKs built on OpenTelemetry, whose spans share the
    # process's global OTel state with Laminar's. A bridge that hooks its SDK
    # at the network level leaves both False.

    #: Keep Laminar's tracer provider off the global OTel slot, so the
    #: source SDK doesn't pick it up as its own.
    requires_private_tracer_provider: bool = False

    #: Nest Laminar spans under the source SDK's span that is active in the
    #: global OTel context.
    inherits_global_context: bool = False

"""Competitor bridges, enabled with `Laminar.initialize(bridges=[...])`.

See `base.py` for the contract and `AGENTS.md` for how they fit together.
"""

from __future__ import annotations

from collections.abc import Iterable

from opentelemetry.sdk.trace import SpanProcessor, TracerProvider

from lmnr.opentelemetry_lib.bridges.base import Bridge, LaminarBridge
from lmnr.opentelemetry_lib.bridges.langfuse import LangfuseBridge
from lmnr.opentelemetry_lib.tracing.instruments import Instruments
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

_BRIDGES: dict[Bridge, LaminarBridge] = {
    Bridge.LANGFUSE: LangfuseBridge(),
}


def get_bridge(bridge: Bridge) -> LaminarBridge:
    return _BRIDGES[bridge]


def resolve_bridges(bridges: Iterable[Bridge | str] | None) -> list[LaminarBridge]:
    """Bridge objects for `bridges`, deduplicated. Raises `ValueError` on an
    unknown name so a typo doesn't silently turn the bridge off."""
    resolved: list[LaminarBridge] = []
    for name in bridges or ():
        try:
            bridge = _BRIDGES[Bridge(name)]
        except ValueError:
            known = ", ".join(repr(b.value) for b in Bridge)
            raise ValueError(f"Unknown Laminar bridge {name!r}. Known bridges: {known}") from None
        if bridge not in resolved:
            resolved.append(bridge)
    return resolved


def conflicting_instruments(bridges: Iterable[LaminarBridge]) -> set[Instruments]:
    return {i for bridge in bridges for i in bridge.conflicting_instruments}


def install_bridges(
    bridges: Iterable[LaminarBridge],
    lmnr_tracer_provider: TracerProvider,
    lmnr_span_processor: SpanProcessor,
) -> None:
    for bridge in bridges:
        try:
            _installed = bridge.install(lmnr_tracer_provider, lmnr_span_processor)
        except Exception:
            logger.warning("Failed to install the %s bridge", bridge.bridge.value, exc_info=True)


def uninstall_bridges() -> None:
    for bridge in _BRIDGES.values():
        bridge.uninstall()


__all__ = [
    "Bridge",
    "LaminarBridge",
    "conflicting_instruments",
    "get_bridge",
    "install_bridges",
    "resolve_bridges",
    "uninstall_bridges",
]

"""Bridges from third-party tracing SDKs, enabled with `Laminar.initialize(bridges=[...])`.

See `base.py` for the contract and `AGENTS.md` for how they fit together.
"""

from __future__ import annotations

from collections.abc import Iterable

from lmnr.opentelemetry_lib.bridges.base import Bridge, BridgeContext, LaminarBridge
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


def available_bridges(bridges: Iterable[LaminarBridge]) -> list[LaminarBridge]:
    """The bridges whose source SDK is installed in a supported version.
    The others are skipped with a warning and have no effect at all."""
    available: list[LaminarBridge] = []
    for bridge in bridges:
        if bridge.is_available():
            available.append(bridge)
        else:
            logger.warning(
                "The %s bridge needs %s; it is disabled for this run.",
                bridge.bridge.value,
                bridge.requirement,
            )
    return available


def conflicting_instruments(bridges: Iterable[LaminarBridge]) -> set[Instruments]:
    return {i for bridge in bridges for i in bridge.conflicting_instruments}


def install_bridges(
    bridges: Iterable[LaminarBridge], context: BridgeContext
) -> list[LaminarBridge]:
    """Install `bridges` and return the ones that are now active."""
    installed: list[LaminarBridge] = []
    for bridge in bridges:
        try:
            if bridge.install(context):
                installed.append(bridge)
        except Exception:
            logger.warning("Failed to install the %s bridge", bridge.bridge.value, exc_info=True)
    return installed


def uninstall_bridges() -> None:
    for bridge in _BRIDGES.values():
        bridge.uninstall()


__all__ = [
    "Bridge",
    "BridgeContext",
    "LaminarBridge",
    "available_bridges",
    "conflicting_instruments",
    "get_bridge",
    "install_bridges",
    "resolve_bridges",
    "uninstall_bridges",
]

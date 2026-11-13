"""Per-instrumentation options (`Laminar.initialize(instrument_options=...)`).

Options are resolved once per `initialize()`, from the argument and the
`LMNR_{INSTRUMENT}_{OPTION}` environment variables, and read by the
instrumentations each time they record a span. Never capture them when an
instrumentor is built: instrumentors are instrumented once per process and
reused across initialize()/shutdown() cycles, so a captured value would ignore
a re-initialization with different options.

A leaf module (no `tracing` or instrumentor imports), so any instrumentation
can import it without closing an import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import InstrumentOptions
from lmnr.sdk.utils import from_env

logger = get_default_logger(__name__)

InstrumentName = Literal["langchain", "langgraph", "temporal"]

#: Every option with its default. The public `InstrumentOptions` TypedDict in
#: `sdk/types.py` declares the same keys (pinned by a test), and the
#: environment variable names are derived from this table, so adding an option
#: means adding it here and there.
OPTION_DEFAULTS: dict[str, dict[str, bool]] = {
    "langchain": {"record_metadata": False},
    "langgraph": {"record_graph": True},
    "temporal": {
        "create_activity_span": True,
        "record_activity_args": True,
        "record_activity_output": True,
    },
}

_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off"})

# Explicitly set values only (argument or environment), so callers with a
# legacy source of the same option (Temporal) can tell "unset" from "default".
# None until resolved; resolved lazily from the environment alone when read
# before `initialize()`.
_resolved: dict[str, dict[str, bool]] | None = None


def env_var_name(instrument: str, option: str) -> str:
    return f"LMNR_{instrument}_{option}".upper()


def _parse_bool(name: str, raw: str) -> bool | None:
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    logger.warning(f"Ignoring {name}={raw!r}: expected true or false.")
    return None


def resolve_instrument_options(
    options: InstrumentOptions | Mapping[str, Any] | None,
) -> dict[str, dict[str, bool]]:
    """The explicitly set options: `options` over the environment.

    Unknown instruments, unknown options and non-boolean values are ignored
    with a warning, never raised: a typo in tracing config must not stop the
    app from starting.
    """
    resolved: dict[str, dict[str, bool]] = {}
    for instrument, defaults in OPTION_DEFAULTS.items():
        for option in defaults:
            name = env_var_name(instrument, option)
            if (raw := from_env(name)) and (value := _parse_bool(name, raw)) is not None:
                resolved.setdefault(instrument, {})[option] = value

    for key, values in (options or {}).items():
        # Accept `Instruments.LANGCHAIN` as well as "langchain"
        instrument = getattr(key, "value", key)
        defaults = OPTION_DEFAULTS.get(instrument)
        if defaults is None:
            logger.warning(
                f"Ignoring instrument_options[{instrument!r}]: no options for this "
                + f"instrument. Known instruments: {', '.join(OPTION_DEFAULTS)}."
            )
            continue
        if not isinstance(values, Mapping):
            logger.warning(f"Ignoring instrument_options[{instrument!r}]: expected a dict.")
            continue
        for option, value in cast(Mapping[str, Any], values).items():
            if option not in defaults:
                logger.warning(
                    f"Ignoring unknown option instrument_options[{instrument!r}][{option!r}]. "
                    + f"Known options: {', '.join(defaults)}."
                )
                continue
            if not isinstance(value, bool):
                logger.warning(
                    f"Ignoring instrument_options[{instrument!r}][{option!r}]={value!r}: "
                    + "expected a bool."
                )
                continue
            resolved.setdefault(instrument, {})[option] = value
    return resolved


def set_instrument_options(options: InstrumentOptions | None) -> None:
    global _resolved
    _resolved = resolve_instrument_options(options)


def reset_instrument_options() -> None:
    global _resolved
    _resolved = None


def explicit_instrument_option(instrument: InstrumentName, option: str) -> bool | None:
    """The value set by `instrument_options` or its environment variable, or None."""
    global _resolved
    if _resolved is None:
        _resolved = resolve_instrument_options(None)
    return _resolved.get(instrument, {}).get(option)


def get_instrument_option(instrument: InstrumentName, option: str) -> bool:
    value = explicit_instrument_option(instrument, option)
    return OPTION_DEFAULTS[instrument][option] if value is None else value

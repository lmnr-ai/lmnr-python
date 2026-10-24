"""Whether Laminar uses the Anthropic SDK's native OpenTelemetry spans.

`anthropic >= 1.14` emits its own span (`anthropic.messages.create`, scope
`com.anthropic.sdk.python`) for every API call whenever a global tracer
provider exists, which Laminar sets. Recording both it and our wrapper's
`anthropic.chat` span counts every call's cost twice, so exactly one of the
two is kept:

- By default our wrappers run and the processor drops the native spans. The
  native span records no message content unless asked to, never emits an
  `exception` event (so app-server shows failed calls as successful), and
  skips debug replay.
- With `ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE=content` (and native
  traces not turned off with `ANTHROPIC_OPEN_TELEMETRY=false`) the user has
  opted into the native spans, so the anthropic instrumentor installs no
  wrappers and the native spans are exported as they are.

Leaf module: imported by both `tracing/processor.py` and the anthropic
instrumentor, so it must not import anything from `lmnr`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

from packaging.version import InvalidVersion, Version

ANTHROPIC_NATIVE_SCOPE = "com.anthropic.sdk.python"
# The first release with native OpenTelemetry spans.
MIN_NATIVE_TELEMETRY_VERSION = Version("1.14.0")


@lru_cache(maxsize=1)
def native_spans_requested() -> bool:
    """Whether the environment opts into the native spans, with their content.

    Read once: the anthropic client reads the same variables when it is
    constructed, and the instrumentor and the processor must agree.
    """
    if os.environ.get("ANTHROPIC_OPEN_TELEMETRY", "").lower() == "false":
        return False
    mode = os.environ.get("ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE", "")
    return mode.lower() == "content"


def has_native_telemetry() -> bool:
    """Whether the installed `anthropic` emits its own spans."""
    try:
        return Version(version("anthropic")) >= MIN_NATIVE_TELEMETRY_VERSION
    except (PackageNotFoundError, InvalidVersion):
        return False


def use_native_spans() -> bool:
    """Whether the anthropic instrumentor should leave tracing to the SDK."""
    return native_spans_requested() and has_native_telemetry()


def is_anthropic_native_scope(scope_name: str | None) -> bool:
    """Whether a span's instrumentation scope is the Anthropic SDK's.

    Bedrock / Vertex / Foundry clients append a platform suffix
    (`com.anthropic.sdk.python.bedrock`).
    """
    return scope_name is not None and (
        scope_name == ANTHROPIC_NATIVE_SCOPE or scope_name.startswith(ANTHROPIC_NATIVE_SCOPE + ".")
    )

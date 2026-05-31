"""Shared replay helpers for the per-provider LLM wrappers (§G, §H).

The provider wrappers keep their own `cached_response_to_*` reconstruction and
streaming wrappers; everything generic — the replay decision, span-path
resolution, and the CACHED span marking — lives here so all providers stay in
lockstep with the in-process `ReplayCache`.
"""

from typing import Any

from opentelemetry.trace import Span

from lmnr.sdk.debug import get_runtime
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

# Span-path attribute set by the processor's on_start (dot-joined in helpers).
SPAN_PATH_ATTRIBUTE = "lmnr.span.path"


def replay_enabled() -> bool:
    """True when this process is a debug run with a replay cache."""
    return get_runtime() is not None


def span_path_from_span(span: Span | None) -> str | None:
    """Resolve the dot-joined span path from a span's attributes.

    Provider spans are created with `tracer.start_span()` and aren't registered
    in Laminar's context, so the path is read from the attribute the processor
    already set.
    """
    if span is None:
        return None
    try:
        path_list = span.attributes.get(SPAN_PATH_ATTRIBUTE)
        if path_list:
            return ".".join(path_list)
    except Exception:
        pass
    return None


def cached_payload_for(span_path: str | None) -> dict[str, Any] | None:
    """Advance the occurrence counter and return the payload to replay, or None.

    Always advances the counter (when a runtime exists) so a live call and a
    replayed call consume the same occurrence slot — keeping the spine's
    occurrence index aligned across record and replay runs.
    """
    runtime = get_runtime()
    if runtime is None or not span_path:
        return None
    return runtime.get_cached(span_path)


def mark_span_cached(span: Span | None) -> None:
    """Stamp the CACHED boundary attributes the frontend renders (§9)."""
    try:
        if span and span.is_recording():
            span.set_attributes(
                {
                    "lmnr.span.type": "CACHED",
                    "lmnr.span.original_type": "LLM",
                }
            )
    except Exception:
        pass

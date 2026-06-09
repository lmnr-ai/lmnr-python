"""Shared replay helpers for the per-provider LLM wrappers (shared spec §6).

The provider wrappers keep their own `cached_response_to_*` reconstruction and
streaming wrappers; everything generic — the replay gate, reading the input
messages off the live span, hashing them, and the server-side cache lookup —
lives here so all four providers stay in lockstep.

v2 has no in-process cache: each live LLM call hashes its input messages and
asks the app-server cache endpoint what to do (HIT / MISS / LIVE, shared spec
§7). The decision is computed here in `cache_outcome_for` / `acache_outcome_for`
and consumed by the wrappers.
"""

import json
from typing import Any

from opentelemetry.trace import Span

from lmnr.sdk.debug import get_runtime
from lmnr.sdk.debug.hash import debug_input_hash
from lmnr.sdk.debug.outcome import CacheOutcome
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

# Span-path attribute set by the processor's on_start (dot-joined in helpers).
SPAN_PATH_ATTRIBUTE = "lmnr.span.path"
# Input-messages attribute every provider stamps on the live span BEFORE the
# rollout wrapper runs. It is exactly what becomes `spans.input` server-side, so
# hashing it here gives the same hash app-server computes — guaranteeing parity
# without re-deriving messages per provider. Stored as a JSON STRING.
GEN_AI_INPUT_MESSAGES_ATTRIBUTE = "gen_ai.input.messages"


def replay_enabled() -> bool:
    """True when this process is a debug run with replay configured.

    Replay configured means a source trace id plus a cache-until span-id needle
    (shared spec §4). A debug-no-replay run (`LMNR_DEBUG` set but no
    `LMNR_DEBUG_REPLAY_TRACE_ID` / `LMNR_DEBUG_CACHE_UNTIL`) returns False — the
    provider wrappers then skip the cache lookup entirely and run every call
    live. v2 builds nothing synchronously, so this is a pure config check.
    """
    runtime = get_runtime()
    return runtime is not None and runtime.replay_configured


def input_messages_from_span(span: Span | None) -> list[Any] | None:
    """Read and parse the `gen_ai.input.messages` JSON off a live span.

    Every provider sets this attribute (via `json_dumps`) before the rollout
    wrapper runs, so it is the canonical input the server hashed into
    `spans.input`. Returns the decoded list, or None when the attribute is
    missing/unparseable — in which case the caller runs the call live rather
    than risk hashing a partial input.
    """
    if span is None:
        return None
    try:
        raw = span.attributes.get(GEN_AI_INPUT_MESSAGES_ATTRIBUTE)
        if not raw:
            return None
        messages = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(messages, list):
            return messages
    except Exception:
        pass
    return None


def cache_outcome_for(span: Span | None) -> CacheOutcome | None:
    """Decide HIT / MISS / LIVE for one live LLM call (sync path).

    Returns None when there is nothing to do (debug off, replay not configured,
    or the span carries no usable input) — the wrapper then just runs live
    without latching anything. Otherwise returns the `CacheOutcome` from the
    server-side cache endpoint (shared spec §7).

    A MISS latches the process-wide run-live flag so every later call skips the
    endpoint and runs live; a LIVE (warmup/transport degrade) runs THIS call
    live without latching. The actual cache HTTP never raises — `cache()`
    degrades to `kind="live"` itself.
    """
    runtime = get_runtime()
    if runtime is None or not runtime.replay_configured:
        return None
    # Function-local import to avoid the laminar <-> debug import cycle.
    from lmnr.sdk.laminar import Laminar

    if Laminar.is_debug_run_live():
        return CacheOutcome(kind="live")
    input_messages = input_messages_from_span(span)
    if input_messages is None:
        return None
    input_hash = debug_input_hash(input_messages)
    outcome = runtime.client.rollout_sessions.cache(
        session_id=runtime.session_id,
        replay_trace_id=runtime.replay_trace_id,
        cache_until=runtime.cache_until_span_id,
        input_hash=input_hash,
    )
    if outcome.kind == "miss":
        Laminar.set_debug_run_live(True)
    return outcome


async def acache_outcome_for(span: Span | None) -> CacheOutcome | None:
    """Async variant of `cache_outcome_for` — uses the async cache client.

    Identical decision logic; only the cache HTTP is awaited through the
    runtime's retained `AsyncLaminarClient`. The run-live latch is process-wide
    and shared with the sync path.
    """
    runtime = get_runtime()
    if runtime is None or not runtime.replay_configured:
        return None
    from lmnr.sdk.laminar import Laminar

    if Laminar.is_debug_run_live():
        return CacheOutcome(kind="live")
    input_messages = input_messages_from_span(span)
    if input_messages is None:
        return None
    input_hash = debug_input_hash(input_messages)
    outcome = await runtime.async_client.rollout_sessions.cache(
        session_id=runtime.session_id,
        replay_trace_id=runtime.replay_trace_id,
        cache_until=runtime.cache_until_span_id,
        input_hash=input_hash,
    )
    if outcome.kind == "miss":
        Laminar.set_debug_run_live(True)
    return outcome


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

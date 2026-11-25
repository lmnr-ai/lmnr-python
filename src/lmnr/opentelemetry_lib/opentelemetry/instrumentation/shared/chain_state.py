"""Which parts of a LangChain / LangGraph chain payload are worth recording.

Shared by the LangChain instrumentation and the LangSmith bridge, which
records the same runs from LangSmith's side. No LangChain import here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

# Graph state keys known to hold small values, in both camelCase and snake_case.
# Best effort: anything else is dropped from chain spans.
_SMALL_STATE_KEYS = frozenset(
    {
        "maxSteps",
        "max_steps",
        "stepsTaken",
        "steps_taken",
        "executionId",
        "execution_id",
        "tokenUsage",
        "token_usage",
        "startTime",
        "start_time",
        "timeout",
        "pendingWakeUp",
        "pending_wake_up",
        "pendingContinuation",
        "pending_continuation",
    }
)
_SMALL_PROMPT_KEYS = frozenset(
    {"model", "fallbackModel", "fallback_model", "thinkingEffort", "thinking_effort"}
)


def small_state(payload: Any) -> dict[str, Any]:
    """Keep only a chain payload's known-small keys.

    LangGraph graphs and nodes receive and return whole graph state,
    whose heavy parts (messages, prompts, tools, raw `input` / `kwargs`) vary
    by setup and repeat on every span. Rather than chase them, nothing is kept
    except an allowlist, so the result is often empty.
    """
    if not isinstance(payload, Mapping):
        return {}
    payload = cast(Mapping[Any, Any], payload)
    kept = {key: value for key, value in payload.items() if key in _SMALL_STATE_KEYS}
    prompt = payload.get("prompt")
    if isinstance(prompt, Mapping):
        small_prompt = {
            key: value
            for key, value in cast(Mapping[Any, Any], prompt).items()
            if key in _SMALL_PROMPT_KEYS
        }
        if small_prompt:
            kept["prompt"] = small_prompt
    return kept

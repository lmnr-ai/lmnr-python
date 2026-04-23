"""Unit tests for system-instructions handling in the openai_agents instrumentation."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.helpers import (  # noqa: E501
    get_current_system_instructions,
    reset_current_system_instructions,
    set_current_system_instructions,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.messages import (  # noqa: E501
    set_gen_ai_input_messages,
)


def _captured_attrs() -> tuple[MagicMock, dict[str, object]]:
    """A minimal LaminarSpan stand-in that records set_attribute calls."""
    span = MagicMock()
    captured: dict[str, object] = {}

    def _set_attribute(key: str, value: object) -> None:
        captured[key] = value

    span.set_attribute.side_effect = _set_attribute
    return span, captured


def test_set_gen_ai_input_messages_without_system_instructions_unchanged():
    """No system_instructions → output matches pre-change behavior."""
    span, captured = _captured_attrs()
    set_gen_ai_input_messages(span, "hello there")
    messages = json.loads(captured["gen_ai.input.messages"])
    assert messages == [{"role": "user", "content": "hello there"}]


def test_set_gen_ai_input_messages_prepends_system_instructions():
    span, captured = _captured_attrs()
    set_gen_ai_input_messages(
        span,
        [{"role": "user", "content": "What is 2+2?"}],
        system_instructions="You are a calculator.",
    )
    messages = json.loads(captured["gen_ai.input.messages"])
    assert messages == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a calculator."}],
        },
        {"role": "user", "content": "What is 2+2?"},
    ]


def test_set_gen_ai_input_messages_system_only_still_emits():
    """If the model was called with only system instructions and no other
    input, the span should still surface the system turn."""
    span, captured = _captured_attrs()
    set_gen_ai_input_messages(
        span, None, system_instructions="Always respond in French."
    )
    messages = json.loads(captured["gen_ai.input.messages"])
    assert messages == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Always respond in French."}],
        }
    ]


def test_set_gen_ai_input_messages_noop_when_both_missing():
    span, captured = _captured_attrs()
    set_gen_ai_input_messages(span, None, system_instructions=None)
    assert captured == {}
    span.set_attribute.assert_not_called()


def test_system_instructions_contextvar_is_task_local():
    """Concurrent tasks must see their own system instructions so multi-agent
    runs (handoffs, parallel sub-agents) do not leak the prompt."""

    async def _scenario(prompt: str) -> str | None:
        token = set_current_system_instructions(prompt)
        try:
            # Yield control so other tasks may run while we hold our value.
            await asyncio.sleep(0)
            return get_current_system_instructions()
        finally:
            reset_current_system_instructions(token)

    async def _runner() -> tuple[str | None, str | None]:
        return await asyncio.gather(_scenario("alpha"), _scenario("beta"))

    results = asyncio.run(_runner())
    assert sorted([s for s in results if s is not None]) == ["alpha", "beta"]
    assert get_current_system_instructions() is None

"""LLM message shapes on Laminar's copy of a Langfuse span.

`fixtures/python_sdk_payloads.json` holds the `langfuse.observation.input` /
`output` that langfuse 5.2 recorded for a real two-call tool loop (a call that
answers with a tool call only, then a call with the tool result in its
history) per integration. Encrypted reasoning blobs are shortened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from lmnr.opentelemetry_lib.tracing.attributes import SPAN_INPUT, SPAN_OUTPUT

from .utils import FakeSpan, translate

PAYLOADS: dict[str, list[dict[str, str]]] = json.loads(
    (Path(__file__).parent / "fixtures" / "python_sdk_payloads.json").read_text()
)

ROLES = {"system", "user", "assistant", "tool"}
TOOL_CALL = {
    "type": "tool_call",
    "id": "call_1",
    "name": "get_weather",
    "arguments": '{"city":"Paris"}',
}


def _generation(attrs: dict[str, Any]) -> dict[str, Any]:
    span = FakeSpan({"langfuse.observation.type": "generation", **attrs})
    return translate(span)


def _messages(attrs: dict[str, Any], key: str) -> list[Any]:
    return json.loads(cast(str, attrs[key]))


def _canonical(message: Any) -> str:
    return json.dumps(message, sort_keys=True)


@pytest.mark.parametrize("scenario", sorted(PAYLOADS))
def test_recorded_tool_loop_becomes_laminar_messages(scenario: str):
    first, second = (_generation(raw) for raw in PAYLOADS[scenario])
    for attrs in (first, second):
        assert SPAN_INPUT not in attrs and SPAN_OUTPUT not in attrs
        assert "gen_ai.tool.definitions" in attrs
        for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
            for message in _messages(attrs, key):
                assert message["role"] in ROLES, message
                content = message["content"]
                parts = cast(list[dict[str, Any]], content) if isinstance(content, list) else []
                for part in parts:
                    assert part["type"] in ("text", "reasoning", "tool_call"), part
                    if part["type"] == "tool_call":
                        assert set(part) == {"type", "id", "name", "arguments"}

    # The first call answers with a tool call only.
    output = _messages(first, "gen_ai.output.messages")
    assert len(output) == 1
    assert output[0]["role"] == "assistant"
    assert [p["type"] for p in output[0]["content"] if p["type"] != "reasoning"] == ["tool_call"]
    # ... and its output is repeated verbatim in the second call's history,
    # so app-server stores the assistant turn once.
    history = {_canonical(m) for m in _messages(second, "gen_ai.input.messages")}
    assert _canonical(output[0]) in history
    tool_results = [m for m in _messages(second, "gen_ai.input.messages") if m["role"] == "tool"]
    assert tool_results[0]["tool_call_id"] == output[0]["content"][-1]["id"]


def test_responses_api_input_is_split_into_messages_and_tools():
    raw = PAYLOADS["openai_responses_stream"][0]
    attrs = _generation(raw)
    request = json.loads(raw["langfuse.observation.input"])
    assert _messages(attrs, "gen_ai.input.messages") == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What's the weather in Paris? Use the tool."},
    ]
    assert _messages(attrs, "gen_ai.tool.definitions") == request["tools"]
    assert "langfuse.observation.input" not in attrs


def test_responses_api_items_of_one_turn_are_one_assistant_message():
    """reasoning + function_call items become one message; the encrypted
    reasoning payload is dropped, the summary is kept."""
    attrs = _generation(
        {
            "langfuse.observation.output": json.dumps(
                [
                    {
                        "id": "rs_1",
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "Call the tool."}],
                        "content": [],
                        "encrypted_content": "gAAAA",
                    },
                    {
                        "id": "fc_1",
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                        "status": "completed",
                    },
                ]
            )
        }
    )
    assert _messages(attrs, "gen_ai.output.messages") == [
        {
            "role": "assistant",
            "content": [{"type": "reasoning", "text": "Call the tool."}, TOOL_CALL],
        }
    ]


def test_responses_api_single_output_item_and_string_input():
    attrs = _generation(
        {
            "langfuse.observation.input": json.dumps(
                {"input": "weather?", "instructions": "Be brief.", "tools": []}
            ),
            # langfuse.openai records a one-item output as the bare item.
            "langfuse.observation.output": json.dumps(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Sunny.", "annotations": []}],
                }
            ),
        }
    )
    assert _messages(attrs, "gen_ai.input.messages") == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "weather?"},
    ]
    assert _messages(attrs, "gen_ai.output.messages") == [
        {"role": "assistant", "content": "Sunny."}
    ]


def test_streamed_chat_text_output_is_an_assistant_message():
    attrs = _generation({"langfuse.observation.output": "It is sunny."})
    assert _messages(attrs, "gen_ai.output.messages") == [
        {"role": "assistant", "content": "It is sunny."}
    ]
    assert "langfuse.observation.output" not in attrs


def test_chat_tool_call_output_matches_the_next_input():
    """The tool-call-only chat output used to be wrapped as `{message: ...}`
    while the same turn in the next input became content parts."""
    message = {
        "role": "assistant",
        "content": None,
        "refusal": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            }
        ],
    }
    attrs = _generation(
        {
            "langfuse.observation.input": json.dumps([message]),
            "langfuse.observation.output": json.dumps(message),
        }
    )
    expected = [{"role": "assistant", "content": [TOOL_CALL]}]
    assert _messages(attrs, "gen_ai.input.messages") == expected
    assert _messages(attrs, "gen_ai.output.messages") == expected


def test_anthropic_messages_become_laminar_messages():
    """A manual generation with Anthropic-native messages: `tool_use` and
    `thinking` blocks, and `tool_result` blocks in a user turn."""
    attrs = _generation(
        {
            "langfuse.observation.input": json.dumps(
                [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "Use the tool.", "signature": "s"},
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "get_weather",
                                "input": {"city": "Paris"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                            {"type": "text", "text": "And tomorrow?"},
                        ],
                    },
                ]
            ),
        }
    )
    assert _messages(attrs, "gen_ai.input.messages") == [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "text": "Use the tool."},
                {
                    "type": "tool_call",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "arguments": {"city": "Paris"},
                },
            ],
        },
        {"role": "tool", "content": "sunny", "tool_call_id": "toolu_1"},
        {"role": "user", "content": "And tomorrow?"},
    ]


def test_openinference_anthropic_content_wrappers_are_unwrapped():
    """openinference's Anthropic instrumentation records a tool call both as a
    `message_content` block and in `tool_calls`; text as `message_content`."""
    prefix = "llm.output_messages.0.message"
    attrs = translate(
        FakeSpan(
            {
                "openinference.span.kind": "LLM",
                f"{prefix}.role": "assistant",
                f"{prefix}.contents.0.message_content.type": "text",
                f"{prefix}.contents.0.message_content.text": "Checking.",
                f"{prefix}.contents.1.message_content.type": "tool_use",
                f"{prefix}.contents.1.tool_call.id": "toolu_1",
                f"{prefix}.contents.1.tool_call.function.name": "get_weather",
                f"{prefix}.contents.1.tool_call.function.arguments": '{"city":"Paris"}',
                f"{prefix}.tool_calls.0.tool_call.id": "toolu_1",
                f"{prefix}.tool_calls.0.tool_call.function.name": "get_weather",
                f"{prefix}.tool_calls.0.tool_call.function.arguments": '{"city":"Paris"}',
            },
            scope_name="openinference.instrumentation.anthropic",
        )
    )
    assert _messages(attrs, "gen_ai.output.messages") == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                {**TOOL_CALL, "id": "toolu_1"},
            ],
        }
    ]


def test_openinference_gemini_model_role_is_assistant():
    attrs = translate(
        FakeSpan(
            {
                "openinference.span.kind": "LLM",
                "llm.output_messages.0.message.role": "model",
                "llm.output_messages.0.message.content": "Sunny.",
            },
            scope_name="openinference.instrumentation.google_genai",
        )
    )
    assert _messages(attrs, "gen_ai.output.messages") == [
        {"role": "assistant", "content": "Sunny."}
    ]


def test_untyped_langchain_tool_calls_are_tool_call_parts():
    attrs = _generation(
        {
            "langfuse.observation.output": json.dumps(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "name": "get_weather", "args": {"a": 1}}],
                }
            )
        }
    )
    assert _messages(attrs, "gen_ai.output.messages") == [
        {"role": "assistant", "content": [{**TOOL_CALL, "arguments": {"a": 1}}]}
    ]

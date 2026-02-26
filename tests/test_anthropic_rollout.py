"""
Tests for Anthropic rollout wrapper functionality.

Tests the caching, override application, and response conversion
specific to the Anthropic instrumentation rollout feature.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic.rollout import (
    AnthropicRolloutWrapper,
    get_anthropic_rollout_wrapper,
)


@pytest.fixture
def wrapper():
    """Create a fresh AnthropicRolloutWrapper instance for each test."""
    return AnthropicRolloutWrapper()


# --- cached_response_to_anthropic tests ---


class TestCachedResponseToAnthropic:
    def test_from_raw_response(self, wrapper):
        """Test converting a cached span with raw response to Anthropic Message."""
        raw_response = {
            "id": "msg_01ABC",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Hello, world!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        cached_span = {
            "attributes": {
                "lmnr.sdk.raw.response": json.dumps(raw_response),
            },
            "output": "",
        }

        result = wrapper.cached_response_to_anthropic(cached_span)

        assert result is not None
        assert result.id == "msg_01ABC"
        assert result.model == "claude-3-5-sonnet-20241022"
        assert result.role == "assistant"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hello, world!"
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_from_raw_response_dict(self, wrapper):
        """Test converting a cached span with raw response as dict."""
        raw_response = {
            "id": "msg_01ABC",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        cached_span = {
            "attributes": {
                "lmnr.sdk.raw.response": raw_response,  # dict, not string
            },
            "output": "",
        }

        result = wrapper.cached_response_to_anthropic(cached_span)

        assert result is not None
        assert result.id == "msg_01ABC"
        assert result.content[0].text == "Hello!"

    def test_from_output_text(self, wrapper):
        """Test converting from output when raw response is not available."""
        cached_span = {
            "attributes": {
                "gen_ai.response.id": "msg_fallback",
                "gen_ai.response.model": "claude-3-5-sonnet-20241022",
            },
            "output": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Fallback response"}],
                        "stop_reason": "end_turn",
                    }
                ]
            ),
        }

        result = wrapper.cached_response_to_anthropic(cached_span)

        assert result is not None
        assert result.id == "msg_fallback"
        assert result.model == "claude-3-5-sonnet-20241022"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Fallback response"

    def test_from_output_tool_use(self, wrapper):
        """Test converting output with tool_use blocks."""
        cached_span = {
            "attributes": {
                "gen_ai.response.id": "msg_tools",
                "gen_ai.response.model": "claude-3-5-sonnet-20241022",
            },
            "output": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_01ABC",
                                "name": "get_weather",
                                "input": {"location": "NYC"},
                            }
                        ],
                        "stop_reason": "tool_use",
                    }
                ]
            ),
        }

        result = wrapper.cached_response_to_anthropic(cached_span)

        assert result is not None
        assert len(result.content) == 1
        block = result.content[0]
        assert block.type == "tool_use"
        assert block.id == "toolu_01ABC"
        assert block.name == "get_weather"
        assert block.input == {"location": "NYC"}

    def test_from_output_tool_use_string_input(self, wrapper):
        """Test converting tool_use blocks with stringified JSON input."""
        cached_span = {
            "attributes": {
                "gen_ai.response.id": "msg_tools",
                "gen_ai.response.model": "claude-3-5-sonnet-20241022",
            },
            "output": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_01ABC",
                                "name": "get_weather",
                                "input": '{"location": "NYC"}',
                            }
                        ],
                        "stop_reason": "tool_use",
                    }
                ]
            ),
        }

        result = wrapper.cached_response_to_anthropic(cached_span)

        assert result is not None
        block = result.content[0]
        assert block.input == {"location": "NYC"}

    def test_from_output_mixed_content(self, wrapper):
        """Test converting output with both text and tool_use blocks."""
        cached_span = {
            "attributes": {
                "gen_ai.response.id": "msg_mixed",
                "gen_ai.response.model": "claude-3-5-sonnet-20241022",
            },
            "output": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Let me check the weather."},
                            {
                                "type": "tool_use",
                                "id": "toolu_01",
                                "name": "get_weather",
                                "input": {"location": "NYC"},
                            },
                        ],
                        "stop_reason": "tool_use",
                    }
                ]
            ),
        }

        result = wrapper.cached_response_to_anthropic(cached_span)

        assert result is not None
        assert len(result.content) == 2
        assert result.content[0].type == "text"
        assert result.content[0].text == "Let me check the weather."
        assert result.content[1].type == "tool_use"
        assert result.content[1].name == "get_weather"

    def test_empty_output(self, wrapper):
        """Test handling empty output."""
        cached_span = {
            "attributes": {},
            "output": "",
        }

        result = wrapper.cached_response_to_anthropic(cached_span)
        assert result is None

    def test_invalid_output(self, wrapper):
        """Test handling invalid output format."""
        cached_span = {
            "attributes": {},
            "output": "not valid json",
        }

        result = wrapper.cached_response_to_anthropic(cached_span)
        assert result is None

    def test_invalid_raw_response_falls_back_to_output(self, wrapper):
        """Test that invalid raw response falls back to output parsing."""
        cached_span = {
            "attributes": {
                "lmnr.sdk.raw.response": "invalid json",
                "gen_ai.response.id": "msg_fallback",
                "gen_ai.response.model": "claude-3-5-sonnet-20241022",
            },
            "output": json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Fallback"}],
                        "stop_reason": "end_turn",
                    }
                ]
            ),
        }

        result = wrapper.cached_response_to_anthropic(cached_span)

        assert result is not None
        assert result.content[0].text == "Fallback"


# --- apply_anthropic_overrides tests ---


class TestApplyAnthropicOverrides:
    def test_no_overrides(self, wrapper):
        """Test that kwargs are returned unchanged when no overrides exist."""
        wrapper._initialized = True
        wrapper._overrides = {}

        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = wrapper.apply_anthropic_overrides(kwargs, "test.path")
        assert result == kwargs

    def test_system_override(self, wrapper):
        """Test applying system prompt override."""
        wrapper._initialized = True
        wrapper._overrides = {
            "test.path": {"system": "You are a helpful pirate."}
        }

        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are a helpful assistant.",
        }

        result = wrapper.apply_anthropic_overrides(kwargs, "test.path")
        assert result["system"] == "You are a helpful pirate."
        assert result["messages"] == kwargs["messages"]  # unchanged

    def test_system_override_without_existing_system(self, wrapper):
        """Test applying system override when no system prompt exists."""
        wrapper._initialized = True
        wrapper._overrides = {
            "test.path": {"system": "You are a new assistant."}
        }

        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = wrapper.apply_anthropic_overrides(kwargs, "test.path")
        assert result["system"] == "You are a new assistant."

    def test_tool_override_update_existing(self, wrapper):
        """Test updating existing tool definitions."""
        wrapper._initialized = True
        wrapper._overrides = {
            "test.path": {
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Updated weather description",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ]
            }
        }

        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        }

        result = wrapper.apply_anthropic_overrides(kwargs, "test.path")
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tools"][0]["description"] == "Updated weather description"
        # parameters from override maps to input_schema in Anthropic format
        assert result["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"

    def test_tool_override_add_new(self, wrapper):
        """Test adding a new tool via override."""
        wrapper._initialized = True
        wrapper._overrides = {
            "test.path": {
                "tools": [
                    {
                        "name": "new_tool",
                        "description": "A new tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"param": {"type": "string"}},
                        },
                    }
                ]
            }
        }

        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [],
        }

        result = wrapper.apply_anthropic_overrides(kwargs, "test.path")
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "new_tool"

    def test_system_and_tool_overrides_combined(self, wrapper):
        """Test applying both system and tool overrides."""
        wrapper._initialized = True
        wrapper._overrides = {
            "test.path": {
                "system": "New system prompt",
                "tools": [
                    {
                        "name": "tool1",
                        "description": "Updated tool",
                        "parameters": {"type": "object"},
                    }
                ],
            }
        }

        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "Old system prompt",
            "tools": [
                {
                    "name": "tool1",
                    "description": "Old tool",
                    "input_schema": {"type": "object"},
                }
            ],
        }

        result = wrapper.apply_anthropic_overrides(kwargs, "test.path")
        assert result["system"] == "New system prompt"
        assert result["tools"][0]["description"] == "Updated tool"

    def test_system_override_updates_span_attribute(self, wrapper):
        """Test that system override updates the span's gen_ai.input.messages."""
        wrapper._initialized = True
        wrapper._overrides = {
            "test.path": {"system": "New system prompt"}
        }

        span = MagicMock()
        span.is_recording.return_value = True
        span.attributes = {
            "gen_ai.input.messages": json.dumps(
                [
                    {"role": "system", "content": "Old system prompt"},
                    {"role": "user", "content": "Hello"},
                ]
            )
        }

        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "Old system prompt",
        }

        wrapper.apply_anthropic_overrides(kwargs, "test.path", span=span)

        # Verify span attribute was updated
        span.set_attribute.assert_called_once()
        call_args = span.set_attribute.call_args
        assert call_args[0][0] == "gen_ai.input.messages"
        updated_messages = json.loads(call_args[0][1])
        assert updated_messages[0]["content"] == "New system prompt"


# --- _apply_tool_overrides tests ---


class TestApplyToolOverrides:
    def test_empty_overrides(self, wrapper):
        """Test that empty overrides return existing tools."""
        existing = [{"name": "tool1", "input_schema": {}}]
        result = wrapper._apply_tool_overrides(existing, [])
        assert result == existing

    def test_update_existing_tool(self, wrapper):
        """Test updating an existing tool's description."""
        existing = [
            {
                "name": "tool1",
                "description": "Old description",
                "input_schema": {"type": "object"},
            }
        ]
        overrides = [
            {
                "name": "tool1",
                "description": "New description",
            }
        ]

        result = wrapper._apply_tool_overrides(existing, overrides)
        assert len(result) == 1
        assert result[0]["description"] == "New description"
        assert result[0]["input_schema"] == {"type": "object"}

    def test_add_new_tool(self, wrapper):
        """Test adding a new tool with parameters."""
        existing = []
        overrides = [
            {
                "name": "new_tool",
                "description": "A new tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

        result = wrapper._apply_tool_overrides(existing, overrides)
        assert len(result) == 1
        assert result[0]["name"] == "new_tool"
        assert result[0]["input_schema"] == {"type": "object", "properties": {}}

    def test_override_without_name_skipped(self, wrapper):
        """Test that overrides without name are skipped."""
        existing = []
        overrides = [{"description": "No name"}]

        result = wrapper._apply_tool_overrides(existing, overrides)
        assert result is None


# --- wrap_create tests ---


class TestWrapCreate:
    def test_not_rollout_mode(self, wrapper):
        """Test that wrap_create passes through when not in rollout mode."""
        wrapped = MagicMock(return_value="original_response")

        with patch.object(wrapper, "should_use_rollout", return_value=False):
            result = wrapper.wrap_create(
                wrapped, None, (), {"model": "claude-3-5-sonnet-20241022"}
            )

        assert result == "original_response"
        wrapped.assert_called_once()

    def test_no_span_path(self, wrapper):
        """Test that wrap_create passes through when no span path."""
        wrapped = MagicMock(return_value="original_response")

        with patch.object(
            wrapper, "should_use_rollout", return_value=True
        ), patch.object(wrapper, "get_span_path", return_value=None):
            result = wrapper.wrap_create(
                wrapped, None, (), {"model": "claude-3-5-sonnet-20241022"}
            )

        assert result == "original_response"

    def test_cache_hit_non_streaming(self, wrapper):
        """Test wrap_create returns cached response for non-streaming."""
        raw_response = {
            "id": "msg_cached",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Cached hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        cached_span = {
            "attributes": {
                "lmnr.sdk.raw.response": json.dumps(raw_response),
            },
            "output": "",
        }

        wrapped = MagicMock()
        span = MagicMock()
        span.is_recording.return_value = True

        with patch.object(
            wrapper, "should_use_rollout", return_value=True
        ), patch.object(
            wrapper, "get_span_path", return_value="root.llm_call"
        ), patch.object(
            wrapper, "should_use_cache", return_value=True
        ), patch.object(
            wrapper, "get_cached_response", return_value=cached_span
        ):
            result = wrapper.wrap_create(
                wrapped,
                None,
                (),
                {"model": "claude-3-5-sonnet-20241022"},
                span=span,
                is_streaming=False,
            )

        # Should not call the original function
        wrapped.assert_not_called()
        assert result.id == "msg_cached"
        assert result.content[0].text == "Cached hello"

        # Verify span was marked as cached
        span.set_attributes.assert_called_once()
        attrs = span.set_attributes.call_args[0][0]
        assert attrs["lmnr.span.type"] == "CACHED"
        assert attrs["lmnr.span.original_type"] == "LLM"

    def test_cache_hit_streaming(self, wrapper):
        """Test wrap_create returns cached streaming response."""
        raw_response = {
            "id": "msg_cached_stream",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Streamed cached"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        cached_span = {
            "attributes": {
                "lmnr.sdk.raw.response": json.dumps(raw_response),
            },
            "output": "",
        }

        wrapped = MagicMock()
        span = MagicMock()
        span.is_recording.return_value = True

        with patch.object(
            wrapper, "should_use_rollout", return_value=True
        ), patch.object(
            wrapper, "get_span_path", return_value="root.llm_call"
        ), patch.object(
            wrapper, "should_use_cache", return_value=True
        ), patch.object(
            wrapper, "get_cached_response", return_value=cached_span
        ):
            result = wrapper.wrap_create(
                wrapped,
                None,
                (),
                {"model": "claude-3-5-sonnet-20241022"},
                span=span,
                is_streaming=True,
            )

        # Should return a generator
        events = list(result)
        assert len(events) > 0

        # Verify we got the expected streaming event types
        event_types = [e.type for e in events]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

    def test_cache_miss_applies_overrides(self, wrapper):
        """Test that overrides are applied when cache misses."""
        wrapped = MagicMock(return_value="live_response")

        wrapper._initialized = True
        wrapper._overrides = {
            "root.llm_call": {"system": "Overridden system prompt"}
        }

        with patch.object(
            wrapper, "should_use_rollout", return_value=True
        ), patch.object(
            wrapper, "get_span_path", return_value="root.llm_call"
        ), patch.object(
            wrapper, "should_use_cache", return_value=False
        ):
            result = wrapper.wrap_create(
                wrapped,
                None,
                (),
                {
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "system": "Original system prompt",
                },
            )

        assert result == "live_response"
        wrapped.assert_called_once()


# --- _create_cached_stream tests ---


class TestCreateCachedStream:
    def test_text_streaming(self, wrapper):
        """Test that cached text response is properly streamed."""
        from anthropic.types import Message, TextBlock, Usage

        message = Message(
            id="msg_test",
            type="message",
            role="assistant",
            model="claude-3-5-sonnet-20241022",
            content=[TextBlock(type="text", text="Hello from cache")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        )

        events = list(wrapper._create_cached_stream(message))

        # Verify event sequence
        assert events[0].type == "message_start"
        assert events[0].message.id == "msg_test"

        assert events[1].type == "content_block_start"
        assert events[1].index == 0
        assert events[1].content_block.type == "text"

        assert events[2].type == "content_block_delta"
        assert events[2].delta.type == "text_delta"
        assert events[2].delta.text == "Hello from cache"

        assert events[3].type == "content_block_stop"

        assert events[4].type == "message_delta"
        assert events[4].delta.stop_reason == "end_turn"

        assert events[5].type == "message_stop"

    def test_tool_use_streaming(self, wrapper):
        """Test that cached tool_use response is properly streamed."""
        from anthropic.types import Message, ToolUseBlock, Usage

        message = Message(
            id="msg_tool",
            type="message",
            role="assistant",
            model="claude-3-5-sonnet-20241022",
            content=[
                ToolUseBlock(
                    type="tool_use",
                    id="toolu_01",
                    name="get_weather",
                    input={"location": "NYC"},
                )
            ],
            stop_reason="tool_use",
            usage=Usage(input_tokens=10, output_tokens=15),
        )

        events = list(wrapper._create_cached_stream(message))

        assert events[0].type == "message_start"
        assert events[1].type == "content_block_start"
        assert events[1].content_block.type == "tool_use"
        assert events[1].content_block.name == "get_weather"

        assert events[2].type == "content_block_delta"
        assert events[2].delta.type == "input_json_delta"
        assert json.loads(events[2].delta.partial_json) == {"location": "NYC"}

    @pytest.mark.asyncio
    async def test_async_cached_stream(self, wrapper):
        """Test async cached stream yields same events as sync."""
        from anthropic.types import Message, TextBlock, Usage

        message = Message(
            id="msg_async",
            type="message",
            role="assistant",
            model="claude-3-5-sonnet-20241022",
            content=[TextBlock(type="text", text="Async cached")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        )

        events = []
        async for event in wrapper._create_async_cached_stream(message):
            events.append(event)

        assert len(events) == 6
        assert events[0].type == "message_start"
        assert events[5].type == "message_stop"


# --- get_anthropic_rollout_wrapper tests ---


class TestGetAnthropicRolloutWrapper:
    def test_returns_none_when_not_rollout(self):
        """Test returns None when not in rollout mode."""
        with patch(
            "lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic.rollout.is_rollout_mode",
            return_value=False,
        ):
            result = get_anthropic_rollout_wrapper()
            assert result is None

    def test_creates_singleton(self):
        """Test that wrapper is created as singleton."""
        import lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic.rollout as mod

        # Reset singleton
        mod._anthropic_rollout_wrapper = None

        with patch(
            "lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic.rollout.is_rollout_mode",
            return_value=True,
        ):
            w1 = get_anthropic_rollout_wrapper()
            w2 = get_anthropic_rollout_wrapper()
            assert w1 is w2
            assert w1 is not None

        # Cleanup
        mod._anthropic_rollout_wrapper = None

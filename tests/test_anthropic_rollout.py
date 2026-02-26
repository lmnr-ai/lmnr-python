"""
Tests for Anthropic rollout instrumentation wrapper.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic.rollout import (
    AnthropicRolloutWrapper,
    get_anthropic_rollout_wrapper,
)


@pytest.fixture
def wrapper():
    """Create a fresh AnthropicRolloutWrapper instance."""
    return AnthropicRolloutWrapper()


class TestCachedResponseToAnthropic:
    """Tests for cached_response_to_anthropic()."""

    def test_from_raw_response(self, wrapper):
        """Test conversion from raw response attribute."""
        raw_msg = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello world"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        cached_span = {
            "attributes": {
                "lmnr.sdk.raw.response": json.dumps(raw_msg),
            },
            "output": "",
        }

        result = wrapper.cached_response_to_anthropic(cached_span)

        assert result is not None
        assert isinstance(result, Message)
        assert result.id == "msg_123"
        assert result.model == "claude-sonnet-4-20250514"
        assert len(result.content) == 1
        assert result.content[0].text == "Hello world"
        assert result.stop_reason == "end_turn"

    def test_from_raw_response_dict(self, wrapper):
        """Test conversion when raw response is already a dict."""
        raw_msg = {
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Test"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        cached_span = {
            "attributes": {
                "lmnr.sdk.raw.response": raw_msg,
            },
            "output": "",
        }

        result = wrapper.cached_response_to_anthropic(cached_span)

        assert result is not None
        assert isinstance(result, Message)
        assert result.id == "msg_456"

    def test_from_output_text(self, wrapper):
        """Test fallback to output parsing with text content."""
        cached_span = {
            "attributes": {
                "gen_ai.response.id": "msg_cached",
                "gen_ai.response.model": "claude-sonnet-4-20250514",
            },
            "output": json.dumps([{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Cached response"},
                ],
                "stop_reason": "end_turn",
            }]),
        }

        result = wrapper.cached_response_to_anthropic(cached_span)

        assert result is not None
        assert isinstance(result, Message)
        assert result.id == "msg_cached"
        assert result.model == "claude-sonnet-4-20250514"
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "Cached response"
        assert result.stop_reason == "end_turn"

    def test_from_output_tool_use(self, wrapper):
        """Test fallback parsing with tool use content."""
        cached_span = {
            "attributes": {
                "gen_ai.response.id": "msg_tool",
                "gen_ai.response.model": "claude-sonnet-4-20250514",
            },
            "output": json.dumps([{
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "get_weather",
                        "input": {"location": "San Francisco"},
                    },
                ],
                "stop_reason": "tool_use",
            }]),
        }

        result = wrapper.cached_response_to_anthropic(cached_span)

        assert result is not None
        assert len(result.content) == 1
        assert isinstance(result.content[0], ToolUseBlock)
        assert result.content[0].name == "get_weather"
        assert result.content[0].input == {"location": "San Francisco"}
        assert result.stop_reason == "tool_use"

    def test_from_output_tool_use_string_input(self, wrapper):
        """Test tool use with stringified JSON input."""
        cached_span = {
            "attributes": {
                "gen_ai.response.id": "msg_tool",
                "gen_ai.response.model": "claude-sonnet-4-20250514",
            },
            "output": json.dumps([{
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_456",
                        "name": "search",
                        "input": '{"query": "test"}',
                    },
                ],
                "stop_reason": "tool_use",
            }]),
        }

        result = wrapper.cached_response_to_anthropic(cached_span)

        assert result is not None
        assert isinstance(result.content[0], ToolUseBlock)
        assert result.content[0].input == {"query": "test"}

    def test_empty_output(self, wrapper):
        """Test with empty output."""
        cached_span = {
            "attributes": {},
            "output": "",
        }

        result = wrapper.cached_response_to_anthropic(cached_span)
        assert result is None

    def test_invalid_output_format(self, wrapper):
        """Test with invalid output format."""
        cached_span = {
            "attributes": {},
            "output": json.dumps({"not": "a list"}),
        }

        result = wrapper.cached_response_to_anthropic(cached_span)
        assert result is None

    def test_raw_response_takes_priority(self, wrapper):
        """Test that raw response is preferred over output parsing."""
        cached_span = {
            "attributes": {
                "lmnr.sdk.raw.response": json.dumps({
                    "id": "msg_raw",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [{"type": "text", "text": "Raw response"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }),
            },
            "output": json.dumps([{
                "role": "assistant",
                "content": [{"type": "text", "text": "Legacy response"}],
            }]),
        }

        result = wrapper.cached_response_to_anthropic(cached_span)

        assert result is not None
        assert result.content[0].text == "Raw response"


class TestApplyAnthropicOverrides:
    """Tests for apply_anthropic_overrides()."""

    def test_system_override(self, wrapper):
        """Test system prompt override."""
        wrapper._initialized = True
        wrapper._overrides = {
            "root.llm": {"system": "You are a pirate."},
        }

        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = wrapper.apply_anthropic_overrides(kwargs, "root.llm")

        assert result["system"] == "You are a pirate."
        assert result["messages"] == kwargs["messages"]
        # Original should not be modified
        assert kwargs["system"] == "You are a helpful assistant."

    def test_system_override_updates_span(self, wrapper):
        """Test that system override also updates span attributes."""
        wrapper._initialized = True
        wrapper._overrides = {
            "root.llm": {"system": "New system prompt"},
        }

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.attributes = {
            "gen_ai.input.messages": json.dumps([
                {"role": "system", "content": "Old system prompt"},
                {"role": "user", "content": "Hello"},
            ]),
        }

        kwargs = {
            "system": "Old system prompt",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        wrapper.apply_anthropic_overrides(kwargs, "root.llm", span=mock_span)

        mock_span.set_attribute.assert_called_once()
        call_args = mock_span.set_attribute.call_args
        assert call_args[0][0] == "gen_ai.input.messages"
        updated_messages = json.loads(call_args[0][1])
        assert updated_messages[0]["content"] == "New system prompt"

    def test_tool_overrides(self, wrapper):
        """Test tool definition overrides."""
        wrapper._initialized = True
        wrapper._overrides = {
            "root.llm": {
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather (updated)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                            },
                        },
                    },
                ],
            },
        }

        kwargs = {
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                    },
                },
            ],
        }

        result = wrapper.apply_anthropic_overrides(kwargs, "root.llm")

        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tools"][0]["description"] == "Get weather (updated)"
        # parameters from override become input_schema for Anthropic
        assert result["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"

    def test_tool_override_add_new_tool(self, wrapper):
        """Test adding a new tool via override."""
        wrapper._initialized = True
        wrapper._overrides = {
            "root.llm": {
                "tools": [
                    {
                        "name": "new_tool",
                        "description": "A new tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"arg": {"type": "string"}},
                        },
                    },
                ],
            },
        }

        kwargs = {"messages": [{"role": "user", "content": "Test"}], "tools": []}

        result = wrapper.apply_anthropic_overrides(kwargs, "root.llm")

        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "new_tool"

    def test_no_overrides_returns_original(self, wrapper):
        """Test that no overrides returns original kwargs."""
        wrapper._initialized = True
        wrapper._overrides = {}

        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result = wrapper.apply_anthropic_overrides(kwargs, "root.llm")

        assert result is kwargs

    def test_combined_system_and_tool_overrides(self, wrapper):
        """Test applying both system and tool overrides."""
        wrapper._initialized = True
        wrapper._overrides = {
            "root.llm": {
                "system": "Be concise.",
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                    },
                ],
            },
        }

        kwargs = {
            "system": "Be verbose.",
            "messages": [{"role": "user", "content": "Search for something"}],
            "tools": [],
        }

        result = wrapper.apply_anthropic_overrides(kwargs, "root.llm")

        assert result["system"] == "Be concise."
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "search"


class TestWrapMessagesCreate:
    """Tests for wrap_messages_create()."""

    def test_not_in_rollout_mode(self, wrapper):
        """Test that wrapper passes through when not in rollout mode."""
        wrapped = Mock(return_value="result")

        with patch.object(wrapper, "should_use_rollout", return_value=False):
            result = wrapper.wrap_messages_create(
                wrapped, None, (), {}, span=None
            )

        wrapped.assert_called_once_with()
        assert result == "result"

    def test_no_span_path(self, wrapper):
        """Test that wrapper passes through when no span path is available."""
        wrapped = Mock(return_value="result")
        mock_span = MagicMock()
        mock_span.attributes = {}

        with patch.object(wrapper, "should_use_rollout", return_value=True):
            with patch.object(
                wrapper, "_get_span_path_from_span", return_value=None
            ):
                result = wrapper.wrap_messages_create(
                    wrapped, None, (), {}, span=mock_span
                )

        wrapped.assert_called_once_with()
        assert result == "result"

    def test_cache_hit_returns_cached_response(self, wrapper):
        """Test that cached response is returned on cache hit."""
        wrapped = Mock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.attributes = {"lmnr.span.path": ("root", "llm")}

        cached_span = {
            "attributes": {
                "lmnr.sdk.raw.response": json.dumps({
                    "id": "msg_cached",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [{"type": "text", "text": "Cached"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }),
            },
        }

        with patch.object(wrapper, "should_use_rollout", return_value=True):
            with patch.object(
                wrapper, "_get_span_path_from_span", return_value="root.llm"
            ):
                with patch.object(
                    wrapper, "get_current_index_for_path", return_value=0
                ):
                    with patch.object(
                        wrapper, "should_use_cache", return_value=True
                    ):
                        with patch.object(
                            wrapper, "get_cached_response", return_value=cached_span
                        ):
                            result = wrapper.wrap_messages_create(
                                wrapped, None, (), {}, span=mock_span
                            )

        wrapped.assert_not_called()
        assert isinstance(result, Message)
        assert result.content[0].text == "Cached"
        mock_span.set_attributes.assert_called_once()

    def test_cache_miss_applies_overrides(self, wrapper):
        """Test that overrides are applied on cache miss."""
        wrapped = Mock(return_value=Mock(spec=Message))
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.attributes = {"lmnr.span.path": ("root", "llm")}

        wrapper._initialized = True
        wrapper._overrides = {
            "root.llm": {"system": "Override prompt"},
        }

        kwargs = {
            "system": "Original prompt",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        with patch.object(wrapper, "should_use_rollout", return_value=True):
            with patch.object(
                wrapper, "_get_span_path_from_span", return_value="root.llm"
            ):
                with patch.object(
                    wrapper, "get_current_index_for_path", return_value=0
                ):
                    with patch.object(
                        wrapper, "should_use_cache", return_value=False
                    ):
                        result = wrapper.wrap_messages_create(
                            wrapped, None, (), kwargs, span=mock_span
                        )

        call_kwargs = wrapped.call_args[1]
        assert call_kwargs["system"] == "Override prompt"

    def test_cache_hit_streaming(self, wrapper):
        """Test cached streaming response."""
        wrapped = Mock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.attributes = {"lmnr.span.path": ("root", "llm")}

        cached_span = {
            "attributes": {
                "lmnr.sdk.raw.response": json.dumps({
                    "id": "msg_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [{"type": "text", "text": "Streamed cached"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }),
            },
        }

        with patch.object(wrapper, "should_use_rollout", return_value=True):
            with patch.object(
                wrapper, "_get_span_path_from_span", return_value="root.llm"
            ):
                with patch.object(
                    wrapper, "get_current_index_for_path", return_value=0
                ):
                    with patch.object(
                        wrapper, "should_use_cache", return_value=True
                    ):
                        with patch.object(
                            wrapper, "get_cached_response", return_value=cached_span
                        ):
                            result = wrapper.wrap_messages_create(
                                wrapped,
                                None,
                                (),
                                {},
                                span=mock_span,
                                is_streaming=True,
                            )

        # Should be a generator for streaming
        events = list(result)
        assert len(events) > 0

        # Check event types
        event_types = [e.type for e in events]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types


class TestGetSpanPathFromSpan:
    """Tests for _get_span_path_from_span()."""

    def test_with_span_path(self, wrapper):
        """Test extracting path from span attributes."""
        mock_span = MagicMock()
        mock_span.attributes = {"lmnr.span.path": ("root", "child", "llm")}

        result = wrapper._get_span_path_from_span(mock_span)
        assert result == "root.child.llm"

    def test_no_span_path(self, wrapper):
        """Test when span has no path attribute."""
        mock_span = MagicMock()
        mock_span.attributes = {}

        result = wrapper._get_span_path_from_span(mock_span)
        assert result is None

    def test_empty_span_path(self, wrapper):
        """Test when span path is empty."""
        mock_span = MagicMock()
        mock_span.attributes = {"lmnr.span.path": ()}

        result = wrapper._get_span_path_from_span(mock_span)
        assert result is None


class TestGetAnthropicRolloutWrapper:
    """Tests for get_anthropic_rollout_wrapper()."""

    def test_returns_none_when_not_in_rollout(self):
        """Test returns None when not in rollout mode."""
        with patch(
            "lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic.rollout.is_rollout_mode",
            return_value=False,
        ):
            result = get_anthropic_rollout_wrapper()
            assert result is None

    def test_creates_singleton(self):
        """Test creates and reuses singleton wrapper."""
        import lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic.rollout as rollout_module

        # Reset singleton
        rollout_module._anthropic_rollout_wrapper = None

        with patch(
            "lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic.rollout.is_rollout_mode",
            return_value=True,
        ):
            wrapper1 = get_anthropic_rollout_wrapper()
            wrapper2 = get_anthropic_rollout_wrapper()

            assert wrapper1 is not None
            assert wrapper1 is wrapper2

        # Clean up singleton
        rollout_module._anthropic_rollout_wrapper = None


class TestCachedStream:
    """Tests for _create_cached_stream()."""

    def test_text_content_stream(self, wrapper):
        """Test streaming a cached text response."""
        response = Message(
            id="msg_test",
            type="message",
            role="assistant",
            model="claude-sonnet-4-20250514",
            content=[TextBlock(type="text", text="Hello world")],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

        events = list(wrapper._create_cached_stream(response))

        assert len(events) == 5  # start, block_start, block_delta, msg_delta, msg_stop
        assert events[0].type == "message_start"
        assert events[1].type == "content_block_start"
        assert events[2].type == "content_block_delta"
        assert events[3].type == "message_delta"
        assert events[4].type == "message_stop"

    def test_tool_use_stream(self, wrapper):
        """Test streaming a cached tool use response."""
        response = Message(
            id="msg_tool",
            type="message",
            role="assistant",
            model="claude-sonnet-4-20250514",
            content=[
                ToolUseBlock(
                    type="tool_use",
                    id="tool_123",
                    name="get_weather",
                    input={"location": "NYC"},
                ),
            ],
            stop_reason="tool_use",
            stop_sequence=None,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

        events = list(wrapper._create_cached_stream(response))

        assert len(events) == 5
        assert events[1].type == "content_block_start"
        assert events[1].content_block.type == "tool_use"
        assert events[2].type == "content_block_delta"

    def test_multiple_content_blocks_stream(self, wrapper):
        """Test streaming a response with multiple content blocks."""
        response = Message(
            id="msg_multi",
            type="message",
            role="assistant",
            model="claude-sonnet-4-20250514",
            content=[
                TextBlock(type="text", text="Let me search for that."),
                ToolUseBlock(
                    type="tool_use",
                    id="tool_789",
                    name="search",
                    input={"query": "test"},
                ),
            ],
            stop_reason="tool_use",
            stop_sequence=None,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

        events = list(wrapper._create_cached_stream(response))

        # start + 2*(block_start + block_delta) + msg_delta + msg_stop = 7
        assert len(events) == 7
        assert events[0].type == "message_start"
        assert events[1].type == "content_block_start"
        assert events[2].type == "content_block_delta"
        assert events[3].type == "content_block_start"
        assert events[4].type == "content_block_delta"
        assert events[5].type == "message_delta"
        assert events[6].type == "message_stop"


class TestAsyncCachedStream:
    """Tests for _create_async_cached_stream()."""

    @pytest.mark.asyncio
    async def test_async_text_stream(self, wrapper):
        """Test async streaming of cached response."""
        response = Message(
            id="msg_async",
            type="message",
            role="assistant",
            model="claude-sonnet-4-20250514",
            content=[TextBlock(type="text", text="Async cached")],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

        events = []
        async for event in wrapper._create_async_cached_stream(response):
            events.append(event)

        assert len(events) == 5
        assert events[0].type == "message_start"
        assert events[4].type == "message_stop"

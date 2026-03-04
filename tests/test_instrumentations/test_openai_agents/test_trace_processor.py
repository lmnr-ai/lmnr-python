"""Unit tests for LaminarAgentsTraceProcessor and helper functions."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents import (
    LaminarAgentsTraceProcessor,
    _apply_llm_attributes,
    _apply_usage,
    _get_first_not_none,
    _get_attr_not_none,
    _map_span_type,
    _normalize_messages,
    _response_to_llm_data,
    _set_gen_ai_input_messages,
    _set_gen_ai_output_messages,
    _set_gen_ai_output_messages_from_response,
    _set_tool_definitions_from_response,
    _span_kind,
    _span_name,
    _apply_span_error,
    _apply_span_data,
    _agent_name,
    _model_as_dict,
)
from lmnr.opentelemetry_lib.tracing.attributes import Attributes


# ---------------------------------------------------------------------------
# Helpers for building fake Agents SDK objects
# ---------------------------------------------------------------------------


class FakeSpanData:
    def __init__(self, type_name: str, **kwargs):
        self._type = type_name
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def type(self):
        return self._type

    def export(self):
        d = {"type": self._type}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                d[k] = v
        return d


class FakeSpan:
    def __init__(self, trace_id="trace-1", span_id="span-1", parent_id=None,
                 name=None, span_data=None, error=None):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.name = name
        self.span_data = span_data
        self.error = error


class FakeTrace:
    def __init__(self, trace_id="trace-1", name="test-trace", metadata=None,
                 group_id=None):
        self.trace_id = trace_id
        self.name = name
        self.metadata = metadata
        self.group_id = group_id


class FakeUsage:
    def __init__(self, input_tokens=10, output_tokens=20, total_tokens=30):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens


class FakeResponse:
    def __init__(self, id="resp-1", model="gpt-4o", usage=None,
                 output=None, tools=None, output_text=None):
        self.id = id
        self.model = model
        self.usage = usage
        self.output = output or []
        self.tools = tools
        self.output_text = output_text


class FakeOutputMessage:
    def __init__(self, role="assistant", content=None, type="message"):
        self.type = type
        self.role = role
        self.content = content or []


class FakeTextContent:
    def __init__(self, text="Hello"):
        self.type = "output_text"
        self.text = text


class FakeFunctionCall:
    def __init__(self, call_id="call-1", name="get_weather", arguments='{"city": "NYC"}'):
        self.type = "function_call"
        self.call_id = call_id
        self.name = name
        self.arguments = arguments

    def model_dump(self):
        return {
            "type": self.type,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }


class FakeTool:
    def __init__(self, name="get_weather", description="Get weather",
                 parameters=None, strict=None):
        self.type = "function"
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}}
        self.strict = strict

    def model_dump(self):
        return {
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
        }


class FakeSpanError:
    def __init__(self, message="Something went wrong", data=None):
        self.message = message
        self.data = data


# ---------------------------------------------------------------------------
# Test _get_first_not_none and _get_attr_not_none
# ---------------------------------------------------------------------------


class TestGetFirstNotNone:
    """Tests for _get_first_not_none - the fix for the token count bug."""

    def test_returns_first_found(self):
        d = {"a": 10, "b": 20}
        assert _get_first_not_none(d, "a", "b") == 10

    def test_falls_through_none(self):
        d = {"a": None, "b": 20}
        assert _get_first_not_none(d, "a", "b") == 20

    def test_zero_is_valid(self):
        """Zero should NOT be treated as missing - this was the original bug."""
        d = {"input_tokens": 0, "prompt_tokens": 100}
        assert _get_first_not_none(d, "input_tokens", "prompt_tokens") == 0

    def test_all_none(self):
        d = {"a": None}
        assert _get_first_not_none(d, "a", "b") is None

    def test_missing_keys(self):
        d = {}
        assert _get_first_not_none(d, "a", "b") is None


class TestGetAttrNotNone:
    def test_returns_first_found(self):
        obj = FakeUsage(input_tokens=10)
        assert _get_attr_not_none(obj, "input_tokens") == 10

    def test_zero_is_valid(self):
        obj = FakeUsage(input_tokens=0, output_tokens=20)
        assert _get_attr_not_none(obj, "input_tokens") == 0

    def test_falls_through_none(self):
        class Obj:
            input_tokens = None
            prompt_tokens = 50
        assert _get_attr_not_none(Obj(), "input_tokens", "prompt_tokens") == 50


# ---------------------------------------------------------------------------
# Test _apply_usage
# ---------------------------------------------------------------------------


class TestApplyUsage:
    def test_dict_usage_with_input_tokens(self):
        span = MagicMock()
        usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        _apply_usage(span, usage)
        span.set_attribute.assert_any_call(Attributes.INPUT_TOKEN_COUNT.value, 10)
        span.set_attribute.assert_any_call(Attributes.OUTPUT_TOKEN_COUNT.value, 20)
        span.set_attribute.assert_any_call(Attributes.TOTAL_TOKEN_COUNT.value, 30)

    def test_dict_usage_with_prompt_tokens(self):
        """Should fall back to prompt_tokens/completion_tokens."""
        span = MagicMock()
        usage = {"prompt_tokens": 15, "completion_tokens": 25}
        _apply_usage(span, usage)
        span.set_attribute.assert_any_call(Attributes.INPUT_TOKEN_COUNT.value, 15)
        span.set_attribute.assert_any_call(Attributes.OUTPUT_TOKEN_COUNT.value, 25)
        # total should be computed
        span.set_attribute.assert_any_call(Attributes.TOTAL_TOKEN_COUNT.value, 40)

    def test_object_usage(self):
        span = MagicMock()
        usage = FakeUsage(input_tokens=5, output_tokens=10, total_tokens=15)
        _apply_usage(span, usage)
        span.set_attribute.assert_any_call(Attributes.INPUT_TOKEN_COUNT.value, 5)
        span.set_attribute.assert_any_call(Attributes.OUTPUT_TOKEN_COUNT.value, 10)
        span.set_attribute.assert_any_call(Attributes.TOTAL_TOKEN_COUNT.value, 15)

    def test_zero_tokens_are_preserved(self):
        """Zero token counts should be set, not ignored."""
        span = MagicMock()
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        _apply_usage(span, usage)
        span.set_attribute.assert_any_call(Attributes.INPUT_TOKEN_COUNT.value, 0)
        span.set_attribute.assert_any_call(Attributes.OUTPUT_TOKEN_COUNT.value, 0)
        span.set_attribute.assert_any_call(Attributes.TOTAL_TOKEN_COUNT.value, 0)

    def test_none_usage(self):
        span = MagicMock()
        _apply_usage(span, None)
        span.set_attribute.assert_not_called()


# ---------------------------------------------------------------------------
# Test _span_kind, _span_name, _map_span_type
# ---------------------------------------------------------------------------


class TestSpanClassification:
    def test_span_kind_none(self):
        assert _span_kind(None) == ""

    def test_span_kind_generation(self):
        sd = FakeSpanData("generation")
        assert _span_kind(sd) == "generation"

    def test_span_name_from_span(self):
        span = FakeSpan(name="my-span")
        assert _span_name(span, None) == "my-span"

    def test_span_name_from_span_data(self):
        sd = FakeSpanData("response")
        span = FakeSpan(name=None)
        assert _span_name(span, sd) == "agents.response"

    def test_span_name_fallback(self):
        span = FakeSpan(name=None)
        assert _span_name(span, None) == "agents.span"

    def test_map_span_type_llm(self):
        for kind in ["generation", "response", "transcription", "speech", "speech_group"]:
            sd = FakeSpanData(kind)
            assert _map_span_type(sd) == "LLM"

    def test_map_span_type_tool(self):
        for kind in ["function", "tool", "mcp_list_tools", "mcp_tools"]:
            sd = FakeSpanData(kind)
            assert _map_span_type(sd) == "TOOL"

    def test_map_span_type_default(self):
        sd = FakeSpanData("agent")
        assert _map_span_type(sd) == "DEFAULT"

        sd = FakeSpanData("handoff")
        assert _map_span_type(sd) == "DEFAULT"


# ---------------------------------------------------------------------------
# Test _normalize_messages
# ---------------------------------------------------------------------------


class TestNormalizeMessages:
    def test_string_input(self):
        result = _normalize_messages("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_list_of_dicts(self):
        msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        result = _normalize_messages(msgs)
        assert result == msgs

    def test_none_input(self):
        assert _normalize_messages(None) == []

    def test_single_dict(self):
        result = _normalize_messages({"role": "user", "content": "Hello"})
        assert result == [{"role": "user", "content": "Hello"}]

    def test_list_with_model_dump_objects(self):
        class Msg:
            def model_dump(self):
                return {"role": "user", "content": "test"}
        result = _normalize_messages([Msg()])
        assert result == [{"role": "user", "content": "test"}]


# ---------------------------------------------------------------------------
# Test _set_gen_ai_input_messages / _set_gen_ai_output_messages
# ---------------------------------------------------------------------------


class TestSetGenAiMessages:
    def test_set_input_messages_string(self):
        span = MagicMock()
        _set_gen_ai_input_messages(span, "Hello world")
        span.set_attribute.assert_called_once()
        call = span.set_attribute.call_args
        assert call[0][0] == "gen_ai.input.messages"
        parsed = json.loads(call[0][1])
        assert parsed == [{"role": "user", "content": "Hello world"}]

    def test_set_input_messages_list(self):
        span = MagicMock()
        msgs = [{"role": "user", "content": "Hi"}]
        _set_gen_ai_input_messages(span, msgs)
        call = span.set_attribute.call_args
        parsed = json.loads(call[0][1])
        assert parsed == msgs

    def test_set_output_messages(self):
        span = MagicMock()
        msgs = [{"role": "assistant", "content": "Hello!"}]
        _set_gen_ai_output_messages(span, msgs)
        call = span.set_attribute.call_args
        assert call[0][0] == "gen_ai.output.messages"
        parsed = json.loads(call[0][1])
        assert parsed == msgs

    @patch.dict(os.environ, {"LMNR_SUPPRESS_INPUTS": "1"})
    def test_suppress_inputs(self):
        # Need to reload to pick up env var
        import importlib
        import lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents as mod
        original = mod.SUPPRESS_INPUTS
        mod.SUPPRESS_INPUTS = True
        try:
            span = MagicMock()
            mod._set_gen_ai_input_messages(span, "Hello")
            span.set_attribute.assert_not_called()
        finally:
            mod.SUPPRESS_INPUTS = original

    @patch.dict(os.environ, {"LMNR_SUPPRESS_OUTPUTS": "1"})
    def test_suppress_outputs(self):
        import lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents as mod
        original = mod.SUPPRESS_OUTPUTS
        mod.SUPPRESS_OUTPUTS = True
        try:
            span = MagicMock()
            mod._set_gen_ai_output_messages(span, "output")
            span.set_attribute.assert_not_called()
        finally:
            mod.SUPPRESS_OUTPUTS = original


# ---------------------------------------------------------------------------
# Test _set_gen_ai_output_messages_from_response
# ---------------------------------------------------------------------------


class TestOutputMessagesFromResponse:
    def test_text_response(self):
        span = MagicMock()
        content = FakeTextContent("Hello from AI")
        msg = FakeOutputMessage(content=[content])
        response = FakeResponse(output=[msg])
        _set_gen_ai_output_messages_from_response(span, response)

        call = span.set_attribute.call_args
        assert call[0][0] == "gen_ai.output.messages"
        parsed = json.loads(call[0][1])
        assert len(parsed) == 1
        assert parsed[0]["role"] == "assistant"
        assert parsed[0]["content"] == "Hello from AI"

    def test_function_call_response(self):
        span = MagicMock()
        func_call = FakeFunctionCall()
        response = FakeResponse(output=[func_call])
        _set_gen_ai_output_messages_from_response(span, response)

        call = span.set_attribute.call_args
        parsed = json.loads(call[0][1])
        assert len(parsed) == 1
        assert parsed[0]["role"] == "assistant"
        assert parsed[0]["content"] is None
        assert len(parsed[0]["tool_calls"]) == 1
        tc = parsed[0]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "NYC"}'

    def test_empty_response(self):
        span = MagicMock()
        response = FakeResponse(output=[])
        _set_gen_ai_output_messages_from_response(span, response)
        span.set_attribute.assert_not_called()

    def test_none_response(self):
        span = MagicMock()
        _set_gen_ai_output_messages_from_response(span, None)
        span.set_attribute.assert_not_called()


# ---------------------------------------------------------------------------
# Test _set_tool_definitions_from_response
# ---------------------------------------------------------------------------


class TestToolDefinitions:
    def test_function_tools(self):
        span = MagicMock()
        tool = FakeTool(
            name="get_weather",
            description="Get weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        response = FakeResponse(tools=[tool])
        _set_tool_definitions_from_response(span, response)

        call = span.set_attribute.call_args
        assert call[0][0] == "gen_ai.tool.definitions"
        parsed = json.loads(call[0][1])
        assert len(parsed) == 1
        assert parsed[0]["type"] == "function"
        assert parsed[0]["function"]["name"] == "get_weather"
        assert parsed[0]["function"]["description"] == "Get weather for a city"
        assert "properties" in parsed[0]["function"]["parameters"]

    def test_no_tools(self):
        span = MagicMock()
        response = FakeResponse(tools=None)
        _set_tool_definitions_from_response(span, response)
        span.set_attribute.assert_not_called()

    def test_empty_tools(self):
        span = MagicMock()
        response = FakeResponse(tools=[])
        _set_tool_definitions_from_response(span, response)
        span.set_attribute.assert_not_called()

    def test_strict_tools(self):
        span = MagicMock()
        tool = FakeTool(name="calc", description="Calculate", strict=True)
        response = FakeResponse(tools=[tool])
        _set_tool_definitions_from_response(span, response)

        parsed = json.loads(span.set_attribute.call_args[0][1])
        assert parsed[0]["function"]["strict"] is True


# ---------------------------------------------------------------------------
# Test _apply_llm_attributes
# ---------------------------------------------------------------------------


class TestApplyLlmAttributes:
    def test_full_attributes(self):
        span = MagicMock()
        data = {
            "model": "gpt-4o",
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "response_id": "resp-123",
        }
        _apply_llm_attributes(span, data)

        span.set_attribute.assert_any_call(Attributes.REQUEST_MODEL.value, "gpt-4o")
        span.set_attribute.assert_any_call(Attributes.RESPONSE_MODEL.value, "gpt-4o")
        span.set_attribute.assert_any_call(Attributes.PROVIDER.value, "openai")
        span.set_attribute.assert_any_call(Attributes.INPUT_TOKEN_COUNT.value, 10)
        span.set_attribute.assert_any_call(Attributes.OUTPUT_TOKEN_COUNT.value, 20)
        span.set_attribute.assert_any_call(Attributes.TOTAL_TOKEN_COUNT.value, 30)
        span.set_attribute.assert_any_call(Attributes.RESPONSE_ID.value, "resp-123")

    def test_none_data(self):
        span = MagicMock()
        _apply_llm_attributes(span, None)
        span.set_attribute.assert_not_called()


# ---------------------------------------------------------------------------
# Test _response_to_llm_data
# ---------------------------------------------------------------------------


class TestResponseToLlmData:
    def test_with_usage_object(self):
        usage = FakeUsage(input_tokens=100, output_tokens=200, total_tokens=300)
        response = FakeResponse(id="r-1", model="gpt-4o", usage=usage)
        data = _response_to_llm_data(response)
        assert data["model"] == "gpt-4o"
        assert data["response_id"] == "r-1"
        assert data["usage"]["input_tokens"] == 100
        assert data["usage"]["output_tokens"] == 200
        assert data["usage"]["total_tokens"] == 300

    def test_with_none_response(self):
        assert _response_to_llm_data(None) == {}

    def test_with_no_usage(self):
        response = FakeResponse(id="r-1", model="gpt-4o", usage=None)
        data = _response_to_llm_data(response)
        assert data["usage"] is None


# ---------------------------------------------------------------------------
# Test _apply_span_error
# ---------------------------------------------------------------------------


class TestApplySpanError:
    def test_with_error(self):
        lmnr_span = MagicMock()
        error = FakeSpanError("Test error")
        span = FakeSpan(error=error)
        _apply_span_error(lmnr_span, span)
        lmnr_span.set_status.assert_called_once()
        args = lmnr_span.set_status.call_args[0]
        assert args[0].status_code.name == "ERROR"
        assert "Test error" in args[0].description

    def test_without_error(self):
        lmnr_span = MagicMock()
        span = FakeSpan(error=None)
        _apply_span_error(lmnr_span, span)
        lmnr_span.set_status.assert_not_called()

    def test_with_string_error(self):
        """SpanError might be represented differently."""
        lmnr_span = MagicMock()
        span = FakeSpan()
        span.error = "raw error string"
        _apply_span_error(lmnr_span, span)
        lmnr_span.set_status.assert_called_once()


# ---------------------------------------------------------------------------
# Test _apply_span_data for various span types
# ---------------------------------------------------------------------------


class TestApplySpanDataAgent:
    def test_agent_span_data(self):
        span = MagicMock()
        sd = FakeSpanData("agent", name="TestAgent",
                          handoffs=["Agent2"], tools=["tool1"], output_type="str")
        _apply_span_data(span, sd)
        span.set_attribute.assert_any_call("openai.agents.agent.name", "TestAgent")
        # Check handoffs and tools are serialized
        calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert "openai.agents.agent.handoffs" in calls
        assert "openai.agents.agent.tools" in calls


class TestApplySpanDataFunction:
    def test_function_span_data(self):
        span = MagicMock()
        sd = FakeSpanData("function", name="get_weather",
                          input='{"city": "NYC"}', output="72F")
        _apply_span_data(span, sd)
        span.set_attribute.assert_any_call("openai.agents.tool.name", "get_weather")
        # Should set gen_ai.input/output.messages
        calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert "gen_ai.input.messages" in calls
        assert "gen_ai.output.messages" in calls


class TestApplySpanDataGeneration:
    def test_generation_with_usage(self):
        span = MagicMock()
        sd = FakeSpanData(
            "generation",
            input=[{"role": "user", "content": "Hello"}],
            output=[{"role": "assistant", "content": "Hi there!"}],
            model="gpt-4o",
            model_config={},
            usage={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
        )
        _apply_span_data(span, sd)

        calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert "gen_ai.input.messages" in calls
        assert "gen_ai.output.messages" in calls
        assert calls[Attributes.REQUEST_MODEL.value] == "gpt-4o"
        assert calls[Attributes.INPUT_TOKEN_COUNT.value] == 5
        assert calls[Attributes.OUTPUT_TOKEN_COUNT.value] == 10

    def test_generation_with_zero_usage(self):
        """Verify zero tokens are correctly reported (the PR bug fix)."""
        span = MagicMock()
        sd = FakeSpanData(
            "generation",
            input=[],
            output=[],
            model="gpt-4o",
            model_config={},
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        _apply_span_data(span, sd)

        calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert calls[Attributes.INPUT_TOKEN_COUNT.value] == 0
        assert calls[Attributes.OUTPUT_TOKEN_COUNT.value] == 0
        assert calls[Attributes.TOTAL_TOKEN_COUNT.value] == 0


class TestApplySpanDataResponse:
    def test_response_with_usage_and_tools(self):
        span = MagicMock()
        usage = FakeUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        content = FakeTextContent("Hello!")
        msg = FakeOutputMessage(content=[content])
        tool = FakeTool(name="calculator")
        response = FakeResponse(
            id="resp-1", model="gpt-4o",
            usage=usage, output=[msg], tools=[tool]
        )
        sd = FakeSpanData("response")
        sd.response = response
        sd.input = [{"role": "user", "content": "What is 2+2?"}]

        _apply_span_data(span, sd)

        calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert "gen_ai.input.messages" in calls
        assert "gen_ai.output.messages" in calls
        assert "gen_ai.tool.definitions" in calls
        assert calls[Attributes.INPUT_TOKEN_COUNT.value] == 100
        assert calls[Attributes.OUTPUT_TOKEN_COUNT.value] == 50
        assert calls[Attributes.RESPONSE_ID.value] == "resp-1"
        assert calls[Attributes.REQUEST_MODEL.value] == "gpt-4o"


class TestApplySpanDataHandoff:
    def test_handoff(self):
        span = MagicMock()
        sd = FakeSpanData("handoff", from_agent="Agent1", to_agent="Agent2")
        _apply_span_data(span, sd)
        span.set_attribute.assert_any_call("openai.agents.handoff.from", "Agent1")
        span.set_attribute.assert_any_call("openai.agents.handoff.to", "Agent2")


class TestApplySpanDataGuardrail:
    def test_guardrail(self):
        span = MagicMock()
        sd = FakeSpanData("guardrail", name="profanity_check", triggered=True)
        _apply_span_data(span, sd)
        span.set_attribute.assert_any_call("openai.agents.guardrail.name", "profanity_check")
        span.set_attribute.assert_any_call("openai.agents.guardrail.triggered", True)


class TestApplySpanDataCustom:
    def test_custom(self):
        span = MagicMock()
        sd = FakeSpanData("custom", name="my_custom", data={"key": "value"})
        _apply_span_data(span, sd)
        span.set_attribute.assert_any_call("openai.agents.custom.name", "my_custom")
        calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert "openai.agents.custom.data" in calls


class TestApplySpanDataMcp:
    def test_mcp_list_tools(self):
        span = MagicMock()
        sd = FakeSpanData("mcp_list_tools", server="my-server",
                          result=["tool1", "tool2"])
        _apply_span_data(span, sd)
        span.set_attribute.assert_any_call("openai.agents.mcp.server", "my-server")
        calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert "openai.agents.mcp.result" in calls


# ---------------------------------------------------------------------------
# Test _agent_name helper
# ---------------------------------------------------------------------------


class TestAgentName:
    def test_dict_with_name(self):
        assert _agent_name({"name": "Agent1"}) == "Agent1"

    def test_string(self):
        assert _agent_name("Agent1") == "Agent1"

    def test_object_with_name(self):
        class Obj:
            name = "Agent1"
        assert _agent_name(Obj()) == "Agent1"

    def test_empty(self):
        assert _agent_name(None) == ""
        assert _agent_name({}) == ""


# ---------------------------------------------------------------------------
# Test _model_as_dict
# ---------------------------------------------------------------------------


class TestModelAsDict:
    def test_dict_passthrough(self):
        d = {"a": 1}
        assert _model_as_dict(d) is d

    def test_none(self):
        assert _model_as_dict(None) is None

    def test_model_dump(self):
        class Obj:
            def model_dump(self):
                return {"b": 2}
        assert _model_as_dict(Obj()) == {"b": 2}

    def test_dict_method(self):
        class Obj:
            def dict(self):
                return {"c": 3}
        assert _model_as_dict(Obj()) == {"c": 3}

    def test_fallback_to_dict(self):
        class Obj:
            def __init__(self):
                self.x = 1
                self._private = 2
        result = _model_as_dict(Obj())
        assert result == {"x": 1}


# ---------------------------------------------------------------------------
# Test LaminarAgentsTraceProcessor
# ---------------------------------------------------------------------------


class TestTraceProcessor:
    """Integration tests for the trace processor lifecycle."""

    @patch("lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.Laminar")
    def test_trace_start_creates_root_span(self, mock_laminar):
        mock_span = MagicMock()
        mock_laminar.start_span.return_value = mock_span

        processor = LaminarAgentsTraceProcessor()
        trace = FakeTrace(trace_id="t-1", name="my-trace")
        processor.on_trace_start(trace)

        assert "t-1" in processor._traces
        mock_laminar.start_span.assert_called()

    @patch("lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.Laminar")
    def test_trace_end_closes_spans(self, mock_laminar):
        mock_root_span = MagicMock()
        mock_laminar.start_span.return_value = mock_root_span

        processor = LaminarAgentsTraceProcessor()
        trace = FakeTrace(trace_id="t-1")
        processor.on_trace_start(trace)
        processor.on_trace_end(trace)

        assert "t-1" not in processor._traces
        mock_root_span.end.assert_called()

    @patch("lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.Laminar")
    def test_span_lifecycle(self, mock_laminar):
        mock_root = MagicMock()
        mock_child = MagicMock()
        mock_laminar.start_span.side_effect = [mock_root, mock_child]

        processor = LaminarAgentsTraceProcessor()
        trace = FakeTrace(trace_id="t-1")
        processor.on_trace_start(trace)

        span = FakeSpan(
            trace_id="t-1", span_id="s-1", name="test",
            span_data=FakeSpanData("agent", name="TestAgent",
                                   handoffs=[], tools=[], output_type="str"),
        )
        processor.on_span_start(span)
        assert "s-1" in processor._traces["t-1"].spans

        processor.on_span_end(span)
        assert "s-1" not in processor._traces["t-1"].spans
        mock_child.end.assert_called()

    @patch("lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.Laminar")
    def test_span_without_span_id_uses_name(self, mock_laminar):
        """Fix for the PR bug: span_id fallback should be consistent."""
        mock_root = MagicMock()
        mock_child = MagicMock()
        mock_laminar.start_span.side_effect = [mock_root, mock_child]

        processor = LaminarAgentsTraceProcessor()
        trace = FakeTrace(trace_id="t-1")
        processor.on_trace_start(trace)

        span = FakeSpan(
            trace_id="t-1", span_id=None, name="my-span",
            span_data=FakeSpanData("agent", name="A",
                                   handoffs=[], tools=[], output_type=None),
        )
        processor.on_span_start(span)
        # Should be stored under the name
        assert "my-span" in processor._traces["t-1"].spans

        processor.on_span_end(span)
        # Should find and close it
        mock_child.end.assert_called()

    @patch("lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.Laminar")
    def test_shutdown_cleans_up(self, mock_laminar):
        mock_root = MagicMock()
        mock_laminar.start_span.return_value = mock_root
        mock_laminar.flush.return_value = True

        processor = LaminarAgentsTraceProcessor()
        trace = FakeTrace(trace_id="t-1")
        processor.on_trace_start(trace)
        processor.shutdown()

        assert len(processor._traces) == 0
        mock_root.end.assert_called()
        mock_laminar.flush.assert_called()

    @patch("lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.Laminar")
    def test_trace_metadata_applied(self, mock_laminar):
        mock_root = MagicMock()
        mock_root.set_trace_metadata = MagicMock()
        mock_root.set_trace_session_id = MagicMock()
        mock_root.set_trace_user_id = MagicMock()
        mock_laminar.start_span.return_value = mock_root

        processor = LaminarAgentsTraceProcessor()
        trace = FakeTrace(
            trace_id="t-1", name="test",
            metadata={"session_id": "sess-1", "user_id": "user-1"},
            group_id="g-1",
        )
        processor.on_trace_start(trace)

        mock_root.set_trace_metadata.assert_called_once()
        metadata = mock_root.set_trace_metadata.call_args[0][0]
        assert metadata["openai.agents.trace_id"] == "t-1"
        assert metadata["openai.agents.group_id"] == "g-1"
        mock_root.set_trace_session_id.assert_called_with("sess-1")
        mock_root.set_trace_user_id.assert_called_with("user-1")

    @patch("lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.Laminar")
    def test_no_trace_id_is_ignored(self, mock_laminar):
        processor = LaminarAgentsTraceProcessor()
        trace = FakeTrace(trace_id=None)
        processor.on_trace_start(trace)
        mock_laminar.start_span.assert_not_called()

    @patch("lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.Laminar")
    def test_parent_span_hierarchy(self, mock_laminar):
        mock_root = MagicMock()
        mock_ctx = MagicMock()
        mock_root.get_laminar_span_context.return_value = mock_ctx

        mock_child1 = MagicMock()
        mock_child1.get_laminar_span_context.return_value = MagicMock()
        mock_child2 = MagicMock()

        mock_laminar.start_span.side_effect = [mock_root, mock_child1, mock_child2]

        processor = LaminarAgentsTraceProcessor()
        trace = FakeTrace(trace_id="t-1")
        processor.on_trace_start(trace)

        # Agent span (child of root)
        agent_span = FakeSpan(
            trace_id="t-1", span_id="agent-1", name="Agent",
            span_data=FakeSpanData("agent", name="A",
                                   handoffs=[], tools=[], output_type=None),
        )
        processor.on_span_start(agent_span)

        # Response span (child of agent)
        response_span = FakeSpan(
            trace_id="t-1", span_id="resp-1", parent_id="agent-1",
            name="openai.response",
            span_data=FakeSpanData("response"),
        )
        processor.on_span_start(response_span)

        # The response span should have been created with agent span's context
        calls = mock_laminar.start_span.call_args_list
        assert len(calls) == 3
        # The third call (response span) should have parent_span_context
        assert calls[2][1]["parent_span_context"] == mock_child1.get_laminar_span_context()

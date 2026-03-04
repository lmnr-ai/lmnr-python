"""Integration tests for OpenAI Agents SDK instrumentation.

These tests verify the full pipeline: Agent SDK -> Trace Processor -> Laminar spans.
They use VCR to record/replay actual OpenAI API calls.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents import (
    LaminarAgentsTraceProcessor,
    _apply_span_data,
    _apply_span_error,
)
from lmnr.opentelemetry_lib.tracing.attributes import Attributes

# Import Agents SDK types for building realistic test data
from agents import (
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    HandoffSpanData,
    GuardrailSpanData,
    CustomSpanData,
    MCPListToolsSpanData,
)

try:
    from agents import SpeechSpanData, TranscriptionSpanData, SpeechGroupSpanData
    HAS_REALTIME = True
except ImportError:
    HAS_REALTIME = False


class FakeSpan:
    """Mimics agents.tracing.spans.SpanImpl for testing."""
    def __init__(self, trace_id="t-1", span_id="s-1", parent_id=None,
                 name=None, span_data=None, error=None):
        self._trace_id = trace_id
        self._span_id = span_id
        self._parent_id = parent_id
        self._name = name
        self._span_data = span_data
        self._error = error

    @property
    def trace_id(self):
        return self._trace_id

    @property
    def span_id(self):
        return self._span_id

    @property
    def parent_id(self):
        return self._parent_id

    @property
    def name(self):
        return self._name

    @property
    def span_data(self):
        return self._span_data

    @property
    def error(self):
        return self._error


class FakeTrace:
    def __init__(self, trace_id="t-1", name="test", metadata=None, group_id=None):
        self.trace_id = trace_id
        self.name = name
        self.metadata = metadata
        self.group_id = group_id


class FakeResponseUsage:
    """Mimics openai.types.responses.ResponseUsage."""
    def __init__(self, input_tokens=50, output_tokens=100, total_tokens=150):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens


class FakeResponse:
    """Mimics openai.types.responses.Response."""
    def __init__(self, id="resp-abc", model="gpt-4o", usage=None,
                 output=None, tools=None, output_text=None):
        self.id = id
        self.model = model
        self.usage = usage
        self.output = output or []
        self.tools = tools
        self.output_text = output_text


class FakeOutputMessage:
    def __init__(self, role="assistant", content=None):
        self.type = "message"
        self.role = role
        self.content = content or []

    def model_dump(self):
        return {
            "type": self.type,
            "role": self.role,
            "content": [c.model_dump() if hasattr(c, 'model_dump') else c for c in self.content],
        }


class FakeTextContent:
    def __init__(self, text="Hello"):
        self.type = "output_text"
        self.text = text

    def model_dump(self):
        return {"type": self.type, "text": self.text}


class FakeFunctionCall:
    def __init__(self, call_id="call-1", name="get_weather", arguments='{"city": "NYC"}'):
        self.type = "function_call"
        self.call_id = call_id
        self.name = name
        self.arguments = arguments
        self.id = f"fc-{call_id}"

    def model_dump(self):
        return {
            "type": self.type,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "id": self.id,
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


# ---------------------------------------------------------------------------
# Test with real Agents SDK span data types
# ---------------------------------------------------------------------------


class TestAgentSpanDataIntegration:
    def test_agent_span_attributes(self):
        """Test that AgentSpanData is correctly processed."""
        lmnr_span = MagicMock()
        sd = AgentSpanData(
            name="AssistantAgent",
            handoffs=["HelperAgent"],
            tools=["search", "calculate"],
            output_type="str",
        )
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}
        assert calls["openai.agents.agent.name"] == "AssistantAgent"
        assert "openai.agents.agent.handoffs" in calls
        handoffs = json.loads(calls["openai.agents.agent.handoffs"])
        assert handoffs == ["HelperAgent"]
        assert "openai.agents.agent.tools" in calls
        tools = json.loads(calls["openai.agents.agent.tools"])
        assert tools == ["search", "calculate"]
        assert calls.get("openai.agents.agent.output_type") == "str"


class TestFunctionSpanDataIntegration:
    def test_function_span_attributes(self):
        """Test that FunctionSpanData is correctly processed."""
        lmnr_span = MagicMock()
        sd = FunctionSpanData(
            name="get_weather",
            input='{"city": "San Francisco"}',
            output="72°F, sunny",
        )
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}
        assert calls["openai.agents.tool.name"] == "get_weather"
        # Input and output should be in gen_ai messages
        assert "gen_ai.input.messages" in calls
        assert "gen_ai.output.messages" in calls
        input_msgs = json.loads(calls["gen_ai.input.messages"])
        assert '{"city": "San Francisco"}' in str(input_msgs)
        output_msgs = json.loads(calls["gen_ai.output.messages"])
        assert "72°F, sunny" in str(output_msgs)


class TestGenerationSpanDataIntegration:
    def test_generation_with_full_data(self):
        """Test GenerationSpanData with complete usage data."""
        lmnr_span = MagicMock()
        sd = GenerationSpanData(
            input=[{"role": "user", "content": "What is 2+2?"}],
            output=[{"role": "assistant", "content": "4"}],
            model="gpt-4o-mini",
            model_config={"temperature": 0.7},
            usage={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
        )
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}

        # Check gen_ai messages
        assert "gen_ai.input.messages" in calls
        input_msgs = json.loads(calls["gen_ai.input.messages"])
        assert input_msgs[0]["role"] == "user"
        assert input_msgs[0]["content"] == "What is 2+2?"

        assert "gen_ai.output.messages" in calls
        output_msgs = json.loads(calls["gen_ai.output.messages"])
        assert output_msgs[0]["role"] == "assistant"
        assert output_msgs[0]["content"] == "4"

        # Check LLM attributes
        assert calls[Attributes.REQUEST_MODEL.value] == "gpt-4o-mini"
        assert calls[Attributes.INPUT_TOKEN_COUNT.value] == 12
        assert calls[Attributes.OUTPUT_TOKEN_COUNT.value] == 3
        assert calls[Attributes.TOTAL_TOKEN_COUNT.value] == 15

    def test_generation_with_zero_tokens(self):
        """Verify the token count bug is fixed - zero should not fall through."""
        lmnr_span = MagicMock()
        sd = GenerationSpanData(
            input=[],
            output=[],
            model="gpt-4o",
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}
        assert calls[Attributes.INPUT_TOKEN_COUNT.value] == 0
        assert calls[Attributes.OUTPUT_TOKEN_COUNT.value] == 0
        assert calls[Attributes.TOTAL_TOKEN_COUNT.value] == 0

    def test_generation_with_none_usage(self):
        """When usage is None, no token attributes should be set."""
        lmnr_span = MagicMock()
        sd = GenerationSpanData(
            input=[],
            output=[],
            model="gpt-4o",
            usage=None,
        )
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}
        assert Attributes.INPUT_TOKEN_COUNT.value not in calls
        assert Attributes.OUTPUT_TOKEN_COUNT.value not in calls


class TestResponseSpanDataIntegration:
    def test_response_span_with_text_output(self):
        """Test ResponseSpanData with text output and usage."""
        from agents.tracing.span_data import ResponseSpanData

        lmnr_span = MagicMock()
        usage = FakeResponseUsage(input_tokens=25, output_tokens=50, total_tokens=75)
        content = FakeTextContent("The answer is 42.")
        msg = FakeOutputMessage(content=[content])
        tool = FakeTool(
            name="calculator",
            description="Performs calculations",
            parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
        )
        response = FakeResponse(
            id="resp-xyz", model="gpt-4o",
            usage=usage, output=[msg], tools=[tool],
        )

        sd = ResponseSpanData(
            response=response,
            input=[{"role": "user", "content": "What is the meaning of life?"}],
        )
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}

        # Check gen_ai messages
        assert "gen_ai.input.messages" in calls
        assert "gen_ai.output.messages" in calls

        output_msgs = json.loads(calls["gen_ai.output.messages"])
        assert output_msgs[0]["content"] == "The answer is 42."

        # Check tool definitions
        assert "gen_ai.tool.definitions" in calls
        tool_defs = json.loads(calls["gen_ai.tool.definitions"])
        assert len(tool_defs) == 1
        assert tool_defs[0]["type"] == "function"
        assert tool_defs[0]["function"]["name"] == "calculator"

        # Check LLM attributes
        assert calls[Attributes.REQUEST_MODEL.value] == "gpt-4o"
        assert calls[Attributes.INPUT_TOKEN_COUNT.value] == 25
        assert calls[Attributes.OUTPUT_TOKEN_COUNT.value] == 50
        assert calls[Attributes.TOTAL_TOKEN_COUNT.value] == 75
        assert calls[Attributes.RESPONSE_ID.value] == "resp-xyz"

    def test_response_span_with_tool_call_output(self):
        """Test ResponseSpanData when the model makes a function call."""
        from agents.tracing.span_data import ResponseSpanData

        lmnr_span = MagicMock()
        usage = FakeResponseUsage(input_tokens=30, output_tokens=15, total_tokens=45)
        func_call = FakeFunctionCall(
            call_id="call-abc",
            name="get_weather",
            arguments='{"city": "NYC"}',
        )
        response = FakeResponse(
            id="resp-fc1", model="gpt-4o",
            usage=usage, output=[func_call],
        )

        sd = ResponseSpanData(
            response=response,
            input=[{"role": "user", "content": "What's the weather?"}],
        )
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}

        # Check output messages contain tool call
        assert "gen_ai.output.messages" in calls
        output_msgs = json.loads(calls["gen_ai.output.messages"])
        assert len(output_msgs) == 1
        assert output_msgs[0]["role"] == "assistant"
        assert output_msgs[0]["content"] is None
        assert len(output_msgs[0]["tool_calls"]) == 1
        tc = output_msgs[0]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "NYC"}'


class TestHandoffSpanDataIntegration:
    def test_handoff_attributes(self):
        lmnr_span = MagicMock()
        sd = HandoffSpanData(from_agent="Agent1", to_agent="Agent2")
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}
        assert calls["openai.agents.handoff.from"] == "Agent1"
        assert calls["openai.agents.handoff.to"] == "Agent2"


class TestGuardrailSpanDataIntegration:
    def test_guardrail_not_triggered(self):
        lmnr_span = MagicMock()
        sd = GuardrailSpanData(name="content_filter", triggered=False)
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}
        assert calls["openai.agents.guardrail.name"] == "content_filter"
        assert calls["openai.agents.guardrail.triggered"] is False

    def test_guardrail_triggered(self):
        lmnr_span = MagicMock()
        sd = GuardrailSpanData(name="content_filter", triggered=True)
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}
        assert calls["openai.agents.guardrail.triggered"] is True


class TestCustomSpanDataIntegration:
    def test_custom_span(self):
        lmnr_span = MagicMock()
        sd = CustomSpanData(name="my_custom_op", data={"key": "value", "count": 42})
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}
        assert calls["openai.agents.custom.name"] == "my_custom_op"
        custom_data = json.loads(calls["openai.agents.custom.data"])
        assert custom_data == {"key": "value", "count": 42}


class TestMCPSpanDataIntegration:
    def test_mcp_list_tools(self):
        lmnr_span = MagicMock()
        sd = MCPListToolsSpanData(server="my-mcp-server", result="[tool1, tool2]")
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}
        assert calls["openai.agents.mcp.server"] == "my-mcp-server"
        assert "openai.agents.mcp.result" in calls


@pytest.mark.skipif(not HAS_REALTIME, reason="Realtime span types not available")
class TestRealtimeSpanDataIntegration:
    def test_speech_span(self):
        lmnr_span = MagicMock()
        sd = SpeechSpanData()
        sd.model = "gpt-4o-realtime"
        sd.input = "Hello, how are you?"
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}
        assert calls[Attributes.REQUEST_MODEL.value] == "gpt-4o-realtime"

    def test_transcription_span(self):
        lmnr_span = MagicMock()
        sd = TranscriptionSpanData()
        sd.model = "whisper-1"
        sd.output = "This is the transcribed text"
        _apply_span_data(lmnr_span, sd)

        calls = {c[0][0]: c[0][1] for c in lmnr_span.set_attribute.call_args_list}
        assert calls[Attributes.REQUEST_MODEL.value] == "whisper-1"


# ---------------------------------------------------------------------------
# Test error handling with real SDK types
# ---------------------------------------------------------------------------


class TestErrorHandlingIntegration:
    def test_span_error_with_span_error_type(self):
        from agents import SpanError
        lmnr_span = MagicMock()
        error = SpanError(message="API rate limit exceeded", data={"status": 429})
        span = FakeSpan(error=error)
        _apply_span_error(lmnr_span, span)

        lmnr_span.set_status.assert_called_once()
        status = lmnr_span.set_status.call_args[0][0]
        assert status.status_code.name == "ERROR"
        assert "API rate limit exceeded" in status.description


# ---------------------------------------------------------------------------
# Full trace processor lifecycle with real span data types
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """Test the complete trace -> span -> end lifecycle with real SDK types."""

    @patch("lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.Laminar")
    def test_simple_agent_run(self, mock_laminar):
        """Simulate a simple agent run: agent span -> generation span -> response."""
        mock_root = MagicMock()
        mock_root.get_laminar_span_context.return_value = MagicMock()
        mock_agent_span = MagicMock()
        mock_agent_span.get_laminar_span_context.return_value = MagicMock()
        mock_gen_span = MagicMock()

        mock_laminar.start_span.side_effect = [mock_root, mock_agent_span, mock_gen_span]

        processor = LaminarAgentsTraceProcessor()
        trace = FakeTrace(trace_id="t-1", name="Agent Run")
        processor.on_trace_start(trace)

        # Agent span
        agent_sd = AgentSpanData(
            name="Assistant", tools=["search"], output_type="str"
        )
        agent_span = FakeSpan(
            trace_id="t-1", span_id="agent-1", name="Assistant",
            span_data=agent_sd,
        )
        processor.on_span_start(agent_span)

        # Generation span under agent
        gen_sd = GenerationSpanData(
            input=[{"role": "user", "content": "Hello"}],
            output=[{"role": "assistant", "content": "Hi there!"}],
            model="gpt-4o",
            usage={"input_tokens": 5, "output_tokens": 8, "total_tokens": 13},
        )
        gen_span = FakeSpan(
            trace_id="t-1", span_id="gen-1", parent_id="agent-1",
            name="Generation",
            span_data=gen_sd,
        )
        processor.on_span_start(gen_span)

        # End generation
        processor.on_span_end(gen_span)
        mock_gen_span.end.assert_called_once()

        # Verify generation span had correct attributes set
        gen_calls = {c[0][0]: c[0][1] for c in mock_gen_span.set_attribute.call_args_list}
        assert "gen_ai.input.messages" in gen_calls
        assert "gen_ai.output.messages" in gen_calls
        assert gen_calls.get(Attributes.REQUEST_MODEL.value) == "gpt-4o"
        assert gen_calls.get(Attributes.INPUT_TOKEN_COUNT.value) == 5
        assert gen_calls.get(Attributes.OUTPUT_TOKEN_COUNT.value) == 8

        # End agent
        processor.on_span_end(agent_span)
        mock_agent_span.end.assert_called_once()

        # End trace
        processor.on_trace_end(trace)
        mock_root.end.assert_called_once()

    @patch("lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.Laminar")
    def test_agent_with_tool_call(self, mock_laminar):
        """Simulate agent using a tool: agent -> gen (tool call) -> function -> gen (final)."""
        mock_root = MagicMock()
        mock_root.get_laminar_span_context.return_value = MagicMock()
        mock_agent = MagicMock()
        mock_agent.get_laminar_span_context.return_value = MagicMock()
        mock_gen1 = MagicMock()
        mock_func = MagicMock()
        mock_gen2 = MagicMock()

        mock_laminar.start_span.side_effect = [
            mock_root, mock_agent, mock_gen1, mock_func, mock_gen2
        ]

        processor = LaminarAgentsTraceProcessor()
        trace = FakeTrace(trace_id="t-1")
        processor.on_trace_start(trace)

        # Agent span
        agent_sd = AgentSpanData(name="WeatherBot", tools=["get_weather"])
        agent_span = FakeSpan(
            trace_id="t-1", span_id="a-1",
            span_data=agent_sd, name="WeatherBot"
        )
        processor.on_span_start(agent_span)

        # First generation: model calls tool
        gen1_sd = GenerationSpanData(
            input=[{"role": "user", "content": "Weather in NYC?"}],
            output=[{"type": "function_call", "name": "get_weather",
                     "arguments": '{"city": "NYC"}'}],
            model="gpt-4o",
            usage={"input_tokens": 10, "output_tokens": 15, "total_tokens": 25},
        )
        gen1_span = FakeSpan(
            trace_id="t-1", span_id="g-1", parent_id="a-1",
            span_data=gen1_sd
        )
        processor.on_span_start(gen1_span)
        processor.on_span_end(gen1_span)

        # Function execution
        func_sd = FunctionSpanData(
            name="get_weather",
            input='{"city": "NYC"}',
            output="72°F, sunny",
        )
        func_span = FakeSpan(
            trace_id="t-1", span_id="f-1", parent_id="a-1",
            span_data=func_sd
        )
        processor.on_span_start(func_span)
        processor.on_span_end(func_span)

        # Verify function span
        func_calls = {c[0][0]: c[0][1] for c in mock_func.set_attribute.call_args_list}
        assert func_calls["openai.agents.tool.name"] == "get_weather"

        # Second generation: model responds with answer
        gen2_sd = GenerationSpanData(
            input=[
                {"role": "user", "content": "Weather in NYC?"},
                {"type": "function_call", "name": "get_weather",
                 "arguments": '{"city": "NYC"}'},
                {"type": "function_call_output", "output": "72°F, sunny"},
            ],
            output=[{"role": "assistant", "content": "It's 72°F and sunny in NYC!"}],
            model="gpt-4o",
            usage={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
        )
        gen2_span = FakeSpan(
            trace_id="t-1", span_id="g-2", parent_id="a-1",
            span_data=gen2_sd
        )
        processor.on_span_start(gen2_span)
        processor.on_span_end(gen2_span)

        # Verify second generation
        gen2_calls = {c[0][0]: c[0][1] for c in mock_gen2.set_attribute.call_args_list}
        assert gen2_calls.get(Attributes.INPUT_TOKEN_COUNT.value) == 30
        assert gen2_calls.get(Attributes.OUTPUT_TOKEN_COUNT.value) == 12

        # Cleanup
        processor.on_span_end(agent_span)
        processor.on_trace_end(trace)

    @patch("lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.Laminar")
    def test_handoff_between_agents(self, mock_laminar):
        """Test handoff from one agent to another."""
        mock_root = MagicMock()
        mock_root.get_laminar_span_context.return_value = MagicMock()
        mock_agent1 = MagicMock()
        mock_agent1.get_laminar_span_context.return_value = MagicMock()
        mock_handoff = MagicMock()
        mock_agent2 = MagicMock()

        mock_laminar.start_span.side_effect = [
            mock_root, mock_agent1, mock_handoff, mock_agent2
        ]

        processor = LaminarAgentsTraceProcessor()
        trace = FakeTrace(trace_id="t-1")
        processor.on_trace_start(trace)

        # First agent
        a1_sd = AgentSpanData(name="Triage", handoffs=["Specialist"])
        a1_span = FakeSpan(
            trace_id="t-1", span_id="a-1",
            span_data=a1_sd, name="Triage"
        )
        processor.on_span_start(a1_span)

        # Handoff span
        handoff_sd = HandoffSpanData(from_agent="Triage", to_agent="Specialist")
        handoff_span = FakeSpan(
            trace_id="t-1", span_id="h-1", parent_id="a-1",
            span_data=handoff_sd, name="Handoff"
        )
        processor.on_span_start(handoff_span)
        processor.on_span_end(handoff_span)

        # Verify handoff attributes
        h_calls = {c[0][0]: c[0][1] for c in mock_handoff.set_attribute.call_args_list}
        assert h_calls["openai.agents.handoff.from"] == "Triage"
        assert h_calls["openai.agents.handoff.to"] == "Specialist"

        # End first agent
        processor.on_span_end(a1_span)

        # Second agent
        a2_sd = AgentSpanData(name="Specialist")
        a2_span = FakeSpan(
            trace_id="t-1", span_id="a-2",
            span_data=a2_sd, name="Specialist"
        )
        processor.on_span_start(a2_span)
        processor.on_span_end(a2_span)

        processor.on_trace_end(trace)


# ---------------------------------------------------------------------------
# Test OpenAIAgentsInstrumentor
# ---------------------------------------------------------------------------


class TestInstrumentor:
    def test_instrumentation_dependencies(self):
        from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents import (
            OpenAIAgentsInstrumentor,
        )
        instr = OpenAIAgentsInstrumentor()
        deps = instr.instrumentation_dependencies()
        assert "openai-agents >= 0.7.0" in deps

    @patch("agents.tracing.add_trace_processor")
    def test_instrument_registers_processor(self, mock_add):
        from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents import (
            OpenAIAgentsInstrumentor,
        )
        instr = OpenAIAgentsInstrumentor()
        instr._instrument()

        mock_add.assert_called_once()
        processor = mock_add.call_args[0][0]
        assert isinstance(processor, LaminarAgentsTraceProcessor)

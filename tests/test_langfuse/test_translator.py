"""Unit tests for the `langfuse.*` -> Laminar attribute translation (no langfuse SDK needed)."""

from __future__ import annotations

import json
from typing import Any, cast

from lmnr.opentelemetry_lib.bridges.langfuse.translate import is_langfuse_span
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_TYPE,
)

from .utils import FakeSpan, translate


def test_translator_ignores_non_langfuse_spans():
    span = FakeSpan({"some.attr": "x"}, scope_name="openai")
    span.attributes = translate(span)
    assert span.attributes == {"some.attr": "x"}


def test_is_langfuse_span_detects_attrs_without_scope():
    """A span carrying `langfuse.*` attributes is a Langfuse span even when its
    instrumentation scope is missing — detection must fall through to the
    attribute check rather than short-circuiting on a None scope."""
    span = FakeSpan({"langfuse.observation.type": "generation"})
    span.instrumentation_scope = None
    assert is_langfuse_span(cast(Any, span)) is True


def test_translator_maps_generation_to_llm_span():
    span = FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": "gpt-4o",
            "langfuse.observation.usage_details": json.dumps(
                {"input": 10, "output": 20, "total": 30}
            ),
            "langfuse.observation.cost_details": json.dumps(
                {"input": 0.001, "output": 0.002, "total": 0.003}
            ),
            # A non-message input dict (no "messages" key) is not a recognized
            # chat-prompt shape, so it falls back to the raw lmnr.span.input blob.
            "langfuse.observation.input": '{"prompt": "hi"}',
            "langfuse.observation.output": '{"role": "assistant", "content": "hello"}',
        }
    )
    span.attributes = translate(span)

    assert span.attributes[SPAN_TYPE] == "LLM"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert span.attributes["gen_ai.response.model"] == "gpt-4o"
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 20
    assert span.attributes["llm.usage.total_tokens"] == 30
    assert span.attributes["gen_ai.usage.input_cost"] == 0.001
    assert span.attributes["gen_ai.usage.output_cost"] == 0.002
    assert span.attributes["gen_ai.usage.cost"] == 0.003
    assert span.attributes[SPAN_INPUT] == '{"prompt": "hi"}'
    # A dict output is normalized into a one-element gen_ai.output.messages.
    assert json.loads(cast(str, span.attributes["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "hello"}
    ]


def test_translator_empty_tool_calls_output_stays_normal_message():
    """Regression: a plain text completion that carries an empty `tool_calls`
    list must NOT be wrapped into the `{"message": ...}` choices shape.

    `all([])` is True, so the OpenAI-output-choice branch would otherwise fire
    on an empty `tool_calls` and warp the message shape, breaking transcript
    rendering."""
    msg: dict[str, str | list[dict[str, str]]] = {
        "role": "assistant",
        "content": "just text",
        "tool_calls": [],
    }
    span = FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.output": json.dumps(msg),
        }
    )
    span.attributes = translate(span)
    out = json.loads(cast(str, span.attributes["gen_ai.output.messages"]))
    assert len(out) == 1
    # Must stay a normal message dict, not a {"message": {...}} choice wrapper.
    assert "message" not in out[0]
    assert out[0]["role"] == "assistant"
    # Empty tool_calls folds into content (no actual calls to append).
    assert out[0]["content"] == [{"type": "text", "text": "just text"}]


def test_translator_splits_openai_input_messages_and_tools():
    """Langfuse's OpenAI integration ships input as {messages, tools, ...}.

    The translator must split that into gen_ai.input.messages +
    gen_ai.tool.definitions rather than dumping the whole dict into
    lmnr.span.input, or the Laminar frontend can't render the transcript.
    """
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "weather in SF?"},
    ]
    span = FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.input": json.dumps({"messages": messages, "tools": tools}),
            "langfuse.observation.output": json.dumps(
                {"role": "assistant", "content": "It's sunny"}
            ),
        }
    )
    span.attributes = translate(span)

    assert json.loads(cast(str, span.attributes["gen_ai.input.messages"])) == messages
    assert json.loads(cast(str, span.attributes["gen_ai.tool.definitions"])) == tools
    # The whole {messages, tools} dict must NOT leak into lmnr.span.input.
    assert SPAN_INPUT not in span.attributes
    assert json.loads(cast(str, span.attributes["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "It's sunny"}
    ]


def test_translator_handles_bare_message_list_input():
    """The vanilla OpenAI case (no tools) ships a bare message array."""
    messages = [{"role": "user", "content": "hi"}]
    span = FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.input": json.dumps(messages),
        }
    )
    span.attributes = translate(span)
    assert json.loads(cast(str, span.attributes["gen_ai.input.messages"])) == messages
    assert "gen_ai.tool.definitions" not in span.attributes
    assert SPAN_INPUT not in span.attributes


def test_translator_maps_openai_functions_to_tool_definitions():
    """The legacy `functions` key maps to tool definitions too."""
    functions: list[dict[str, str | dict[str, str]]] = [{"name": "get_weather", "parameters": {}}]
    span = FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.input": json.dumps(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "functions": functions,
                }
            ),
        }
    )
    span.attributes = translate(span)
    assert json.loads(cast(str, span.attributes["gen_ai.tool.definitions"])) == functions


def test_translator_splits_langchain_openai_inlined_tool_defs():
    """langfuse.langchain inlines OpenAI tool definitions into the message array
    as role="tool" messages whose content is {type:"function", function:{...}}.
    Those must be pulled out into gen_ai.tool.definitions, leaving only the real
    messages in gen_ai.input.messages."""
    fn = {"name": "get_weather", "parameters": {"type": "object"}}
    span = FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.input": json.dumps(
                [
                    {"role": "user", "content": "weather?"},
                    {"role": "tool", "content": {"type": "function", "function": fn}},
                ]
            ),
        }
    )
    span.attributes = translate(span)
    assert json.loads(cast(str, span.attributes["gen_ai.input.messages"])) == [
        {"role": "user", "content": "weather?"}
    ]
    assert json.loads(cast(str, span.attributes["gen_ai.tool.definitions"])) == [fn]


def test_translator_splits_langchain_anthropic_inlined_tool_defs():
    """Regression: langchain + Anthropic via langfuse.

    Anthropic-langchain inlines tool DEFINITIONS into the input array as
    role="tool" messages whose content is the Anthropic-native shape
    {name, input_schema, description} — there is NO {type:"function"} wrapper.
    The old splitter did `content["type"]` (KeyError), which the on_end
    try/except swallowed, so the whole span was left untranslated (all
    langfuse.* attrs stayed in their original shape). The splitter must
    recognize the Anthropic shape and route it to gen_ai.tool.definitions."""
    tool_def = {
        "name": "transform_1",
        "input_schema": {
            "properties": {
                "number": {"type": "integer"},
                "number2": {"type": "integer"},
            },
            "required": ["number"],
            "type": "object",
        },
        "description": "Runs transformation 1 on the given number.",
    }
    output = {
        "role": "assistant",
        "content": [
            {"text": "I'll apply these transformations.", "type": "text"},
            {
                "id": "toolu_01EF",
                "input": {"number": 2},
                "name": "transform_1",
                "type": "tool_use",
            },
        ],
        "tool_calls": [
            {
                "name": "transform_1",
                "args": {"number": 2},
                "id": "toolu_01EF",
                "type": "tool_call",
            }
        ],
    }
    span = FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": "claude-haiku-4-5-20251001",
            "langfuse.observation.usage_details": json.dumps(
                {
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "input": 862,
                    "output": 75,
                }
            ),
            "langfuse.observation.input": json.dumps(
                [
                    {"role": "user", "content": "Apply transformation 1 to 2."},
                    {"role": "tool", "content": tool_def},
                ]
            ),
            "langfuse.observation.output": json.dumps(output),
        }
    )
    span.attributes = translate(span)

    # The span is recognized as an LLM call and model/tokens are translated.
    assert span.attributes[SPAN_TYPE] == "LLM"
    assert span.attributes["gen_ai.request.model"] == "claude-haiku-4-5-20251001"
    assert span.attributes["gen_ai.usage.input_tokens"] == 862
    assert span.attributes["gen_ai.usage.output_tokens"] == 75
    # The Anthropic tool definition is pulled out of the message array.
    assert json.loads(cast(str, span.attributes["gen_ai.input.messages"])) == [
        {"role": "user", "content": "Apply transformation 1 to 2."}
    ]
    assert json.loads(cast(str, span.attributes["gen_ai.tool.definitions"])) == [tool_def]
    # The output keeps the tool_use content blocks and drops the redundant
    # top-level tool_calls mirror.
    out_messages = json.loads(cast(str, span.attributes["gen_ai.output.messages"]))
    assert len(out_messages) == 1
    assert "tool_calls" not in out_messages[0]
    assert out_messages[0]["content"] == output["content"]


def test_translator_non_llm_input_falls_back_to_span_input():
    """Non-LLM observations keep the raw input/output blob behaviour."""
    span = FakeSpan(
        {
            "langfuse.observation.type": "span",
            "langfuse.observation.input": '{"messages": [{"role": "user"}]}',
            "langfuse.observation.output": '{"role": "assistant"}',
        }
    )
    span.attributes = translate(span)
    assert span.attributes[SPAN_INPUT] == '{"messages": [{"role": "user"}]}'
    assert span.attributes[SPAN_OUTPUT] == '{"role": "assistant"}'
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.output.messages" not in span.attributes


def test_translator_converts_openinference_llm_span():
    """openinference instrumentations (groq / google_genai) emit a flat,
    indexed attribute layout with no langfuse.* keys. The translator must
    recognize and convert them into Laminar / GenAI conventions."""
    span = FakeSpan(
        {
            "openinference.span.kind": "LLM",
            "llm.model_name": "gemini-2.0-flash",
            "llm.token_count.prompt": 12,
            "llm.token_count.completion": 8,
            "llm.token_count.total": 20,
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": "You are helpful",
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.content": "weather in SF?",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_1",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"city": "SF"}',
            "llm.tools.0.tool.json_schema": json.dumps(
                {"type": "function", "function": {"name": "get_weather"}}
            ),
        },
        scope_name="openinference.instrumentation.google_genai",
    )
    span.attributes = translate(span)

    assert span.attributes[SPAN_TYPE] == "LLM"
    assert span.attributes["gen_ai.request.model"] == "gemini-2.0-flash"
    assert span.attributes["gen_ai.usage.input_tokens"] == 12
    assert span.attributes["gen_ai.usage.output_tokens"] == 8
    assert span.attributes["llm.usage.total_tokens"] == 20
    assert json.loads(cast(str, span.attributes["gen_ai.input.messages"])) == [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "weather in SF?"},
    ]
    out = json.loads(cast(str, span.attributes["gen_ai.output.messages"]))
    assert out[0]["role"] == "assistant"
    assert out[0]["content"][0]["type"] == "tool_call"
    assert out[0]["content"][0]["name"] == "get_weather"
    assert out[0]["content"][0]["arguments"] == '{"city": "SF"}'
    assert json.loads(cast(str, span.attributes["gen_ai.tool.definitions"])) == [
        {"type": "function", "function": {"name": "get_weather"}}
    ]


def test_translator_converts_openinference_without_span_kind():
    """openinference spans are still recognized via the llm.* indexed keys
    even when openinference.span.kind is absent."""
    span = FakeSpan(
        {
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "hi",
            "llm.token_count.prompt": 3,
        },
        scope_name="openinference.instrumentation.groq",
    )
    span.attributes = translate(span)
    assert span.attributes[SPAN_TYPE] == "LLM"
    assert json.loads(cast(str, span.attributes["gen_ai.input.messages"])) == [
        {"role": "user", "content": "hi"}
    ]
    assert span.attributes["gen_ai.usage.input_tokens"] == 3


def test_translator_does_not_mistype_tool_forced_llm_call_as_tool():
    """litellm's `langfuse_otel` callback (via arize `_utils.set_attributes`)
    forces `openinference.span.kind=TOOL` on ANY completion that passes
    `tools=[...]`, even though it's a genuine LLM call. Such a span still
    carries LLM signals (model name / token counts), so it must be typed LLM,
    not TOOL — otherwise `litellm_request` spans following a tool observation
    leak the TOOL type (LAM-1784)."""
    span = FakeSpan(
        {
            "openinference.span.kind": "TOOL",
            "llm.model_name": "gpt-4o-mini",
            "llm.token_count.prompt": 12,
            "llm.token_count.completion": 8,
            "llm.tools.0.name": "get_weather",
            "langfuse.observation.input": json.dumps(
                [{"role": "user", "content": "weather in SF?"}]
            ),
            "langfuse.observation.output": json.dumps(
                {"role": "assistant", "content": "It is sunny."}
            ),
        },
        scope_name="litellm",
    )
    span.attributes = translate(span)
    assert span.attributes[SPAN_TYPE] == "LLM"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o-mini"


def test_translator_keeps_genuine_tool_span_as_tool():
    """A true tool-execution span (TOOL kind, no model / tokens / messages)
    must still be typed TOOL — the LLM-signal guard must not over-trigger."""
    span = FakeSpan(
        {
            "openinference.span.kind": "TOOL",
            "input.value": json.dumps({"city": "SF"}),
            "output.value": "sunny",
        },
        scope_name="litellm",
    )
    span.attributes = translate(span)
    assert span.attributes[SPAN_TYPE] == "TOOL"


def test_translator_ignores_plain_non_langfuse_non_oi_spans():
    """A span that is neither langfuse- nor openinference-shaped is untouched."""
    span = FakeSpan({"some.attr": "x"}, scope_name="my.app")
    span.attributes = translate(span)
    assert span.attributes == {"some.attr": "x"}


def test_translator_maps_tool_observation():
    span = FakeSpan({"langfuse.observation.type": "tool"})
    span.attributes = translate(span)
    assert span.attributes[SPAN_TYPE] == "TOOL"


def test_translator_promotes_trace_session_user_tags_metadata():
    span = FakeSpan(
        {
            "session.id": "s456",
            "user.id": "u123",
            "langfuse.trace.tags": ("t1", "t2"),
            "langfuse.trace.metadata.mk": "mv",
            "langfuse.observation.type": "span",
        }
    )
    span.attributes = translate(span)
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.session_id"] == "s456"
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.user_id"] == "u123"
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.tags"] == ["t1", "t2"]
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.metadata.mk"] == "mv"


def test_translator_infers_total_tokens_when_absent():
    span = FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.usage_details": json.dumps({"input": 7, "output": 3}),
        }
    )
    span.attributes = translate(span)
    assert span.attributes["llm.usage.total_tokens"] == 10


def test_translator_handles_dict_usage_without_json_wrap():
    """Langfuse sometimes passes a dict rather than a JSON string."""
    span = FakeSpan(
        {
            "langfuse.observation.type": "generation",
            # Directly a dict (not JSON-encoded) — the translator should still
            # handle it, but OTel's attribute type-check will have flattened it by
            # the time on_end runs. We keep the helper resilient.
            "langfuse.observation.usage_details": {"input": 1, "output": 2},
        }
    )
    span.attributes = translate(span)
    assert span.attributes["gen_ai.usage.input_tokens"] == 1
    assert span.attributes["gen_ai.usage.output_tokens"] == 2


def test_usage_details_are_additive_like_langfuse():
    """Langfuse's `input` / `output` exclude cached / reasoning tokens, which
    come as separate `input_*` / `output_*` keys. Laminar's totals include them."""
    span = FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.usage_details": json.dumps(
                {
                    "input": 9,
                    "input_cache_read": 2,
                    "input_cache_creation": 3,
                    "output": 5,
                    "output_reasoning": 4,
                }
            ),
        }
    )
    attrs = translate(span)
    assert attrs["gen_ai.usage.input_tokens"] == 14
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 2
    assert attrs["gen_ai.usage.cache_write.input_tokens"] == 3
    assert attrs["gen_ai.usage.output_tokens"] == 9
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 4
    assert attrs["llm.usage.total_tokens"] == 23


def test_openai_shaped_usage_is_accepted():
    span = FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.usage_details": json.dumps(
                {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
            ),
        }
    )
    attrs = translate(span)
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 2
    assert attrs["llm.usage.total_tokens"] == 9


def test_json_encoded_metadata_is_decoded():
    """langfuse 5 JSON-encodes every metadata value."""
    span = FakeSpan(
        {
            "langfuse.trace.metadata.team": '"a"',
            "langfuse.trace.metadata.count": "3",
            "langfuse.trace.metadata.nested": '{"x": 1}',
        }
    )
    attrs = translate(span, json_encoded_metadata=True)
    assert attrs[f"{ASSOCIATION_PROPERTIES}.metadata.team"] == "a"
    assert attrs[f"{ASSOCIATION_PROPERTIES}.metadata.count"] == 3
    assert json.loads(cast(str, attrs[f"{ASSOCIATION_PROPERTIES}.metadata.nested"])) == {"x": 1}

    # Older versions send raw strings, which are kept as they are.
    attrs = translate(FakeSpan({"langfuse.trace.metadata.count": "3"}))
    assert attrs[f"{ASSOCIATION_PROPERTIES}.metadata.count"] == "3"


def test_observation_metadata_is_not_promoted():
    span = FakeSpan({"langfuse.observation.metadata.ls_provider": "openai"})
    attrs = translate(span)
    assert not any(k.startswith(f"{ASSOCIATION_PROPERTIES}.metadata") for k in attrs)


def test_model_parameters_map_to_gen_ai_request_attributes():
    span = FakeSpan(
        {
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.parameters": json.dumps(
                {
                    "temperature": 1,
                    "top_p": 0.9,
                    "max_completion_tokens": 100,
                    "max_output_tokens": 50,
                    "frequency_penalty": 0,
                    "seed": 3,
                    "n": 2,
                    "stop": "END",
                    "reasoning_effort": "high",
                    "reasoning_summary": "auto",
                    "service_tier": "flex",
                    "verbosity": "low",
                }
            ),
        }
    )
    attrs = translate(span)
    assert attrs["gen_ai.request.temperature"] == 1
    assert attrs["gen_ai.request.top_p"] == 0.9
    assert attrs["gen_ai.request.max_tokens"] == 100
    assert attrs["gen_ai.request.frequency_penalty"] == 0
    assert attrs["gen_ai.request.seed"] == 3
    assert attrs["gen_ai.request.choice.count"] == 2
    assert attrs["gen_ai.request.stop_sequences"] == ["END"]
    assert attrs["gen_ai.request.reasoning.level"] == "high"
    assert attrs["gen_ai.request.reasoning_summary"] == "auto"
    assert attrs["gen_ai.request.service_tier"] == "flex"
    # Unmapped keys stay only in the raw attribute.
    assert not any("verbosity" in k for k in attrs if k.startswith("gen_ai."))

    # LangChain records an unset `max_tokens` as "Infinity".
    attrs = translate(
        FakeSpan({"langfuse.observation.model.parameters": '{"max_tokens": "Infinity"}'})
    )
    assert "gen_ai.request.max_tokens" not in attrs


def test_provider_comes_from_langchain_ls_provider():
    attrs = translate(
        FakeSpan({"langfuse.observation.metadata.ls_provider": '"openai"'}),
        json_encoded_metadata=True,
    )
    assert attrs["gen_ai.system"] == "openai"
    attrs = translate(FakeSpan({"langfuse.observation.metadata.ls_provider": "amazon_bedrock"}))
    assert attrs["gen_ai.system"] == "bedrock"
    # `langfuse.openai` records no provider.
    assert "gen_ai.system" not in translate(FakeSpan({"langfuse.observation.model.name": "gpt-4o"}))


def test_openinference_provider_and_invocation_parameters():
    span = FakeSpan(
        {
            "openinference.span.kind": "LLM",
            "llm.model_name": "claude-sonnet-4",
            "llm.provider": "anthropic",
            "llm.invocation_parameters": json.dumps(
                {"temperature": 0.5, "max_tokens": 256, "reasoning_effort": "low"}
            ),
        }
    )
    attrs = translate(span)
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.request.temperature"] == 0.5
    assert attrs["gen_ai.request.max_tokens"] == 256
    assert attrs["gen_ai.request.reasoning.level"] == "low"
    # LiteLLM writes "Unknown" when it has no provider.
    attrs = translate(FakeSpan({"openinference.span.kind": "LLM", "llm.provider": "Unknown"}))
    assert "gen_ai.system" not in attrs


def test_converted_attributes_are_dropped_from_laminar_copy():
    attrs = translate(
        FakeSpan(
            {
                "langfuse.observation.type": "generation",
                "langfuse.observation.model.name": "gpt-4o",
                "langfuse.observation.model.parameters": json.dumps(
                    {"temperature": 0.2, "verbosity": "low"}
                ),
                "langfuse.observation.usage_details": json.dumps({"input": 3, "output": 1}),
                "langfuse.observation.cost_details": json.dumps({"total": 0.1}),
                "langfuse.observation.input": json.dumps([{"role": "user", "content": "hi"}]),
                "langfuse.observation.output": json.dumps({"role": "assistant", "content": "yo"}),
                "langfuse.observation.metadata.team": "a",
            }
        )
    )
    for key in (
        "langfuse.observation.model.name",
        "langfuse.observation.usage_details",
        "langfuse.observation.cost_details",
        "langfuse.observation.input",
        "langfuse.observation.output",
    ):
        assert key not in attrs, key
    # Parameters with no request attribute stay.
    assert json.loads(cast(str, attrs["langfuse.observation.model.parameters"])) == {
        "verbosity": "low"
    }
    assert attrs["langfuse.observation.metadata.team"] == "a"

    # Fully mapped parameters are dropped; unmapped-only ones are kept as they are.
    attrs = translate(FakeSpan({"langfuse.observation.model.parameters": '{"temperature": 1}'}))
    assert "langfuse.observation.model.parameters" not in attrs
    attrs = translate(FakeSpan({"langfuse.observation.model.parameters": '{"verbosity": "low"}'}))
    assert attrs["langfuse.observation.model.parameters"] == '{"verbosity": "low"}'

    # A raw input copied to `lmnr.span.input` is dropped too; one that wasn't
    # converted (a non-string on a non-LLM span) stays.
    attrs = translate(FakeSpan({"langfuse.observation.input": '{"q": "hi"}'}))
    assert attrs[SPAN_INPUT] == '{"q": "hi"}'
    assert "langfuse.observation.input" not in attrs
    attrs = translate(FakeSpan({"langfuse.observation.input": ["a", "b"]}))
    assert attrs["langfuse.observation.input"] == ["a", "b"]


def test_converted_openinference_attributes_are_dropped():
    attrs = translate(
        FakeSpan(
            {
                "openinference.span.kind": "LLM",
                "llm.model_name": "gpt-4o",
                "llm.invocation_parameters": '{"temperature": 0.5}',
                "llm.token_count.prompt": 3,
                "llm.input_messages.0.message.role": "user",
                "llm.input_messages.0.message.content": "hi",
                "input.value": '{"messages": [{"role": "user", "content": "hi"}]}',
                "langfuse.observation.input": '[{"role": "user", "content": "hi"}]',
            },
            scope_name="litellm",
        )
    )
    assert attrs["gen_ai.usage.input_tokens"] == 3
    assert not any(
        k.startswith(("llm.input_messages.", "llm.model_name", "llm.token_count", "llm.invocation"))
        for k in attrs
    )
    assert "input.value" not in attrs
    assert "langfuse.observation.input" not in attrs


def test_hybrid_fallback_drops_both_input_and_output_copies():
    """Without indexed messages, `input.value` becomes `lmnr.span.input`, and
    the `langfuse.observation.input` copy of the same payload goes too."""
    attrs = translate(
        FakeSpan(
            {
                "openinference.span.kind": "LLM",
                "llm.model_name": "gpt-4o",
                "input.value": '{"q": "hi"}',
                "langfuse.observation.input": '{"q": "hi"}',
                "output.value": '{"a": "yo"}',
                "langfuse.observation.output": '{"a": "yo"}',
            },
            scope_name="litellm",
        )
    )
    assert attrs[SPAN_INPUT] == '{"q": "hi"}'
    assert attrs[SPAN_OUTPUT] == '{"a": "yo"}'
    for key in (
        "input.value",
        "langfuse.observation.input",
        "output.value",
        "langfuse.observation.output",
    ):
        assert key not in attrs, key

"""Regression tests for OpenAI Agents span typing, attribute shapes, and de-dup.

Pure unit tests - they drive the span-data appliers and the OpenAI chat wrapper
directly, so no network or cassette is involved.
"""

import json
from types import SimpleNamespace

import pytest
from agents.tracing import span_data as agents_span_data
from opentelemetry.context import set_value
from opentelemetry.trace import StatusCode, get_tracer

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    chat_wrapper,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.helpers import (
    DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY,
    map_span_type,
    reset_current_model_name,
    set_current_model_name,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.messages import (
    set_gen_ai_input_messages,
    set_gen_ai_output_messages,
    set_gen_ai_output_messages_from_response,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.span_data import (
    apply_span_data,
)
from lmnr.opentelemetry_lib.tracing.attributes import Attributes
from lmnr.opentelemetry_lib.tracing.context import (
    attach_context,
    detach_context,
    get_current_context,
)


def _attrs_after_applying(span_exporter, span_data):
    span = Laminar.start_active_span("test")
    apply_span_data(span, span_data)
    span.end()
    return dict(span_exporter.get_finished_spans()[-1].attributes or {})


class _Block:
    """Stand-in for a response output block (`model_as_dict` reads __dict__)."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


# Every span type the Agents SDK can emit, and the Laminar type we record it as.
EXPECTED_SPAN_TYPES = {
    "agent": "DEFAULT",
    "custom": "DEFAULT",
    "function": "TOOL",
    "generation": "LLM",
    "guardrail": "DEFAULT",
    "handoff": "TOOL",
    "mcp_tools": "TOOL",
    "response": "LLM",
    "speech": "LLM",
    "speech_group": "LLM",
    "task": "DEFAULT",
    "transcription": "LLM",
    "turn": "DEFAULT",
}


def _sdk_span_type_strings() -> set[str]:
    types = set()
    for name in dir(agents_span_data):
        cls = getattr(agents_span_data, name)
        if (
            not isinstance(cls, type)
            or not name.endswith("SpanData")
            or name == "SpanData"
        ):
            continue
        # A constant property on every subclass, readable without constructing.
        types.add(cls.type.fget(None))
    return types


def test_every_sdk_span_type_is_accounted_for():
    """If this fails after an `openai-agents` bump, add a branch in
    `apply_span_data` rather than relaxing the assertion."""
    assert _sdk_span_type_strings() == set(EXPECTED_SPAN_TYPES)


@pytest.mark.parametrize("kind,expected", sorted(EXPECTED_SPAN_TYPES.items()))
def test_map_span_type(kind, expected):
    assert map_span_type(SimpleNamespace(type=kind)) == expected


def test_task_span_data_records_attributes(span_exporter):
    attrs = _attrs_after_applying(
        span_exporter,
        agents_span_data.TaskSpanData(
            name="my run", usage={"input_tokens": 12, "output_tokens": 3}
        ),
    )

    assert attrs["openai.agents.task.name"] == "my run"
    assert json.loads(attrs["openai.agents.task.usage"]) == {
        "input_tokens": 12,
        "output_tokens": 3,
    }
    assert not [key for key in attrs if key.startswith("gen_ai.usage.")]


def test_turn_span_data_records_attributes(span_exporter):
    attrs = _attrs_after_applying(
        span_exporter,
        agents_span_data.TurnSpanData(
            turn=2, agent_name="WeatherBot", usage={"input_tokens": 7}
        ),
    )

    assert attrs["openai.agents.turn.index"] == 2
    assert attrs["openai.agents.turn.agent_name"] == "WeatherBot"
    assert json.loads(attrs["openai.agents.turn.usage"]) == {"input_tokens": 7}
    assert not [key for key in attrs if key.startswith("gen_ai.usage.")]


def test_turn_span_data_records_turn_zero(span_exporter):
    """Turn 0 is falsy - it must still be recorded."""
    attrs = _attrs_after_applying(
        span_exporter, agents_span_data.TurnSpanData(turn=0, agent_name="Assistant")
    )

    assert attrs["openai.agents.turn.index"] == 0


def test_response_output_messages_is_a_flat_list(span_exporter):
    """`gen_ai.output.messages` is an array for `response` spans, not a dict."""
    response = SimpleNamespace(
        id="resp_123",
        output=[
            _Block(type="message", content="hello"),
            _Block(type="function_call", name="get_weather", arguments="{}"),
        ],
    )

    span = Laminar.start_active_span("test")
    set_gen_ai_output_messages_from_response(span, response)
    span.end()
    attrs = dict(span_exporter.get_finished_spans()[-1].attributes or {})

    messages = json.loads(attrs["gen_ai.output.messages"])
    assert isinstance(messages, list)
    assert [m["type"] for m in messages] == ["message", "function_call"]
    # The response id lives on gen_ai.response.id, not folded into the array.
    assert "resp_123" not in attrs["gen_ai.output.messages"]


def test_response_and_generation_output_shapes_agree(span_exporter):
    from_response = Laminar.start_active_span("from_response")
    set_gen_ai_output_messages_from_response(
        from_response, SimpleNamespace(id="r", output=[_Block(type="message")])
    )
    from_response.end()

    from_generation = Laminar.start_active_span("from_generation")
    set_gen_ai_output_messages(from_generation, [{"type": "message"}])
    from_generation.end()

    shapes = {
        span.name: type(json.loads(span.attributes["gen_ai.output.messages"]))
        for span in span_exporter.get_finished_spans()
    }
    assert shapes == {"from_response": list, "from_generation": list}


def test_response_output_messages_skipped_when_not_a_list(span_exporter):
    span = Laminar.start_active_span("test")
    set_gen_ai_output_messages_from_response(
        span, SimpleNamespace(id="r", output="not a list")
    )
    span.end()
    attrs = dict(span_exporter.get_finished_spans()[-1].attributes or {})

    assert "gen_ai.output.messages" not in attrs


def test_function_span_output_keeps_its_structure(span_exporter):
    """`export()` would stringify this dict into an unparseable Python repr."""
    attrs = _attrs_after_applying(
        span_exporter,
        agents_span_data.FunctionSpanData(
            name="add_numbers",
            input='{"a": 21, "b": 21}',
            output={"type": "text", "text": "42"},
        ),
    )

    assert json.loads(attrs["lmnr.span.output"]) == {"type": "text", "text": "42"}


@pytest.mark.parametrize("falsy", [0, "", False], ids=["zero", "empty-str", "false"])
def test_function_span_records_falsy_output(span_exporter, falsy):
    """`export()` maps any falsy output to None, losing a legitimate result."""
    attrs = _attrs_after_applying(
        span_exporter,
        agents_span_data.FunctionSpanData(name="t", input=None, output=falsy),
    )

    assert json.loads(attrs["lmnr.span.output"]) == falsy


def _input_messages(span_exporter, input_data, system_instructions):
    span = Laminar.start_active_span("test")
    set_gen_ai_input_messages(span, input_data, system_instructions)
    span.end()
    attrs = dict(span_exporter.get_finished_spans()[-1].attributes or {})
    return json.loads(attrs["gen_ai.input.messages"])


def test_system_instructions_prepended_when_absent(span_exporter):
    """The Responses path sends `instructions` separately, so we must add it."""
    messages = _input_messages(
        span_exporter, [{"role": "user", "content": "Say A."}], "Be terse."
    )

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == [{"type": "input_text", "text": "Be terse."}]


@pytest.mark.parametrize(
    "existing_content",
    [
        "Be terse.",
        [{"type": "input_text", "text": "Be terse."}],
        [{"type": "text", "text": "Be terse."}],
    ],
    ids=["str", "input_text-block", "text-block"],
)
def test_system_instructions_not_duplicated(span_exporter, existing_content):
    """The ChatCompletions path already put the system message in span_data.input."""
    messages = _input_messages(
        span_exporter,
        [
            {"role": "system", "content": existing_content},
            {"role": "user", "content": "Say B."},
        ],
        "Be terse.",
    )

    assert [m["role"] for m in messages] == ["system", "user"]


def test_different_system_message_is_still_prepended(span_exporter):
    """Only an identical system message counts as already-present."""
    messages = _input_messages(
        span_exporter,
        [
            {"role": "system", "content": "Some other prompt."},
            {"role": "user", "content": "Say C."},
        ],
        "Be terse.",
    )

    assert [m["role"] for m in messages] == ["system", "system", "user"]


def _call_chat_wrapper(exc):
    """Call the OpenAI chat wrapper with a `wrapped` that raises `exc`.

    Raising is the cheapest way to reach a terminal span without a real
    response payload to parse.
    """
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(kwargs)
        raise exc

    wrapper = chat_wrapper(get_tracer(__name__))
    with pytest.raises(type(exc)):
        wrapper(
            wrapped,
            SimpleNamespace(_client=None),
            (),
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
    return calls


def test_chat_completions_not_traced_inside_an_agents_span(span_exporter):
    """No duplicate LLM span when an agent runs on a ChatCompletions model."""
    token = attach_context(
        set_value(
            DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY,
            True,
            get_current_context(),
        )
    )
    try:
        calls = _call_chat_wrapper(RuntimeError("boom"))
    finally:
        detach_context(token)

    # The call still goes through, it just isn't traced a second time.
    assert len(calls) == 1
    assert span_exporter.get_finished_spans() == ()


def test_chat_completions_traced_outside_an_agents_span(span_exporter):
    """Control for the test above."""
    calls = _call_chat_wrapper(RuntimeError("boom"))

    assert len(calls) == 1
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["openai.chat"]
    assert spans[0].status.status_code == StatusCode.ERROR


def test_output_blocks_carry_authorship(span_exporter):
    """A roleless block renders as the user's message, so tool calls the model
    made get attributed to the user."""
    response = SimpleNamespace(
        id="resp_1",
        output=[
            _Block(type="function_call", name="get_weather", arguments="{}"),
            _Block(type="reasoning", summary=[]),
            _Block(type="function_call_output", output="72"),
            _Block(type="message", role="assistant", content="It is 72."),
        ],
    )

    span = Laminar.start_active_span("test")
    set_gen_ai_output_messages_from_response(span, response)
    span.end()
    attrs = dict(span_exporter.get_finished_spans()[-1].attributes or {})

    messages = json.loads(attrs["gen_ai.output.messages"])
    assert [(m["type"], m["role"]) for m in messages] == [
        ("function_call", "assistant"),
        ("reasoning", "assistant"),
        ("function_call_output", "tool"),
        ("message", "assistant"),
    ]


def test_generation_output_blocks_carry_authorship(span_exporter):
    span = Laminar.start_active_span("test")
    set_gen_ai_output_messages(span, [{"type": "function_call", "name": "t"}])
    span.end()
    attrs = dict(span_exporter.get_finished_spans()[-1].attributes or {})

    assert json.loads(attrs["gen_ai.output.messages"])[0]["role"] == "assistant"


def test_stamping_a_role_does_not_mutate_the_sdk_object(span_exporter):
    block = {"type": "function_call", "name": "t"}

    span = Laminar.start_active_span("test")
    set_gen_ai_output_messages(span, [block])
    span.end()

    assert block == {"type": "function_call", "name": "t"}


def test_failed_response_span_still_records_the_model(span_exporter):
    """A 4xx leaves `response=None`; the model was known before the request."""
    token = set_current_model_name("gpt-4.1-nano")
    try:
        attrs = _attrs_after_applying(
            span_exporter, agents_span_data.ResponseSpanData(response=None, input=[])
        )
    finally:
        reset_current_model_name(token)

    assert attrs[Attributes.REQUEST_MODEL.value] == "gpt-4.1-nano"
    assert attrs[Attributes.RESPONSE_MODEL.value] == "gpt-4.1-nano"
    assert attrs[Attributes.PROVIDER.value] == "openai"


def test_failed_response_span_without_a_known_model(span_exporter):
    attrs = _attrs_after_applying(
        span_exporter, agents_span_data.ResponseSpanData(response=None, input=[])
    )

    assert Attributes.REQUEST_MODEL.value not in attrs

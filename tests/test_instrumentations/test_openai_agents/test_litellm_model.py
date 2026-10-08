"""Agents SDK + `LitellmModel`: one LLM span per model call, not two.

`LitellmModel.get_response` opens a `generation` span and calls
`litellm.acompletion` inside it. Both instrumentations used to report that one
call: the trace processor as `agents.generation` and the litellm instrumentor as
a nested `litellm.completion`, each carrying the full token usage — so every
turn's tokens and cost were counted twice.

`mock_response` keeps these tests offline.
"""

import time

import litellm
import pytest
from agents import Agent, ModelSettings, Runner
from agents.extensions.models.litellm_model import LitellmModel

from lmnr.opentelemetry_lib.tracing.attributes import Attributes

MOCK_KWARGS = {"mock_response": "Paris."}


def _spans(span_exporter):
    time.sleep(0.1)
    return span_exporter.get_finished_spans()


def _llm_spans(spans):
    return [s for s in spans if (s.attributes or {}).get("lmnr.span.type") == "LLM"]


@pytest.mark.asyncio
async def test_litellm_model_emits_a_single_llm_span(
    instrument_openai_agents, span_exporter
):
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model=LitellmModel(model="anthropic/claude-sonnet-5", api_key="test_key"),
        model_settings=ModelSettings(extra_args=MOCK_KWARGS),
    )

    result = await Runner.run(agent, "What is the capital of France?")
    assert "Paris" in result.final_output

    spans = _spans(span_exporter)
    llm_spans = _llm_spans(spans)
    assert [s.name for s in llm_spans] == ["agents.generation"]
    assert not [s for s in spans if s.name == "litellm.completion"]

    generation = llm_spans[0]
    # Provider comes from the litellm model string, not a hardcoded "openai".
    assert generation.attributes[Attributes.PROVIDER.value] == "anthropic"
    assert generation.attributes[Attributes.INPUT_TOKEN_COUNT.value] > 0
    assert generation.attributes[Attributes.OUTPUT_TOKEN_COUNT.value] > 0


@pytest.mark.asyncio
async def test_litellm_model_streaming_emits_a_single_llm_span(
    instrument_openai_agents, span_exporter
):
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model=LitellmModel(model="anthropic/claude-sonnet-5", api_key="test_key"),
        model_settings=ModelSettings(extra_args=MOCK_KWARGS),
    )

    streamed = Runner.run_streamed(agent, "What is the capital of France?")
    async for _ in streamed.stream_events():
        pass
    assert "Paris" in streamed.final_output

    spans = _spans(span_exporter)
    assert [s.name for s in _llm_spans(spans)] == ["agents.generation"]
    assert not [s for s in spans if s.name == "litellm.completion"]


@pytest.mark.asyncio
async def test_litellm_outside_a_generation_span_is_still_instrumented(
    span_exporter,
):
    """The suppression is scoped to generation spans only.

    Tools, handoffs and plain SDK usage may all call litellm legitimately, and
    those calls have no enclosing span that already describes them.
    """
    await litellm.acompletion(
        model="anthropic/claude-sonnet-5",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        api_key="test_key",
        **MOCK_KWARGS,
    )

    assert [s.name for s in _llm_spans(_spans(span_exporter))] == ["litellm.completion"]

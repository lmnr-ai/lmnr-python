"""Reasoning usage survives every LiteLLM wrapper without changing totals."""

import pytest

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.litellm import (
    wrappers,
)


@pytest.mark.parametrize("api", ["completion", "responses"])
@pytest.mark.parametrize(
    "streaming", [False, True], ids=["nonstream", "stream"]
)
@pytest.mark.parametrize(
    "asynchronous", [False, True], ids=["sync", "async"]
)
@pytest.mark.parametrize(
    "details,expected",
    [
        ({"reasoning_tokens": 123}, 123),
        ({"reasoning_tokens": 0}, 0),
        ({"reasoning_tokens": None}, None),
        ({}, None),
        (None, None),
        ("absent", None),
    ],
    ids=["positive", "zero", "null-count", "empty", "null-details", "absent"],
)
@pytest.mark.asyncio
async def test_litellm_reasoning_token_usage(
    span_exporter, api, streaming, asynchronous, details, expected
):
    if api == "completion":
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
            "prompt_tokens_details": {
                "cached_tokens": 200,
                "cache_creation_tokens": 50,
            },
        }
        detail_key = "completion_tokens_details"
        wrapper = wrappers.wrap_completion
    else:
        usage = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "input_tokens_details": {"cached_tokens": 200},
        }
        detail_key = "output_tokens_details"
        wrapper = wrappers.wrap_responses
    if details != "absent":
        usage[detail_key] = details

    response = {
        "id": "test-reasoning-usage",
        "model": "openai/test-model",
        "choices": [],
        "output": [],
        "usage": usage,
    }
    if api == "completion":
        # A later chunk without usage must not erase the usage-only chunk.
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
            response,
            {"choices": [], "usage": None},
        ]
    else:
        chunks = [
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.completed", "response": response},
        ]

    async def async_stream():
        for chunk in chunks:
            yield chunk

    def call(**kwargs):
        return iter(chunks) if streaming else response

    async def async_call(**kwargs):
        return async_stream() if streaming else response

    result = wrapper(
        {"span_name": f"litellm.{api}", "span_type": "LLM"},
        async_call if asynchronous else call,
        None,
        (),
        {"model": "openai/test-model", "stream": streaming},
    )
    if streaming:
        received = (
            [chunk async for chunk in result] if asynchronous else list(result)
        )
        assert received == chunks
    else:
        received = await result if asynchronous else result
        assert received is response

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = spans[0].attributes
    if expected is None:
        assert "gen_ai.usage.reasoning_tokens" not in attributes
    else:
        assert attributes["gen_ai.usage.reasoning_tokens"] == expected
    assert attributes["gen_ai.usage.input_tokens"] == 1000
    assert attributes["gen_ai.usage.output_tokens"] == 500
    assert attributes["llm.usage.total_tokens"] == 1500
    assert attributes["gen_ai.usage.cache_read_input_tokens"] == 200
    if api == "completion":
        assert attributes["gen_ai.usage.cache_creation_input_tokens"] == 50

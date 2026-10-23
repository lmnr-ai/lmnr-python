"""Unit coverage for the OpenAI Responses-API debugger replay surface.

- `OpenAIRolloutWrapper.cached_response_to_responses` — reconstructs an
  `openai.types.responses.Response` from a cached `{type: raw|genAi}` envelope.
- `OpenAIRolloutWrapper.cached_response_to_responses_events` and the
  `(Async)CachedResponsesEventStream` — replay that response for `stream=True`.
  The synthesized events are checked against the openai SDK's own
  `ResponseStreamState`, which is what `responses.stream()` accumulates with.
- `responses_wrappers` input/output message builders — the `gen_ai.input.messages`
  the replay cache hashes is built by the SAME code path on the recording run and
  the replay pre-stamp, so the hashes match by construction.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from openai import omit
from openai.lib.streaming.responses import ResponseStreamState
from openai.types.responses import Response, ResponseStreamEvent

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.rollout import (
    AsyncCachedResponsesEventStream,
    CachedResponsesEventStream,
    OpenAIRolloutWrapper,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.v1 import (
    responses_wrappers as rw,
)
from lmnr.sdk.debug.hash import debug_input_hash


def _genai_envelope() -> dict[str, Any]:
    return {
        "type": "genAi",
        "model": "gpt-4o",
        "messages": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Paris", "annotations": []}],
            }
        ],
    }


def test_cached_response_to_responses_genai():
    wrapper = OpenAIRolloutWrapper()
    response = wrapper.cached_response_to_responses(_genai_envelope())
    assert response is not None
    assert response.model == "gpt-4o"
    assert response.status == "completed"
    assert response.output_text == "Paris"


def test_cached_genai_responses_have_unique_ids():
    # The Responses instrumentation keys its `responses` accumulator by id; a
    # constant id collides across sequential cached HITs and stamps the first
    # call's stale input on every later span. Each reconstruction must be unique.
    wrapper = OpenAIRolloutWrapper()
    a = wrapper.cached_response_to_responses(_genai_envelope())
    b = wrapper.cached_response_to_responses(_genai_envelope())
    assert a is not None and b is not None
    assert a.id != b.id


def test_cached_response_to_responses_raw():
    wrapper = OpenAIRolloutWrapper()
    raw: dict[str, Any] = {
        "id": "resp_123",
        "created_at": 0,
        "model": "gpt-4o",
        "object": "response",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Rome", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
    }
    envelope: dict[str, Any] = {"type": "raw", "response": raw}
    response = wrapper.cached_response_to_responses(envelope)
    assert response is not None
    assert response.id == "resp_123"
    assert response.output_text == "Rome"


def test_cached_response_to_responses_raw_accepts_json_string():
    wrapper = OpenAIRolloutWrapper()
    raw: dict[str, Any] = {
        "id": "resp_str",
        "created_at": 0,
        "model": "gpt-4o",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
    }
    envelope = {"type": "raw", "response": json.dumps(raw)}
    response = wrapper.cached_response_to_responses(envelope)
    assert response is not None
    assert response.id == "resp_str"


def test_cached_response_to_responses_unknown_type():
    wrapper = OpenAIRolloutWrapper()
    assert wrapper.cached_response_to_responses({"type": "bogus"}) is None


def test_cached_response_to_responses_genai_without_messages():
    wrapper = OpenAIRolloutWrapper()
    assert wrapper.cached_response_to_responses({"type": "genAi"}) is None


def test_input_messages_string_input():
    assert rw.build_genai_input_messages("hello") == [{"role": "user", "content": "hello"}]


def test_input_messages_list_input_passthrough():
    items = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": [{"type": "output_text", "text": "Paris"}]},
    ]
    assert rw.build_genai_input_messages(items) == items


def test_input_messages_none_for_unusable_input():
    assert rw.build_genai_input_messages(None) is None
    assert rw.build_genai_input_messages(42) is None


def test_recording_and_replay_input_hash_match():
    # The recording run stamps gen_ai.input.messages via
    # build_genai_input_messages(process_input(kwargs["input"])); the replay
    # pre-stamp uses the identical chain. The hashes must be equal so a HIT is
    # addressable.
    kwargs_input = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "user", "content": "And Italy?"},
    ]
    recorded = rw.build_genai_input_messages(rw.process_input(cast(Any, kwargs_input)))
    prestamp = rw.build_genai_input_messages(rw.process_input(cast(Any, kwargs_input)))
    assert recorded == prestamp
    assert debug_input_hash(recorded) == debug_input_hash(prestamp)


# --- stream=True replay ---------------------------------------------------------


def _response(output: list[dict[str, Any]], status: str = "completed") -> Response:
    return Response.model_validate(
        {
            "id": "resp_1",
            "created_at": 0,
            "model": "gpt-4o",
            "object": "response",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": status,
        }
    )


def _events(response: Response) -> list[ResponseStreamEvent]:
    events = OpenAIRolloutWrapper().cached_response_to_responses_events(response)
    assert events is not None
    return events


def _accumulate(events: list[ResponseStreamEvent]) -> Response:
    """Feed events through the openai SDK's `responses.stream()` accumulator,
    which raises on any event it doesn't expect at that point."""
    state = ResponseStreamState(input_tools=omit, text_format=omit)
    for event in events:
        _ = state.handle_event(event)
    final = state._completed_response  # pyright: ignore[reportPrivateUsage]
    assert final is not None
    return final


_MESSAGE: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Paris", "annotations": []}],
}
_FUNCTION_CALL: dict[str, Any] = {
    "id": "fc_1",
    "type": "function_call",
    "call_id": "call_1",
    "name": "get_weather",
    "arguments": '{"location":"London"}',
    "status": "completed",
}
_REFUSAL: dict[str, Any] = {
    "id": "msg_2",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "refusal", "refusal": "I can't help with that."}],
}


def test_cached_events_text_message():
    events = _events(_response([_MESSAGE]))
    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [event.sequence_number for event in events] == list(range(len(events)))
    final = _accumulate(events)
    assert final.output_text == "Paris"
    assert final.id == "resp_1"


def test_cached_events_function_call_and_refusal():
    response = _response([_FUNCTION_CALL, _REFUSAL])
    events = _events(response)
    types = [event.type for event in events]
    assert "response.function_call_arguments.delta" in types
    assert "response.function_call_arguments.done" in types
    assert "response.refusal.delta" in types
    assert "response.refusal.done" in types
    final = _accumulate(events)
    function_call, refusal = final.output
    assert function_call.type == "function_call"
    assert function_call.model_dump(include={"id", "call_id", "name", "arguments"}) == {
        key: _FUNCTION_CALL[key] for key in ("id", "call_id", "name", "arguments")
    }
    assert refusal.type == "message"
    assert refusal.content[0].model_dump() == _REFUSAL["content"][0]


@pytest.mark.parametrize(
    ("status", "terminal"),
    [
        ("incomplete", "response.incomplete"),
        ("failed", "response.failed"),
    ],
)
def test_cached_events_terminal_status(status: str, terminal: str):
    events = _events(_response([_MESSAGE], status=status))
    assert events[-1].type == terminal


def test_cached_events_cancelled_has_no_terminal_event():
    # The API has no `response.cancelled` event; a cancelled stream just ends.
    events = _events(_response([_MESSAGE], status="cancelled"))
    assert events[-1].type == "response.output_item.done"
    assert not any(
        event.type in ("response.completed", "response.incomplete", "response.failed")
        for event in events
    )


def test_cached_events_stream_sync():
    events = _events(_response([_MESSAGE]))
    with CachedResponsesEventStream(events) as stream:
        assert stream.response.status_code == 200
        assert list(stream) == events

    stream = CachedResponsesEventStream(events)
    assert next(stream) == events[0]
    stream.close()
    assert list(stream) == []


@pytest.mark.asyncio
async def test_cached_events_stream_async():
    events = _events(_response([_MESSAGE]))
    async with AsyncCachedResponsesEventStream(events) as stream:
        assert [event async for event in stream] == events

    stream = AsyncCachedResponsesEventStream(events)
    assert await stream.__anext__() == events[0]
    await stream.close()
    assert [event async for event in stream] == []

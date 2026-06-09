"""Unit coverage for the OpenAI Responses-API debugger replay surface.

Two pieces, both non-streaming (streaming is out of scope for the v2 cache):

- `OpenAIRolloutWrapper.cached_response_to_responses` — reconstructs an
  `openai.types.responses.Response` from a cached `{type: raw|genAi}` envelope.
- `responses_wrappers` input/output message builders — the `gen_ai.input.messages`
  the replay cache hashes is built by the SAME code path on the recording run and
  the replay pre-stamp, so the hashes match by construction.
"""

import json

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.rollout import (
    OpenAIRolloutWrapper,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.v1 import (
    responses_wrappers as rw,
)
from lmnr.sdk.debug.hash import debug_input_hash


def _genai_envelope():
    return {
        "type": "genAi",
        "model": "gpt-4o",
        "messages": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Paris", "annotations": []}
                ],
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


def test_cached_response_to_responses_raw():
    wrapper = OpenAIRolloutWrapper()
    raw = {
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
                "content": [
                    {"type": "output_text", "text": "Rome", "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
    }
    envelope = {"type": "raw", "response": raw}
    response = wrapper.cached_response_to_responses(envelope)
    assert response is not None
    assert response.id == "resp_123"
    assert response.output_text == "Rome"


def test_cached_response_to_responses_raw_accepts_json_string():
    wrapper = OpenAIRolloutWrapper()
    raw = {
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
    assert rw.build_genai_input_messages("hello") == [
        {"role": "user", "content": "hello"}
    ]


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
    recorded = rw.build_genai_input_messages(rw.process_input(kwargs_input))
    prestamp = rw.build_genai_input_messages(rw.process_input(kwargs_input))
    assert recorded == prestamp
    assert debug_input_hash(recorded) == debug_input_hash(prestamp)

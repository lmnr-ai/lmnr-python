"""Serialization of google-genai models into span attributes.

`types.Blob.model_config` sets `ser_json_bytes="base64"`, which is pydantic's
URL-SAFE alphabet (`-`/`_`). Consumers feed the value to a standard base64
decoder, and `base64.b64decode` defaults to `validate=False`, so it silently
DROPS the out-of-alphabet characters instead of raising: every following byte
shifts and the image decodes to garbage. So these attributes must carry
standard base64 (`+`/`/`).

Assertions here go through `json_dumps` — the serialized attribute is the
contract, and the intermediate dicts intentionally carry raw `bytes`.
"""

import base64
import json
from unittest.mock import patch

from google.genai import types

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai import (
    _set_raw_response_attribute,
    _set_request_attributes,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.utils import (
    content_union_to_dict,
    part_to_dict,
)
from lmnr.sdk.utils import json_dumps

# 0xFB 0xEF 0xBE 0xFF 0xFF 0xFF -> "++++////" standard, "----____" URL-safe.
# Every 6-bit group lands on index 62 or 63, the only two indices where the
# two alphabets differ.
ALPHABET_DIVERGENT_BYTES = bytes([0xFB, 0xEF, 0xBE, 0xFF, 0xFF, 0xFF])
STANDARD_B64 = base64.b64encode(ALPHABET_DIVERGENT_BYTES).decode("utf-8")
URLSAFE_B64 = base64.urlsafe_b64encode(ALPHABET_DIVERGENT_BYTES).decode("utf-8")


class _RecordingSpan:
    """Minimal span double — these writers only set attributes."""

    def __init__(self):
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def is_recording(self):
        return True


def _image_part() -> types.Part:
    return types.Part(
        inline_data=types.Blob(mime_type="image/png", data=ALPHABET_DIVERGENT_BYTES)
    )


def _image_response() -> types.GenerateContentResponse:
    return types.GenerateContentResponse.model_validate(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": STANDARD_B64,
                                }
                            }
                        ],
                    },
                    "finish_reason": "STOP",
                }
            ]
        }
    )


def _roundtrip(value) -> dict:
    """Serialize the way the instrumentation does, then read it back."""
    return json.loads(json_dumps(value))


def test_alphabets_actually_diverge_for_this_fixture():
    # Guards the fixture itself: a payload whose base64 happens to avoid
    # indices 62/63 would make every test below pass vacuously.
    assert STANDARD_B64 == "++++////"
    assert URLSAFE_B64 == "----____"


def test_urlsafe_data_silently_corrupts_under_a_standard_decoder():
    # Documents WHY this matters: b64decode defaults to validate=False, so the
    # URL-safe form neither raises nor round-trips — it returns short, shifted
    # bytes. This is what made the corruption present as a valid-looking PNG
    # header with garbage pixels.
    corrupted = base64.b64decode(URLSAFE_B64)

    assert corrupted != ALPHABET_DIVERGENT_BYTES
    assert len(corrupted) < len(ALPHABET_DIVERGENT_BYTES)


def test_content_inline_data_serializes_to_standard_base64():
    content = types.Content(role="user", parts=[_image_part()])

    result = _roundtrip(content_union_to_dict(content))

    data = result["parts"][0]["inline_data"]["data"]
    assert data == STANDARD_B64
    assert base64.b64decode(data) == ALPHABET_DIVERGENT_BYTES


def test_part_inline_data_serializes_to_standard_base64():
    result = _roundtrip(part_to_dict(_image_part()))

    assert result["inline_data"]["data"] == STANDARD_B64


def test_thought_signature_serializes_to_standard_base64():
    # `Part.thought_signature` is the other bytes field on a Part.
    part = types.Part(text="hi", thought_signature=ALPHABET_DIVERGENT_BYTES)

    assert _roundtrip(part_to_dict(part))["thought_signature"] == STANDARD_B64


def test_raw_dict_part_carrying_bytes_serializes_to_standard_base64():
    # A `PartDict` may carry raw bytes directly, bypassing pydantic entirely.
    result = _roundtrip(
        content_union_to_dict(
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": ALPHABET_DIVERGENT_BYTES,
                        }
                    }
                ],
            }
        )
    )

    assert result["parts"][0]["inline_data"]["data"] == STANDARD_B64


def test_blob_object_nested_inside_a_dict_part_serializes_to_standard_base64():
    # `PartDict` also permits a `Blob` MODEL as the value, which reaches the
    # dict branch of `part_to_dict` un-dumped.
    result = _roundtrip(
        content_union_to_dict(
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": types.Blob(
                            mime_type="image/png", data=ALPHABET_DIVERGENT_BYTES
                        )
                    }
                ],
            }
        )
    )

    assert result["parts"][0]["inline_data"]["data"] == STANDARD_B64


def test_dict_and_str_content_paths_are_unaffected():
    # Non-Content inputs never carried bytes through pydantic; pin them so the
    # fix doesn't change their shape.
    assert content_union_to_dict("hello") == {
        "role": "user",
        "parts": [{"text": "hello"}],
    }
    assert content_union_to_dict({"parts": [{"text": "hi"}], "role": "model"}) == {
        "role": "model",
        "parts": [{"text": "hi"}],
    }


def test_non_bytes_fields_keep_their_json_shape():
    # `mode="python"` must not regress the enum / nested-arg coercion that
    # `mode="json"` was providing.
    part = _roundtrip(
        part_to_dict(
            types.Part(
                executable_code=types.ExecutableCode(
                    code="print(1)", language="PYTHON"
                ),
            )
        )
    )
    assert part["executable_code"] == {"code": "print(1)", "language": "PYTHON"}

    part = _roundtrip(
        part_to_dict(
            types.Part(
                function_call=types.FunctionCall(
                    name="f", args={"a": 1, "b": [1, 2.5, None, True]}
                )
            )
        )
    )
    assert part["function_call"]["args"] == {"a": 1, "b": [1, 2.5, None, True]}


def test_a_non_string_dict_key_does_not_discard_the_whole_payload():
    # orjson aborts the ENTIRE document on an unencodable mapping key, which
    # would drop every sibling part. `function_response.response` is typed
    # `dict[str, Any]`, so nested keys are unvalidated and can be tuples.
    content = types.Content(
        role="user",
        parts=[
            types.Part(text="describe this"),
            types.Part(
                function_response=types.FunctionResponse(
                    name="lookup", response={"grid": {(0, 1): "hit"}}
                )
            ),
        ],
    )

    result = _roundtrip(content_union_to_dict(content))

    assert result["parts"][0] == {"text": "describe this"}
    assert result["parts"][1]["function_response"]["name"] == "lookup"


def test_input_messages_attribute_decodes_back_to_the_original_bytes():
    # End-to-end through the real attribute writer.
    span = _RecordingSpan()

    _set_request_attributes(
        span,
        (),
        {
            "model": "gemini-2.5-flash",
            "contents": [types.Content(role="user", parts=[_image_part()])],
        },
    )

    messages = json.loads(span.attributes["gen_ai.input.messages"])
    data = messages[0]["parts"][0]["inline_data"]["data"]
    assert data == STANDARD_B64
    assert base64.b64decode(data) == ALPHABET_DIVERGENT_BYTES


def test_output_messages_attribute_decodes_back_to_the_original_bytes():
    # Image-generating models return inline_data on the RESPONSE, so the output
    # path has the same hazard as the input path.
    span = _RecordingSpan()

    _set_raw_response_attribute(span, _image_response())

    candidates = json.loads(span.attributes["gen_ai.output.messages"])
    data = candidates[0]["content"]["parts"][0]["inline_data"]["data"]
    assert data == STANDARD_B64
    assert base64.b64decode(data) == ALPHABET_DIVERGENT_BYTES


def test_raw_response_attribute_round_trips_through_the_replay_parser():
    # `lmnr.sdk.raw.response` is re-parsed by the debug-replay wrapper via
    # `GenerateContentResponse.model_validate_json`, so it must survive the
    # round trip byte-for-byte.
    span = _RecordingSpan()

    _set_raw_response_attribute(span, _image_response(), record_raw_response=True)

    raw = span.attributes["lmnr.sdk.raw.response"]
    reparsed = types.GenerateContentResponse.model_validate_json(raw)
    assert (
        reparsed.candidates[0].content.parts[0].inline_data.data
        == ALPHABET_DIVERGENT_BYTES
    )


def test_unserializable_raw_response_is_skipped_not_stamped_empty():
    # `json_dumps` swallows failures and returns "{}". The replay cache PREFERS
    # a non-empty raw response over reconstructing from gen_ai.output.messages,
    # so stamping "{}" would shadow the usable fallback.
    span = _RecordingSpan()

    _set_raw_response_attribute(span, _image_response(), record_raw_response=True)
    assert "lmnr.sdk.raw.response" in span.attributes

    empty = _RecordingSpan()
    with patch(
        "lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.json_dumps",
        return_value="{}",
    ):
        _set_raw_response_attribute(empty, _image_response(), record_raw_response=True)

    assert "lmnr.sdk.raw.response" not in empty.attributes


def test_an_unserializable_tool_value_does_not_drop_the_conversation():
    """One exotic value must degrade to itself, not take the attribute with it.

    `function_call.args` / `function_response.response` are typed
    `dict[str, Any]`, so they can hold arbitrary Python objects. `mode="json"`
    raises `PydanticSerializationError` on those, and since the callers are
    `@dont_throw` that dropped the ENTIRE `gen_ai.input.messages` attribute —
    every message in the conversation. `mode="python"` hands the value to
    `json_dumps`, which degrades only that leaf.
    """
    import collections.abc

    class ReadOnlyMapping(collections.abc.Mapping):
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    span = _RecordingSpan()

    _set_request_attributes(
        span,
        (),
        {
            "model": "gemini-2.5-flash",
            "contents": [
                types.Content(role="user", parts=[types.Part(text="weather?")]),
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name="get_weather",
                                response={"raw": ReadOnlyMapping({"temp": 20})},
                            )
                        )
                    ],
                ),
            ],
        },
    )

    messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert len(messages) == 2
    assert messages[0]["parts"][0] == {"text": "weather?"}
    # The exotic value survives structurally rather than as a Python repr.
    assert messages[1]["parts"][0]["function_response"]["response"] == {
        "raw": {"temp": 20}
    }

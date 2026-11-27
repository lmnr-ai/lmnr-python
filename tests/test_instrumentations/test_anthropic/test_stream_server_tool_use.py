import json
from types import SimpleNamespace as NS
from typing import Any, cast

import pytest
from opentelemetry.sdk.trace import TracerProvider

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.anthropic.streaming import (
    build_from_streaming_response,
)


def _events(block_type: str):
    usage = {"input_tokens": 3, "output_tokens": 0}
    return [
        NS(type="message_start", message=NS(model="claude-x", usage=usage, id="msg_1")),
        NS(
            type="content_block_start",
            index=0,
            content_block=NS(type=block_type, id="srvtoolu_1", name="web_search", input={}),
        ),
        NS(
            type="content_block_delta",
            index=0,
            delta=NS(type="input_json_delta", partial_json='{"query": "lmnr"}'),
        ),
        NS(type="content_block_stop", index=0),
        NS(type="message_delta", delta=NS(stop_reason="end_turn"), usage={"output_tokens": 5}),
        NS(type="message_stop"),
    ]


@pytest.mark.parametrize("block_type", ["server_tool_use", "mcp_tool_use"])
def test_stream_server_tool_use_is_recorded(
    block_type: str,
):
    provider = TracerProvider()
    tracer = provider.get_tracer("t")
    span = tracer.start_span("anthropic.chat")
    events = _events(block_type)

    gen = build_from_streaming_response(span, cast("Any", iter(events)), None, {})
    assert gen is not None
    assert list(cast("Any", gen)) == events

    # recorded on the span object itself
    out = json.loads(cast("Any", span).attributes["gen_ai.output.messages"])
    assert out[0]["content"] == [
        {
            "type": block_type,
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "lmnr"},
        }
    ]

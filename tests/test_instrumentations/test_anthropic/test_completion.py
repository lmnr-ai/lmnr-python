import json
import time
from typing import Any, cast

import pytest
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.vcr
def test_anthropic_completion_legacy(
    instrumentor: Any,
    anthropic_client: Anthropic,
    span_exporter: InMemorySpanExporter,
):
    _res = anthropic_client.completions.create(
        prompt=f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}",
        model="claude-instant-1.2",
        max_tokens_to_sample=2048,
        top_p=0.1,
    )
    try:
        anthropic_client.completions.create(  # pyright: ignore[reportCallIssue] intentional
            unknown_parameter="unknown",
        )
    except Exception:
        print("expected exception")

    time.sleep(0.1)

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.completion" for span in spans)

    anthropic_span = spans[0]

    # Verify input messages in new format
    input_messages = json.loads(cast(str, (anthropic_span.attributes or {})["gen_ai.input.messages"]))
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert (
        input_messages[0]["content"]
        == f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}"
    )

    # Verify output messages in new format
    output_messages = json.loads(cast(str, (anthropic_span.attributes or {})["gen_ai.output.messages"]))
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert len(output_messages[0]["content"]) >= 1
    assert output_messages[0]["content"][0]["type"] == "text"

    assert (
        (anthropic_span.attributes or {}).get("gen_ai.response.id")
        == "compl_01EjfrPvPEsRDRUKD6VoBxtK"
    )

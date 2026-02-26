import json
import pytest
import time
from anthropic import AI_PROMPT, HUMAN_PROMPT


@pytest.mark.vcr
def test_anthropic_completion_legacy(
    instrument_legacy, anthropic_client, span_exporter
):
    anthropic_client.completions.create(
        prompt=f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}",
        model="claude-instant-1.2",
        max_tokens_to_sample=2048,
        top_p=0.1,
    )
    try:
        anthropic_client.completions.create(
            unknown_parameter="unknown",
        )
    except Exception:
        pass

    time.sleep(0.1)

    spans = span_exporter.get_finished_spans()
    assert all(span.name == "anthropic.completion" for span in spans)

    anthropic_span = spans[0]

    # Verify input messages in new format
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert (
        input_messages[0]["content"]
        == f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}"
    )

    # Verify output messages in new format
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert len(output_messages[0]["content"]) >= 1
    assert output_messages[0]["content"][0]["type"] == "text"

    assert (
        anthropic_span.attributes.get("gen_ai.response.id")
        == "compl_01EjfrPvPEsRDRUKD6VoBxtK"
    )

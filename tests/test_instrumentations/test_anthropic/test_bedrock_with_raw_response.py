import json
import os
import pytest

try:
    from anthropic import AsyncAnthropicBedrock
except ImportError:
    AsyncAnthropicBedrock = None


@pytest.fixture
def async_anthropic_bedrock_client(instrument_legacy):
    if AsyncAnthropicBedrock is None:
        pytest.skip("AsyncAnthropicBedrock not available")

    # Try to get credentials from environment first
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "test-key")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "test-secret")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")

    return AsyncAnthropicBedrock(
        aws_region=aws_region,
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
    )


# @pytest.mark.skip
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_async_anthropic_bedrock_with_raw_response(
    instrument_legacy,
    async_anthropic_bedrock_client,
    span_exporter,
):
    """Test that AsyncAnthropicBedrock with_raw_response.create generates spans"""
    response = await async_anthropic_bedrock_client.messages.with_raw_response.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "anthropic.chat" for span in spans)

    anthropic_span = spans[0]

    # Verify input messages in new format
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["content"] == "Tell me a joke about OpenTelemetry"

    # Verify output messages in new format
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    # For raw response, content is accessed differently
    response_content = (
        response.parse().content[0].text
        if hasattr(response, "parse")
        else response.content[0].text
    )
    assert any(
        block["type"] == "text" and block["text"] == response_content
        for block in output_messages[0]["content"]
    )

    assert anthropic_span.attributes["gen_ai.usage.input_tokens"] > 0
    assert anthropic_span.attributes["gen_ai.usage.output_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_async_anthropic_bedrock_regular_create(
    instrument_legacy,
    async_anthropic_bedrock_client,
    span_exporter,
):
    """Test that regular AsyncAnthropicBedrock create works (for comparison)"""
    response = await async_anthropic_bedrock_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="anthropic.claude-3-haiku-20240307-v1:0",
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "anthropic.chat" for span in spans)

    anthropic_span = spans[0]

    # Verify input messages in new format
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["content"] == "Tell me a joke about OpenTelemetry"

    # Verify output messages in new format
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert any(
        block["type"] == "text" and block["text"] == response.content[0].text
        for block in output_messages[0]["content"]
    )

    assert anthropic_span.attributes["gen_ai.usage.input_tokens"] > 0
    assert anthropic_span.attributes["gen_ai.usage.output_tokens"] > 0


# @pytest.mark.skip
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_async_anthropic_bedrock_beta_with_raw_response(
    instrument_legacy,
    async_anthropic_bedrock_client,
    span_exporter,
):
    """Test that AsyncAnthropicBedrock beta.messages.with_raw_response.create generates spans"""
    response = (
        await async_anthropic_bedrock_client.beta.messages.with_raw_response.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Tell me a joke about OpenTelemetry",
                }
            ],
            model="anthropic.claude-3-haiku-20240307-v1:0",
        )
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert all(span.name == "anthropic.chat" for span in spans)

    anthropic_span = spans[0]

    # Verify input messages in new format
    input_messages = json.loads(anthropic_span.attributes["gen_ai.input.messages"])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["content"] == "Tell me a joke about OpenTelemetry"

    # Verify output messages in new format
    output_messages = json.loads(anthropic_span.attributes["gen_ai.output.messages"])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    # For raw response, content is accessed differently
    response_content = (
        response.parse().content[0].text
        if hasattr(response, "parse")
        else response.content[0].text
    )
    assert any(
        block["type"] == "text" and block["text"] == response_content
        for block in output_messages[0]["content"]
    )

    assert anthropic_span.attributes["gen_ai.usage.input_tokens"] > 0
    assert anthropic_span.attributes["gen_ai.usage.output_tokens"] > 0

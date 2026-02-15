import asyncio
from unittest.mock import patch

import httpx
import pytest
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.utils import (
    is_reasoning_supported,
)

from .utils import assert_request_contains_tracecontext, spy_decorator


@pytest.mark.vcr
def test_chat(instrument_legacy, span_exporter, log_exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        service_tier="default",
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-CdGqqj1iK4R9EgoAo2k2ZvzxgmgGt"
    )
    assert open_ai_span.attributes.get("openai.request.service_tier") == "default"
    assert open_ai_span.attributes.get("openai.response.service_tier") == "default"

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_tool_calls(instrument_legacy, span_exporter, log_exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "San Francisco"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "1",
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[
            f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.arguments"
        ]
        == '{"location": "San Francisco"}'
    )

    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_call_id"] == "1"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9gKNZbUWSC4s2Uh2QfVV7PYiqWIuH"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_pydantic_based_tool_calls(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    ChatCompletionMessageFunctionToolCall(
                        id="1",
                        type="function",
                        function={
                            "name": "get_current_weather",
                            "arguments": '{"location": "San Francisco"}',
                        },
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "1",
                "content": "The weather in San Francisco is 70 degrees and sunny.",
            },
        ],
    )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]

    assert f"{SpanAttributes.LLM_PROMPTS}.0.content" not in open_ai_span.attributes
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.name"]
        == "get_current_weather"
    )
    assert (
        open_ai_span.attributes[
            f"{SpanAttributes.LLM_PROMPTS}.0.tool_calls.0.arguments"
        ]
        == '{"location": "San Francisco"}'
    )

    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == "The weather in San Francisco is 70 degrees and sunny."
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_call_id"] == "1"
    )
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-9lvGJKrBUPeJjHi3KKSEbGfcfomOP"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_streaming(instrument_legacy, span_exporter, log_exporter, openai_client):
    response = openai_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
        service_tier="default",
    )

    chunk_count = 0
    for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-CdGr0EeaCKMNoLQ4cH79NjnMpgckv"
    )
    assert open_ai_span.attributes.get("openai.request.service_tier") == "default"
    assert open_ai_span.attributes.get("openai.response.service_tier") == "default"

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_streaming(
    instrument_legacy, span_exporter, log_exporter, async_openai_client
):
    response = await async_openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    chunk_count = 0
    async for _ in response:
        chunk_count += 1

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    events = open_ai_span.events
    assert len(events) == chunk_count

    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-CdGt5qCx5Rzql1NaxAplDRwFojACg"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_with_asyncio_run(
    instrument_legacy, span_exporter, log_exporter, async_openai_client
):
    asyncio.run(
        async_openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chatcmpl-CdGt66e4DLUiaHScvU4EpKsSU0sCu"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_context_propagation(
    instrument_legacy, span_exporter, log_exporter, vllm_openai_client
):
    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        vllm_openai_client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chat-43f4347c3299481e9704ab77439fbdb8"
    )
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_async_context_propagation(
    instrument_legacy, span_exporter, log_exporter, async_vllm_openai_client
):
    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        await async_vllm_openai_client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes.get("gen_ai.response.id")
        == "chat-4db07f02ecae49cbafe1d359db1650df"
    )
    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, open_ai_span)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_chat_history_message_dict(instrument_legacy, span_exporter, openai_client):
    first_user_message = {
        "role": "user",
        "content": "Generate a random noun in Korean. Respond with just that word.",
    }
    second_user_message = {
        "role": "user",
        "content": "Now, generate a sentence using the word you just gave me.",
    }
    first_response = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[first_user_message],
    )

    second_response = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            first_user_message,
            {
                "role": "assistant",
                "content": first_response.choices[0].message.content,
            },
            second_user_message,
        ],
    )

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 2
    first_span = spans[0]
    assert first_span.name == "openai.chat"
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == first_user_message["content"]
    )
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
        == first_user_message["role"]
    )
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == first_response.choices[0].message.content
    )
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    )

    second_span = spans[1]
    assert second_span.name == "openai.chat"
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == first_user_message["content"]
    )
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == second_response.choices[0].message.content
    )
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == first_response.choices[0].message.content
    )
    assert second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "assistant"
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"]
        == second_user_message["content"]
    )
    assert second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "user"


@pytest.mark.vcr
def test_chat_history_message_pydantic(instrument_legacy, span_exporter, openai_client):
    first_user_message = {
        "role": "user",
        "content": "Generate a random noun in Korean. Respond with just that word.",
    }
    second_user_message = {
        "role": "user",
        "content": "Now, generate a sentence using the word you just gave me.",
    }
    first_response = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[first_user_message],
    )

    second_response = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            first_user_message,
            first_response.choices[0].message,
            second_user_message,
        ],
    )

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 2
    first_span = spans[0]
    assert first_span.name == "openai.chat"
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == first_user_message["content"]
    )
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
        == first_user_message["role"]
    )
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == first_response.choices[0].message.content
    )
    assert (
        first_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    )

    second_span = spans[1]
    assert second_span.name == "openai.chat"
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == first_user_message["content"]
    )
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == second_response.choices[0].message.content
    )
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
        == first_response.choices[0].message.content
    )
    assert second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "assistant"
    assert (
        second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"]
        == second_user_message["content"]
    )
    assert second_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "user"


@pytest.mark.vcr
@pytest.mark.skipif(
    not is_reasoning_supported(),
    reason="Reasoning is not supported in older OpenAI library versions",
)
def test_chat_reasoning(instrument_legacy, span_exporter, log_exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Count r's in strawberry"}],
        reasoning_effort="low",
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[-1]
    assert span.attributes["gen_ai.request.reasoning_effort"] == "low"
    assert span.attributes["gen_ai.usage.reasoning_tokens"] > 0


def test_chat_exception(instrument_legacy, span_exporter, openai_client):
    openai_client.api_key = "invalid"
    with pytest.raises(Exception):
        openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert open_ai_span.status.status_code == StatusCode.ERROR
    assert open_ai_span.status.description.startswith("Error code: 401")
    events = open_ai_span.events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "openai.AuthenticationError"
    assert event.attributes["exception.message"].startswith("Error code: 401")
    assert open_ai_span.attributes.get("error.type") == "AuthenticationError"
    assert (
        "Traceback (most recent call last):" in event.attributes["exception.stacktrace"]
    )
    assert "openai.AuthenticationError" in event.attributes["exception.stacktrace"]
    assert "invalid_api_key" in event.attributes["exception.stacktrace"]


@pytest.mark.asyncio
async def test_chat_async_exception(
    instrument_legacy, span_exporter, async_openai_client
):
    async_openai_client.api_key = "invalid"
    with pytest.raises(Exception):
        await async_openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.chat",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_OPENAI_API_BASE)
        == "https://api.openai.com/v1/"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is False
    assert open_ai_span.status.status_code == StatusCode.ERROR
    assert open_ai_span.status.description.startswith("Error code: 401")
    events = open_ai_span.events
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "openai.AuthenticationError"
    assert event.attributes["exception.message"].startswith("Error code: 401")
    assert (
        "Traceback (most recent call last):" in event.attributes["exception.stacktrace"]
    )
    assert "openai.AuthenticationError" in event.attributes["exception.stacktrace"]
    assert "invalid_api_key" in event.attributes["exception.stacktrace"]
    assert open_ai_span.attributes.get("error.type") == "AuthenticationError"


@pytest.mark.vcr
def test_chat_streaming_not_consumed(
    instrument_legacy, span_exporter, log_exporter, reader, openai_client
):
    """Test that streaming responses are properly instrumented even when not consumed"""

    # Create streaming response but don't consume it
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    # Don't consume the response - this should still create proper traces and metrics
    del response

    # Force garbage collection to trigger cleanup
    import gc

    gc.collect()

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.chat"

    # Verify span was properly closed
    assert open_ai_span.status.status_code == StatusCode.OK
    assert open_ai_span.end_time is not None
    assert open_ai_span.end_time > open_ai_span.start_time

    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "gpt-3.5-turbo"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True
    assert open_ai_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "chat"

    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content")
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"


@pytest.mark.vcr
def test_chat_streaming_partial_consumption(
    instrument_legacy, span_exporter, log_exporter, reader, openai_client
):
    """Test that streaming responses are properly instrumented when partially consumed"""

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True,
    )

    # Consume only the first chunk
    first_chunk = next(iter(response))
    assert first_chunk is not None

    del response

    import gc

    gc.collect()

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.chat"

    assert open_ai_span.status.status_code == StatusCode.OK
    assert open_ai_span.end_time is not None

    assert (
        open_ai_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL) == "gpt-3.5-turbo"
    )
    assert open_ai_span.attributes.get(SpanAttributes.LLM_IS_STREAMING) is True

    # Should have at least one event from the consumed chunk
    events = open_ai_span.events
    assert len(events) >= 1


@pytest.mark.vcr
def test_chat_streaming_exception_during_consumption(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    """Test that streaming responses handle exceptions during consumption properly"""

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a short story"}],
        stream=True,
    )

    # Simulate exception during consumption
    count = 0
    try:
        for chunk in response:
            count += 1
            if count == 2:  # Interrupt after second chunk
                raise Exception("Simulated interruption")
    except Exception as e:
        # Force cleanup by deleting the response object
        del response
        import gc

        gc.collect()
        # Re-raise to verify the exception was caught
        assert "Simulated interruption" in str(e)

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    open_ai_span = spans[0]
    assert open_ai_span.name == "openai.chat"

    # Verify span was properly closed (status should be OK since exception was in user code, not in our iterator)
    assert open_ai_span.status.status_code == StatusCode.OK
    assert open_ai_span.end_time is not None

    # Should have events from the consumed chunks before exception
    events = open_ai_span.events
    assert len(events) >= 2  # At least 2 chunk events before exception


@pytest.mark.vcr
def test_chat_streaming_memory_leak_prevention(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    """Test that creating many streams without consuming them doesn't cause memory leaks"""
    import gc
    import weakref

    initial_spans = len(span_exporter.get_finished_spans())

    # Create a stream without consuming it
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    # Create weak reference to track if object is garbage collected
    weak_ref = weakref.ref(response)

    del response

    gc.collect()

    # Verify object was garbage collected
    assert weak_ref() is None, "Stream object was not garbage collected"

    # Verify span was properly closed
    final_spans = span_exporter.get_finished_spans()
    new_spans = len(final_spans) - initial_spans
    assert new_spans == 1, f"Expected 1 new span, got {new_spans}"

    # Verify span is properly closed
    span = final_spans[-1]
    assert span.name == "openai.chat"
    assert span.status.status_code == StatusCode.OK
    assert span.end_time is not None

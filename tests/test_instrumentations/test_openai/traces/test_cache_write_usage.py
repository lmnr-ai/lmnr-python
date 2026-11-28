import httpx2
from openai import OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import (
    OpenAIInstrumentor,
)


def _client(body: dict[str, object]) -> OpenAI:
    transport = httpx2.MockTransport(lambda _: httpx2.Response(200, json=body))
    return OpenAI(api_key="test", http_client=httpx2.Client(transport=transport))


def test_chat_records_cache_write_tokens(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
):
    client = _client(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5.6",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 5,
                "total_tokens": 2005,
                "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1900},
            },
        }
    )

    _ = client.chat.completions.create(
        model="gpt-5.6", messages=[{"role": "user", "content": "hi"}]
    )

    attributes = span_exporter.get_finished_spans()[0].attributes or {}
    assert attributes["gen_ai.usage.cache_write.input_tokens"] == 1900


def test_responses_records_cache_write_tokens(
    instrumentor: OpenAIInstrumentor,
    span_exporter: InMemorySpanExporter,
):
    client = _client(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 2000,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1900},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2005,
            },
        }
    )

    _ = client.responses.create(model="gpt-5.6", input="hi")

    attributes = span_exporter.get_finished_spans()[0].attributes or {}
    assert attributes["gen_ai.usage.cache_write.input_tokens"] == 1900

"""Runs in a fresh interpreter: importing `langfuse.openai` patches the openai
SDK for the rest of the process, so in the main test session every later
openai test would also record Langfuse's `OpenAI-generation` spans.

    python langfuse_openai.py WITH_TOOLS

WITH_TOOLS: `0` or `1`. Prints one JSON document with the attributes of the
LLM spans Laminar received.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, cast

os.environ.update(
    LANGFUSE_PUBLIC_KEY="pk-lf-test",
    LANGFUSE_SECRET_KEY="sk-lf-test",
    LANGFUSE_HOST="http://127.0.0.1:1",
    LANGFUSE_MEDIA_UPLOAD_ENABLED="False",
)

import httpx2
from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse.openai import OpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.bridges import BridgeContext
from lmnr.opentelemetry_lib.bridges.langfuse import LangfuseBridge
from lmnr.opentelemetry_lib.tracing.attributes import SPAN_TYPE
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor

with_tools = sys.argv[1] == "1"

laminar_exporter = InMemorySpanExporter()
laminar_processor = LaminarSpanProcessor(
    api_key="test", exporter=laminar_exporter, disable_batch=True
)
laminar_provider = TracerProvider()
laminar_provider.add_span_processor(laminar_processor)

langfuse = Langfuse(tracer_provider=TracerProvider())
# Nothing leaves the process: Langfuse exports to an in-memory exporter.
for rm in LangfuseResourceManager._instances.values():
    active = cast(Any, rm.tracer_provider)._active_span_processor
    for proc in getattr(active, "_span_processors", ()):
        batch = getattr(proc, "_batch_processor", None)
        if batch is not None:
            batch._exporter = InMemorySpanExporter()

bridge = LangfuseBridge()
assert bridge.install(
    BridgeContext(span_processor=laminar_processor, tracer_provider=laminar_provider)
)


def _chat_completion() -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": None if with_tools else "4"}
    if with_tools:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 2, "b": 2}'},
            }
        ]
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if with_tools else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    }


client = OpenAI(
    api_key="sk-test",
    http_client=httpx2.Client(
        transport=httpx2.MockTransport(
            lambda _request: httpx2.Response(200, json=_chat_completion())
        )
    ),
)
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "parameters": {"type": "object", "properties": {"a": {}, "b": {}}},
        },
    }
]
kwargs: dict[str, Any] = {"tools": tools} if with_tools else {}
_response = client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "2+2?"}], **kwargs
)
langfuse.flush()
langfuse.shutdown()
bridge.uninstall()

print(
    json.dumps(
        [
            {k: list(v) if isinstance(v, tuple) else v for k, v in (s.attributes or {}).items()}
            for s in laminar_exporter.get_finished_spans()
            if (s.attributes or {}).get(SPAN_TYPE) == "LLM"
        ],
        default=str,
    )
)

"""LiteLLM's `langfuse_otel` callback: translation of its hybrid spans, and
attaching to the private providers it builds."""

from __future__ import annotations

import json
from typing import Any, cast

from opentelemetry.sdk.trace import TracerProvider

from lmnr.opentelemetry_lib.bridges.langfuse import LangfuseBridge
from lmnr.opentelemetry_lib.bridges.langfuse.processor import LangfuseBridgeSpanProcessor
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SPAN_TYPE,
)

from .conftest import Laminar
from .utils import FakeSpan, litellm_required, translate


def _bridge_processors(provider: TracerProvider) -> list[LangfuseBridgeSpanProcessor]:
    return [
        p
        for p in provider._active_span_processor._span_processors  # pyright: ignore[reportPrivateUsage]
        if isinstance(p, LangfuseBridgeSpanProcessor)
    ]


def test_translator_routes_litellm_hybrid_span_through_openinference():
    """LiteLLM's `langfuse_otel` callback emits a HYBRID span: openinference
    `llm.*` attrs (model, tokens, indexed messages, tools) AND `langfuse.*`
    trace attrs (session.id, user.id) — but NO `langfuse.observation.type`.

    The openinference path must win (it carries the LLM data the langfuse path
    can't see without an observation type) AND it must still promote the
    `langfuse.*` trace-level session/user attributes.
    """
    span = FakeSpan(
        {
            # openinference half
            "openinference.span.kind": "LLM",
            "llm.model_name": "gpt-4o",
            "llm.token_count.prompt": 11,
            "llm.token_count.completion": 5,
            "llm.token_count.total": 16,
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "hi",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "hello",
            # langfuse half (no observation.type!)
            "session.id": "sess-9",
            "user.id": "user-3",
            "langfuse.observation.input": '{"messages": [{"role": "user"}]}',
        },
        scope_name="litellm",
    )
    span.attributes = translate(span)

    assert span.attributes[SPAN_TYPE] == "LLM"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert span.attributes["gen_ai.usage.input_tokens"] == 11
    assert span.attributes["gen_ai.usage.output_tokens"] == 5
    assert json.loads(cast(str, span.attributes["gen_ai.input.messages"])) == [
        {"role": "user", "content": "hi"}
    ]
    assert json.loads(cast(str, span.attributes["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "hello"}
    ]
    # Trace-level promotion from the langfuse half must still happen.
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.session_id"] == "sess-9"
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.user_id"] == "user-3"


@litellm_required
def test_litellm_bridge_attaches_to_existing_logger(laminar: Laminar, bridge: LangfuseBridge):
    from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
    from litellm.litellm_core_utils import litellm_logging

    logger_obj = LangfuseOtelLogger(callback_name="langfuse_otel")
    provider = logger_obj._tracer_provider
    assert isinstance(provider, TracerProvider)

    original_loggers = list(litellm_logging._in_memory_loggers)
    litellm_logging._in_memory_loggers.append(logger_obj)
    try:
        assert bridge.install(laminar.provider, laminar.processor)
        assert len(_bridge_processors(provider)) == 1

        # LiteLLM exports everything on its provider to Langfuse, so
        # everything is forwarded.
        with provider.get_tracer("litellm").start_as_current_span("litellm_request"):
            pass
        assert "litellm_request" in laminar.spans()

        bridge.uninstall()
        assert _bridge_processors(provider) == []
    finally:
        litellm_logging._in_memory_loggers[:] = original_loggers


@litellm_required
def test_litellm_bridge_patches_factory_for_late_loggers(laminar: Laminar, bridge: LangfuseBridge):
    from litellm.litellm_core_utils import litellm_logging

    original_loggers = list(litellm_logging._in_memory_loggers)
    original_factory = litellm_logging._init_custom_logger_compatible_class  # pyright: ignore[reportUnknownVariableType]
    try:
        assert bridge.install(laminar.provider, laminar.processor)
        assert litellm_logging._init_custom_logger_compatible_class is not original_factory

        logger_obj = litellm_logging._init_custom_logger_compatible_class(
            "langfuse_otel", internal_usage_cache=None, llm_router=None
        )
        provider = cast(Any, logger_obj)._tracer_provider
        assert len(_bridge_processors(provider)) == 1

        # Per-credential providers are created lazily and attached as they appear.
        _tracer = cast(Any, logger_obj)._get_tracer_with_dynamic_headers(
            {"Authorization": "Basic abc"}
        )
        for cached in cast(Any, logger_obj)._tracer_provider_cache.values():
            assert len(_bridge_processors(getattr(cached, "provider", cached))) == 1

        # Other OTel-based callbacks built by the same factory are left alone.
        other = litellm_logging._init_custom_logger_compatible_class(
            "otel", internal_usage_cache=None, llm_router=None
        )
        other_provider = getattr(other, "_tracer_provider", None)
        if isinstance(other_provider, TracerProvider):
            assert _bridge_processors(other_provider) == []

        bridge.uninstall()
        assert litellm_logging._init_custom_logger_compatible_class is original_factory
    finally:
        litellm_logging._init_custom_logger_compatible_class = original_factory
        litellm_logging._in_memory_loggers[:] = original_loggers

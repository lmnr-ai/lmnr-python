"""Tests for the LiteLLM `langfuse_otel` bridge."""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    SpanExporter,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import use_span

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse import (
    LangfuseAttributeTranslator,
    get_langfuse_instrumentor,
    is_llm_span,
)
from lmnr.opentelemetry_lib.tracing import (
    get_tracer_wrapper,
    init_tracing,
    reset_tracing,
)
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SPAN_TYPE,
)

from .utils import (
    FakeSpan,
    litellm_required,
    reset_langfuse_bridge_state,
)


def test_translator_routes_litellm_hybrid_span_through_openinference():
    """LiteLLM's `langfuse_otel` callback emits a HYBRID span: openinference
    `llm.*` attrs (model, tokens, indexed messages, tools) AND `langfuse.*`
    trace attrs (session.id, user.id) — but NO `langfuse.observation.type`.

    The openinference path must win (it carries the LLM data the langfuse path
    can't see without an observation type) AND it must still promote the
    `langfuse.*` trace-level session/user attributes.
    """
    translator = LangfuseAttributeTranslator()
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
    translator.on_end(cast(Any, span))

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
def test_litellm_bridge_attaches_to_existing_logger():
    """A `LangfuseOtelLogger` already constructed before the bridge installs
    must get the translator + Laminar span processor attached to its private
    `_tracer_provider`."""
    from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
    from litellm.litellm_core_utils import litellm_logging

    reset_langfuse_bridge_state()

    logger_obj = LangfuseOtelLogger(callback_name="langfuse_otel")
    provider = logger_obj._tracer_provider
    assert isinstance(provider, TracerProvider)

    original_loggers = list(litellm_logging._in_memory_loggers)
    litellm_logging._in_memory_loggers.append(logger_obj)
    try:
        wrapper = get_tracer_wrapper()
        assert wrapper is not None
        instrumentor = get_langfuse_instrumentor()
        instrumentor.instrument(
            lmnr_tracer_provider=wrapper.tracer_provider,
            lmnr_span_processor=wrapper.span_processor,
        )
        processors = list(provider._active_span_processor._span_processors)
        assert any(
            isinstance(p, LangfuseAttributeTranslator) for p in processors
        ), "translator must be attached to the LiteLLM logger's provider"
        assert (
            wrapper.span_processor in processors
        ), "Laminar span processor must be attached to the LiteLLM provider"

        instrumentor.uninstrument()
        after = list(provider._active_span_processor._span_processors)
        assert not any(
            isinstance(p, LangfuseAttributeTranslator) for p in after
        ), "uninstrument must detach the translator from the LiteLLM provider"
        assert wrapper.span_processor not in after
    finally:
        litellm_logging._in_memory_loggers[:] = original_loggers
        reset_langfuse_bridge_state()


@litellm_required
def test_litellm_bridge_patches_factory_for_late_loggers():
    """A `langfuse_otel` logger constructed AFTER the bridge installs (via
    LiteLLM's `_init_custom_logger_compatible_class` factory) must also get
    dual-attached. The factory patch must be reverted on uninstrument."""
    from litellm.litellm_core_utils import litellm_logging

    reset_langfuse_bridge_state()

    original_loggers = list(litellm_logging._in_memory_loggers)
    original_factory = litellm_logging._init_custom_logger_compatible_class  # pyright: ignore[reportUnknownVariableType]
    try:
        wrapper = get_tracer_wrapper()
        assert wrapper is not None
        instrumentor = get_langfuse_instrumentor()
        instrumentor.instrument(
            lmnr_tracer_provider=wrapper.tracer_provider,
            lmnr_span_processor=wrapper.span_processor,
        )
        # The factory must have been wrapped.
        assert (
            litellm_logging._init_custom_logger_compatible_class is not original_factory
        )

        logger_obj = litellm_logging._init_custom_logger_compatible_class(
            "langfuse_otel",
            internal_usage_cache=None,
            llm_router=None,
        )
        assert logger_obj is not None
        provider = cast(Any, logger_obj)._tracer_provider
        processors = list(provider._active_span_processor._span_processors)
        assert any(
            isinstance(p, LangfuseAttributeTranslator) for p in processors
        ), "late-constructed LiteLLM logger must be bridged via the factory"
        assert wrapper.span_processor in processors

        instrumentor.uninstrument()
        # Factory patch must be reverted.
        assert litellm_logging._init_custom_logger_compatible_class is original_factory
    finally:
        litellm_logging._init_custom_logger_compatible_class = original_factory
        litellm_logging._in_memory_loggers[:] = original_loggers
        reset_langfuse_bridge_state()


@litellm_required
def test_litellm_factory_does_not_bridge_non_langfuse_loggers():
    """The factory patch must bridge ONLY `langfuse_otel`. Other OTel-based
    LiteLLM callbacks built through the same factory also carry a private
    `_tracer_provider` (the base `otel` callback is exactly such a case);
    attaching Laminar's translator to them would ship unrelated spans into
    Laminar. The base `otel` provider is in fact the global (Laminar's own)
    provider, so we can't assert on its processors — instead we assert the
    factory patch never routes a non-langfuse logger through
    `ProviderAttachment.attach`."""
    from litellm.litellm_core_utils import litellm_logging

    reset_langfuse_bridge_state()

    original_loggers = list(litellm_logging._in_memory_loggers)
    original_factory = litellm_logging._init_custom_logger_compatible_class  # pyright: ignore[reportUnknownVariableType]
    try:
        wrapper = get_tracer_wrapper()
        assert wrapper is not None
        instrumentor = get_langfuse_instrumentor()
        instrumentor.instrument(
            lmnr_tracer_provider=wrapper.tracer_provider,
            lmnr_span_processor=wrapper.span_processor,
        )
        attached: list[TracerProvider | None] = []
        assert instrumentor._provider_attachment is not None
        original_attach = instrumentor._provider_attachment.attach

        def _spy(provider: TracerProvider | None):
            attached.append(provider)
            return original_attach(provider)

        instrumentor._provider_attachment.attach = _spy  # pyright: ignore[reportAttributeAccessIssue]

        # The base `otel` callback is an `OpenTelemetry` instance (NOT a
        # `LangfuseOtelLogger`) that carries its own `_tracer_provider`. The
        # factory patch must NOT attach to it.
        _logger = litellm_logging._init_custom_logger_compatible_class(
            "otel",
            internal_usage_cache=None,
            llm_router=None,
        )
        assert attached == [], (
            "non-langfuse LiteLLM logger must NOT be routed to " "ProviderAttachment.attach"
        )

        # A `langfuse_otel` logger built through the same factory MUST attach.
        _logger = litellm_logging._init_custom_logger_compatible_class(
            "langfuse_otel",
            internal_usage_cache=None,
            llm_router=None,
        )
        assert attached, "langfuse_otel logger must be bridged via the factory"

        instrumentor.uninstrument()
    finally:
        litellm_logging._init_custom_logger_compatible_class = original_factory
        litellm_logging._in_memory_loggers[:] = original_loggers
        reset_langfuse_bridge_state()


def test_is_llm_span_detects_llm_parents():
    """`_is_llm_span` gates LiteLLM primary-span forcing: it must recognise
    Laminar's own `litellm.completion` LLM span (so folding stays correct) and
    NOT mistake a plain `@observe` root for one (so the bridge forces a
    `litellm_request` span there)."""
    assert is_llm_span(cast(Any, FakeSpan({SPAN_TYPE: "LLM"})))
    assert is_llm_span(cast(Any, FakeSpan({"openinference.span.kind": "LLM"})))
    assert is_llm_span(cast(Any, FakeSpan({"gen_ai.request.model": "gpt-4o"})))
    assert is_llm_span(cast(Any, FakeSpan({"gen_ai.response.model": "gpt-4o"})))
    assert is_llm_span(cast(Any, FakeSpan({"gen_ai.system": "openai"})))
    # A plain root / tool span must NOT read as LLM.
    assert not is_llm_span(cast(Any, FakeSpan({})))
    assert not is_llm_span(cast(Any, FakeSpan({SPAN_TYPE: "DEFAULT"})))
    assert not is_llm_span(cast(Any, FakeSpan({"langfuse.observation.type": "span"})))


@litellm_required
def test_litellm_bridge_wraps_and_unwraps_logger_methods():
    """The bridge must wrap each `langfuse_otel` logger's
    `_get_tracer_with_dynamic_headers` (layer 1: attach to per-credential cache
    providers) and `_get_span_context` (layer 2: force a `litellm_request`
    span instead of folding onto a non-LLM parent). The layer-2 wrapper reads
    `trace.get_current_span()`, so we drive it by activating a span:
    - a non-LLM active span (a user's `@observe` root) must be reported as
      `parent_span=None` while the context is preserved, flipping LiteLLM's
      `should_create_primary_span` to True;
    - an LLM active span (Laminar's own `litellm.completion`) must be left as
      the parent so folding stays correct and no duplicate span is created.
    `uninstrument` must restore both originals.
    """
    from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
    from litellm.litellm_core_utils import litellm_logging

    reset_langfuse_bridge_state()

    logger_obj = LangfuseOtelLogger(callback_name="langfuse_otel")
    orig_get_tracer = logger_obj._get_tracer_with_dynamic_headers  # pyright: ignore[reportUnknownVariableType]
    orig_get_ctx = logger_obj._get_span_context  # pyright: ignore[reportUnknownVariableType]

    original_loggers = list(litellm_logging._in_memory_loggers)
    litellm_logging._in_memory_loggers.append(logger_obj)
    probe_tracer = TracerProvider().get_tracer("probe")
    try:
        wrapper = get_tracer_wrapper()
        assert wrapper is not None
        instrumentor = get_langfuse_instrumentor()
        instrumentor.instrument(
            lmnr_tracer_provider=wrapper.tracer_provider,
            lmnr_span_processor=wrapper.span_processor,
        )

        # Both methods must now be wrapped.
        assert logger_obj._get_tracer_with_dynamic_headers is not orig_get_tracer
        assert logger_obj._get_span_context is not orig_get_ctx

        # Layer 2 gate: a non-LLM active span (e.g. an `@observe` root) is
        # nulled out as a parent so LiteLLM creates its own litellm_request.
        non_llm = probe_tracer.start_span("run")
        with use_span(non_llm, end_on_exit=False):
            ctx, parent = logger_obj._get_span_context({})
        assert parent is None, "non-LLM parent must be nulled to force a primary span"
        assert ctx is not None, "parent context must be preserved for nesting"

        # ...an LLM active span (Laminar's litellm.completion) is preserved so
        # LiteLLM folds onto it instead of creating a duplicate nested LLM span.
        llm = probe_tracer.start_span(
            "litellm.completion", attributes={SPAN_TYPE: "LLM"}
        )
        with use_span(llm, end_on_exit=False):
            _, parent_llm = logger_obj._get_span_context({})
        assert parent_llm is llm, "LLM parent must be preserved (no duplicate span)"

        instrumentor.uninstrument()
        # Both originals restored.
        assert logger_obj._get_tracer_with_dynamic_headers == orig_get_tracer
        assert logger_obj._get_span_context == orig_get_ctx
        assert instrumentor._litellm_bridge is None
    finally:
        litellm_logging._in_memory_loggers[:] = original_loggers
        reset_langfuse_bridge_state()


def test_bridge_is_rebound_to_the_new_processor_after_a_reinit():
    """`LangfuseInstrumentor.instrument()` returns early once already
    instrumented, so an initialize()/shutdown()/initialize() cycle used to leave the
    bridge — and every Langfuse-owned provider it attached to — holding the
    RETIRED `LaminarSpanProcessor`. Langfuse spans then hit `on_end` on a
    shut-down processor and never reached the new exporter.

    `Laminar.initialize()` now drives `LangfuseInstrumentor.rebind()`; nothing
    else would, since LANGFUSE is never in the default instrument set and
    `connect_to_langfuse()` is a one-shot.
    """
    import lmnr.sdk.laminar as laminar_mod
    from lmnr.opentelemetry_lib.tracing import wrapper as wrapper_mod

    def boot(exporter: SpanExporter):
        def injected(*args: Any, **kwargs: Any):
            kwargs["exporter"] = exporter
            return init_tracing(*args, **kwargs)

        # The session fixture already initialized Laminar, and `initialize()`
        # short-circuits on its own flag.
        Laminar._Laminar__initialized = False  # pyright: ignore[reportAttributeAccessIssue]
        with patch.object(laminar_mod, "init_tracing", side_effect=injected):
            Laminar.initialize(
                project_api_key="k", disable_batch=True, instruments=set()
            )

    def processors(provider: TracerProvider) -> tuple[SpanProcessor, ...]:
        return provider._active_span_processor._span_processors

    saved_wrapper = wrapper_mod._tracer_wrapper
    saved_options = wrapper_mod._session_recording_options
    saved_initialized = Laminar.is_initialized()
    wrapper_mod._tracer_wrapper = None
    wrapper_mod._session_recording_options = None
    reset_langfuse_bridge_state()

    try:
        exp1 = InMemorySpanExporter()
        boot(exp1)
        wrapper = get_tracer_wrapper()
        assert wrapper is not None
        retired = wrapper.span_processor
        assert Laminar.connect_to_langfuse() is True

        # A Langfuse-owned provider, as the resource-manager hook would supply.
        lf_provider = TracerProvider()
        lf_instrumentor = get_langfuse_instrumentor()
        assert lf_instrumentor._provider_attachment is not None
        lf_instrumentor._provider_attachment.attach(lf_provider)
        assert retired in processors(lf_provider)

        lf_provider.get_tracer("langfuse-sdk").start_span("lf1").end()
        assert [s.name for s in exp1.get_finished_spans()] == ["lf1"]

        Laminar.shutdown()

        exp2 = InMemorySpanExporter()
        boot(exp2)
        wrapper = get_tracer_wrapper()
        assert wrapper is not None
        current = wrapper.span_processor
        assert current is not retired

        assert get_langfuse_instrumentor()._lmnr_span_processor is current
        assert retired not in processors(lf_provider)
        assert current in processors(lf_provider)

        exp1.clear()
        lf_provider.get_tracer("langfuse-sdk").start_span("lf2").end()
        assert [s.name for s in exp2.get_finished_spans()] == ["lf2"]
        assert exp1.get_finished_spans() == ()
    finally:
        get_langfuse_instrumentor().uninstrument()
        reset_langfuse_bridge_state()
        reset_tracing()
        wrapper_mod._tracer_wrapper = saved_wrapper
        wrapper_mod._session_recording_options = saved_options
        Laminar._Laminar__initialized = saved_initialized  # pyright: ignore[reportAttributeAccessIssue]


def test_rebind_is_a_no_op_when_the_bridge_is_not_installed():
    reset_langfuse_bridge_state()
    assert (
        get_langfuse_instrumentor().rebind(
            lmnr_tracer_provider=MagicMock(), lmnr_span_processor=MagicMock()
        )
        is False
    )

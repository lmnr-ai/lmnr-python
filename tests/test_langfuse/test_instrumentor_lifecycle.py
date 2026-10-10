"""Install / uninstall lifecycle tests for `LangfuseInstrumentor`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse import (
    LangfuseAttributeTranslator,
    LangfuseInstrumentor,
    get_langfuse_instrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse.provider_attachment import (
    ProviderAttachment,
)
from lmnr.opentelemetry_lib.tracing import get_tracer_wrapper

from .utils import (
    langfuse_sdk_required,
    reset_langfuse_bridge_state,
    silence_langfuse_background_threads,
)


@langfuse_sdk_required
def test_late_attach_patches_future_langfuse_clients(span_exporter: InMemorySpanExporter):
    """The resource-manager patch means Langfuse clients created AFTER the
    bridge is installed still get dual-attached."""
    from langfuse._client.resource_manager import LangfuseResourceManager

    # Install bridge with NO langfuse client yet.
    LangfuseResourceManager._instances.clear()
    assert Laminar.connect_to_langfuse() is True

    # Now create a Langfuse client — this triggers
    # `LangfuseResourceManager._initialize_instance`, which our bridge
    # has monkey-patched to attach Laminar's processor.
    from langfuse import Langfuse, observe

    client = Langfuse()
    silence_langfuse_background_threads(LangfuseResourceManager._instances.values())

    try:

        @observe
        def late() -> int:
            return 7

        _seven = late()
    finally:
        client.shutdown()
        LangfuseResourceManager._instances.clear()

    names = [s.name for s in span_exporter.get_finished_spans()]
    assert "late" in names, f"late-created Langfuse client was not bridged: {names}"


def test_instrumentor_skips_laminar_own_provider():
    """Guardrail: if Langfuse somehow ends up sharing Laminar's
    TracerProvider, `ProviderAttachment.attach` must skip it (otherwise
    Laminar's processor would be attached to Laminar's provider twice →
    duplicate spans on every Laminar export).
    """

    wrapper = get_tracer_wrapper()
    assert wrapper is not None
    initial_count = len(
        wrapper.tracer_provider._active_span_processor._span_processors
    )
    provider_attachment = ProviderAttachment(
        translator=None, lmnr_span_processor=wrapper.span_processor
    )
    provider_attachment.attach(wrapper.tracer_provider)
    after_count = len(wrapper.tracer_provider._active_span_processor._span_processors)
    assert (
        after_count == initial_count
    ), "attached to Laminar's own provider — would cause duplicate exports"


def test_instrument_rolls_back_translator_if_attach_phase_raises():
    """Regression: if `attach_to_existing_langfuse_providers` or
    `patch_resource_manager` raises, the translator that was already
    prepended to Laminar's provider in step 1 must be removed and all
    instance-level state cleared. Otherwise, a subsequent `instrument()`
    (e.g. via `Laminar.connect_to_langfuse()`) would pass the
    `is_instrumented_by_opentelemetry` guard and prepend a SECOND translator,
    double-translating every Langfuse span.
    """

    # Clean state.
    reset_langfuse_bridge_state()

    provider = TracerProvider()

    def count_translators() -> int:
        return sum(
            1
            for p in provider._active_span_processor._span_processors
            if isinstance(p, LangfuseAttributeTranslator)
        )

    baseline = count_translators()

    instrumentor = get_langfuse_instrumentor()

    # Force `attach_to_existing_langfuse_providers` to blow up on first
    # install.
    def exploding_attach(self):
        raise RuntimeError("simulated attach failure")

    original_attach = ProviderAttachment.attach_to_existing_langfuse_providers
    ProviderAttachment.attach_to_existing_langfuse_providers = exploding_attach
    try:
        with pytest.raises(RuntimeError, match="simulated attach failure"):
            instrumentor.instrument(
                lmnr_tracer_provider=provider,
                lmnr_span_processor=MagicMock(),
            )
    finally:
        ProviderAttachment.attach_to_existing_langfuse_providers = original_attach

    # Translator must have been rolled back.
    assert (
        count_translators() == baseline
    ), "translator must be removed on partial install failure"
    assert instrumentor.is_instrumented_by_opentelemetry is False
    assert instrumentor._translator is None

    # A subsequent successful install must attach exactly ONE translator —
    # not two, which is what would happen if the failed-install translator
    # were still around.
    instrumentor2 = get_langfuse_instrumentor()
    instrumentor2.instrument(
        lmnr_tracer_provider=provider,
        lmnr_span_processor=MagicMock(),
    )
    assert count_translators() == baseline + 1
    instrumentor2.uninstrument()


def test_instrument_skips_shared_laminar_provider_without_tracerwrapper():
    """Regression: during auto-install via `init_instrumentations`,
    the tracer wrapper is published AFTER `init_instrumentations`
    returns. If a pre-existing Langfuse client happens to share Laminar's
    newly-created `TracerProvider`, `ProviderAttachment.attach`'s
    `get_tracer_wrapper()` fallback guard returns None
    (the wrapper isn't set yet) and the translator + Laminar span processor
    would be double-attached.

    `instrument()` must pre-register `id(lmnr_tracer_provider)` in
    `_handled_providers` so the short-circuit works independently of the
    tracing lifecycle.
    """
    from opentelemetry.sdk.trace import TracerProvider

    # Start from a clean slate.
    reset_langfuse_bridge_state()

    # A fresh provider standing in for Laminar's. We deliberately do NOT
    # publish a tracer wrapper pointing at this provider — that's
    # exactly the case the guard has to cover during auto-install.
    shared_provider = TracerProvider()

    mock_processor = MagicMock()
    instrumentor = get_langfuse_instrumentor()
    instrumentor.instrument(
        lmnr_tracer_provider=shared_provider,
        lmnr_span_processor=mock_processor,
    )

    # One translator from `_prepend_span_processor` in `instrument()` —
    # baseline.
    processors_after_install = list(
        shared_provider._active_span_processor._span_processors
    )
    translator_count = sum(
        1
        for p in processors_after_install
        if isinstance(p, LangfuseAttributeTranslator)
    )
    assert translator_count == 1
    assert mock_processor not in processors_after_install, (
        "Laminar span processor should NOT be attached to Laminar's own "
        "provider — the translator is enough there"
    )

    # Now simulate Langfuse's resource manager calling `attach` with the same
    # provider. The pre-registered id must prevent a double-attach.
    assert instrumentor._provider_attachment is not None
    instrumentor._provider_attachment.attach(shared_provider)

    processors_after_simulated_langfuse = list(
        shared_provider._active_span_processor._span_processors
    )
    translator_count_after = sum(
        1
        for p in processors_after_simulated_langfuse
        if isinstance(p, LangfuseAttributeTranslator)
    )
    assert (
        translator_count_after == 1
    ), "translator should not be attached twice to Laminar's provider"
    assert mock_processor not in processors_after_simulated_langfuse, (
        "Laminar span processor should not be appended to Laminar's own "
        "provider via the Langfuse attach path"
    )

    instrumentor.uninstrument()


@langfuse_sdk_required
def test_uninstrument_removes_translator_and_clears_state():
    """Regression: `uninstrument` must detach the translator from Laminar's
    provider and clear instance-level state (`_provider_attachment`,
    `_translator`, `_lmnr_span_processor`, `_lmnr_tracer_provider`).
    Otherwise a subsequent `instrument()` call would prepend a SECOND
    translator onto Laminar's provider (the first was never removed), and
    the retained `_provider_attachment` would still hold ids from the
    previous session, making `attach_to_existing_langfuse_providers` skip
    providers it saw last time around.
    """
    from langfuse._client.resource_manager import LangfuseResourceManager

    LangfuseResourceManager._instances.clear()
    wrapper = get_tracer_wrapper()
    assert wrapper is not None
    lmnr_provider = wrapper.tracer_provider

    def count_translators() -> int:
        return sum(
            1
            for p in lmnr_provider._active_span_processor._span_processors
            if isinstance(p, LangfuseAttributeTranslator)
        )

    # Start from a clean slate in case an earlier test left state behind.
    reset_langfuse_bridge_state()

    baseline_translators = count_translators()

    instrumentor = get_langfuse_instrumentor()
    instrumentor.instrument(
        lmnr_tracer_provider=lmnr_provider,
        lmnr_span_processor=wrapper.span_processor,
    )
    assert instrumentor.is_instrumented_by_opentelemetry is True
    assert count_translators() == baseline_translators + 1

    instrumentor.uninstrument()

    # Translator must have been removed from Laminar's provider.
    assert count_translators() == baseline_translators
    # All instance-level state must be cleared so a fresh install starts clean.
    assert instrumentor.is_instrumented_by_opentelemetry is False
    assert instrumentor._translator is None
    assert instrumentor._lmnr_span_processor is None
    assert instrumentor._lmnr_tracer_provider is None
    assert instrumentor._provider_attachment is None

    # Re-install: the translator count should increase by exactly one again —
    # NOT two, which is what would happen if the previous translator was
    # still attached.
    instrumentor2 = get_langfuse_instrumentor()
    instrumentor2.instrument(
        lmnr_tracer_provider=lmnr_provider,
        lmnr_span_processor=wrapper.span_processor,
    )
    assert count_translators() == baseline_translators + 1
    instrumentor2.uninstrument()
    assert count_translators() == baseline_translators


@langfuse_sdk_required
def test_stray_reconstruction_does_not_wipe_live_state():
    """Regression: `BaseInstrumentor.__new__` caches a single instance per
    class, but Python still calls `__init__` on that cached instance every
    time `LangfuseInstrumentor()` is invoked. A stray direct construction
    (bypassing `get_langfuse_instrumentor()`) after the bridge is already
    installed must NOT reset `_translator` / `_provider_attachment` /
    `_litellm_bridge` to `None` while `is_instrumented_by_opentelemetry`
    stays `True` — that would make `instrument()` no-op (already
    instrumented), `rebind()` report success without anything to rebind, and
    `_teardown()` unable to find what to detach.
    """
    from langfuse._client.resource_manager import LangfuseResourceManager

    LangfuseResourceManager._instances.clear()
    wrapper = get_tracer_wrapper()
    assert wrapper is not None
    lmnr_provider = wrapper.tracer_provider

    reset_langfuse_bridge_state()

    instrumentor = get_langfuse_instrumentor()
    instrumentor.instrument(
        lmnr_tracer_provider=lmnr_provider,
        lmnr_span_processor=wrapper.span_processor,
    )
    assert instrumentor.is_instrumented_by_opentelemetry is True
    live_translator = instrumentor._translator
    live_provider_attachment = instrumentor._provider_attachment
    live_litellm_bridge = instrumentor._litellm_bridge
    assert live_translator is not None
    assert live_provider_attachment is not None
    assert live_litellm_bridge is not None

    # A stray direct construction must be the SAME instance and must NOT
    # clobber the state set up by `instrument()`.
    same_instance = LangfuseInstrumentor()
    assert same_instance is instrumentor
    assert same_instance.is_instrumented_by_opentelemetry is True
    assert same_instance._translator is live_translator
    assert same_instance._provider_attachment is live_provider_attachment
    assert same_instance._litellm_bridge is live_litellm_bridge

    instrumentor.uninstrument()
    assert instrumentor.is_instrumented_by_opentelemetry is False
    assert instrumentor._translator is None


def test_uninstrument_detaches_processors_from_langfuse_providers(span_exporter):
    """Regression: after `uninstrument`, every provider we previously attached
    the translator / Laminar span processor to must lose those processors.
    Re-install must also succeed (a stale `_provider_attachment` must not
    short-circuit re-attachment).

    We simulate a Langfuse-owned TracerProvider with a plain
    `sdk.trace.TracerProvider`, directed at `ProviderAttachment.attach`: the
    bridge treats any non-Laminar provider the same way, so this exercises
    the install/uninstall/reinstall flow deterministically without depending
    on whether the real Langfuse SDK reuses the global provider.
    """
    from opentelemetry.sdk.trace import TracerProvider

    wrapper = get_tracer_wrapper()
    assert wrapper is not None
    lmnr_provider = wrapper.tracer_provider
    lmnr_processor = wrapper.span_processor

    # Start from a known-clean state.
    reset_langfuse_bridge_state()

    fake_lf_provider = TracerProvider()
    baseline = list(fake_lf_provider._active_span_processor._span_processors)

    instrumentor = get_langfuse_instrumentor()
    instrumentor.instrument(
        lmnr_tracer_provider=lmnr_provider,
        lmnr_span_processor=lmnr_processor,
    )
    # Manually attach to our fake Langfuse-owned provider the same way the
    # monkey-patched `_initialize_instance` would in production.
    assert instrumentor._provider_attachment is not None
    instrumentor._provider_attachment.attach(fake_lf_provider)

    after_install = list(fake_lf_provider._active_span_processor._span_processors)
    assert any(isinstance(p, LangfuseAttributeTranslator) for p in after_install)
    assert lmnr_processor in after_install
    assert id(fake_lf_provider) in instrumentor._provider_attachment._attached_providers

    instrumentor.uninstrument()

    after_uninstall = list(fake_lf_provider._active_span_processor._span_processors)
    assert not any(
        isinstance(p, LangfuseAttributeTranslator) for p in after_uninstall
    ), "translator should be removed from the Langfuse-owned provider"
    assert (
        lmnr_processor not in after_uninstall
    ), "Laminar span processor should be removed from the Langfuse-owned provider"
    # Ordering of unrelated processors should be preserved.
    assert after_uninstall == baseline

    # Re-install must work — a stale `_provider_attachment` was cleared so
    # the existing provider is seen again.
    instrumentor.instrument(
        lmnr_tracer_provider=lmnr_provider,
        lmnr_span_processor=lmnr_processor,
    )
    instrumentor._provider_attachment.attach(fake_lf_provider)
    reinstalled = list(fake_lf_provider._active_span_processor._span_processors)
    assert any(isinstance(p, LangfuseAttributeTranslator) for p in reinstalled)
    assert lmnr_processor in reinstalled
    instrumentor.uninstrument()

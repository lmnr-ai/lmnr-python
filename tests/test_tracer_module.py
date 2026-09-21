"""Tests for the module-level tracing lifecycle in `lmnr.opentelemetry_lib.tracing`."""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from lmnr.opentelemetry_lib import tracing as tracing_mod
from lmnr.opentelemetry_lib.tracing import (
    get_session_recording_options,
    get_tracer_wrapper,
    is_tracing_initialized,
    reset_tracing,
)
from lmnr.opentelemetry_lib.tracing.tracer import (
    get_laminar_tracer_provider,
    get_tracer,
)


def test_get_laminar_tracer_provider_returns_the_laminar_provider(span_exporter):
    """Regression: this used to read `TracerWrapper.instance.__tracer_provider`,
    an attribute that never existed (no name mangling at module scope), so every
    call raised AttributeError — on a function exported from `lmnr`."""
    wrapper = get_tracer_wrapper()
    assert wrapper is not None

    provider = get_laminar_tracer_provider()

    assert isinstance(provider, TracerProvider)
    assert provider is wrapper.tracer_provider


def test_tracer_helpers_do_not_initialize_tracing(span_exporter, monkeypatch):
    """`get_tracer()` and `get_laminar_tracer_provider()` used to call
    `TracerWrapper()`, which silently built a full API-key-less tracer (exporter
    threads and an atexit hook included) when tracing was not initialized."""
    monkeypatch.setattr(tracing_mod, "_tracer_wrapper", None)

    assert is_tracing_initialized() is False

    provider = get_laminar_tracer_provider()
    assert provider is trace.get_tracer_provider()

    with get_tracer() as tracer:
        assert tracer is not None

    assert get_tracer_wrapper() is None, "must not have initialized tracing"


def test_session_recording_options_default_before_init(monkeypatch):
    monkeypatch.setattr(tracing_mod, "_session_recording_options", None)
    assert get_session_recording_options() == {"mask_input_options": None}


def test_reset_tracing_is_idempotent(monkeypatch):
    monkeypatch.setattr(tracing_mod, "_tracer_wrapper", None)
    monkeypatch.setattr(tracing_mod, "_session_recording_options", None)

    reset_tracing()
    reset_tracing()

    assert get_tracer_wrapper() is None

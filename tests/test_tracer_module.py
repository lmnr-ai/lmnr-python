"""Tests for the module-level tracing lifecycle in `lmnr.opentelemetry_lib.tracing`."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from lmnr.opentelemetry_lib import tracing as tracing_mod
from lmnr.opentelemetry_lib.tracing import (
    get_session_recording_options,
    get_tracer_wrapper,
    init_tracing,
    is_tracing_initialized,
    reset_tracing,
    shutdown_tracing,
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


@pytest.fixture
def isolated_tracing():
    """Detach the session-wide wrapper so a test can run its own
    initialize/shutdown cycles, then put it back.

    The providers are process-lifetime by design and are deliberately NOT
    saved/restored — sharing them across cycles is the behavior under test.
    """
    saved_wrapper = tracing_mod._tracer_wrapper
    saved_options = tracing_mod._session_recording_options
    tracing_mod._tracer_wrapper = None
    tracing_mod._session_recording_options = None
    try:
        yield
    finally:
        reset_tracing()
        tracing_mod._tracer_wrapper = saved_wrapper
        tracing_mod._session_recording_options = saved_options


def _names(exporter):
    return [s.name for s in exporter.get_finished_spans()]


def _boot(exporter):
    init_tracing(
        project_api_key="k",
        disable_batch=True,
        exporter=exporter,
        instruments=set(),
    )


def test_instrument_time_bound_tracer_survives_a_reinit(isolated_tracing):
    """Instrumentors that bind a tracer at `_instrument()` time (MCP,
    pydantic_ai, the traceloop-derived ones) are NOT re-instrumented on a later
    `init_tracing` — `BaseInstrumentor.instrument()` no-ops once instrumented —
    and OTel refuses to override an already-set global TracerProvider. So if
    `shutdown_tracing` shut the provider down and re-init built a new one, those
    instrumentors would emit into a dead provider forever and their spans would
    vanish. The provider is reused across cycles precisely to prevent that.
    """
    exp1 = InMemorySpanExporter()
    _boot(exp1)
    provider = get_tracer_wrapper().tracer_provider

    # Exactly what those instrumentors do inside `_instrument()`.
    bound_tracer = trace.get_tracer("probe", "1", provider)

    with bound_tracer.start_as_current_span("before"):
        pass
    assert _names(exp1) == ["before"]

    shutdown_tracing()

    exp2 = InMemorySpanExporter()
    _boot(exp2)
    assert get_tracer_wrapper().tracer_provider is provider
    assert trace.get_tracer_provider() is provider

    exp1.clear()
    with bound_tracer.start_as_current_span("after"):
        pass

    assert _names(exp2) == ["after"], "tracer bound pre-shutdown went nowhere"
    assert _names(exp1) == [], "retired exporter must not still receive spans"


def test_shutdown_detaches_the_retired_processors(isolated_tracing):
    """The reused provider must not accumulate one shut-down span processor per
    initialize()/shutdown() cycle."""
    exp1 = InMemorySpanExporter()
    _boot(exp1)
    provider = get_tracer_wrapper().tracer_provider
    retired = get_tracer_wrapper().span_processor

    def attached():
        return provider._active_span_processor._span_processors

    # Other processors (e.g. the session fixture's) may share this provider,
    # so assert on the delta rather than an absolute count.
    baseline = len(attached()) - 1
    assert retired in attached()

    shutdown_tracing()
    assert retired not in attached()
    assert len(attached()) == baseline

    for _ in range(3):
        _boot(InMemorySpanExporter())
        current = get_tracer_wrapper().span_processor
        assert len(attached()) == baseline + 1
        shutdown_tracing()
        assert current not in attached()

    assert len(attached()) == baseline

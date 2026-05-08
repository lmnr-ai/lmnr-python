"""Tests for the Laminar <-> Langfuse bridge.

These tests exercise the bridge end-to-end against the real `langfuse` SDK
(pinned to 3.14.6 in dev deps). Langfuse v3 is OTel-native, so the bridge
works by attaching Laminar's `SpanProcessor` (plus our attribute translator)
to Langfuse's `TracerProvider`.

Coverage:
- Auto-enable behaviour in `init_instrumentations` when langfuse is present.
- Attribute translation from `langfuse.*` to `gen_ai.*` / `lmnr.*`.
- Langfuse spans reach Laminar's exporter (dual-export).
- `Laminar.connect_to_langfuse()` manual helper path.
- `_LANGFUSE_PROVIDER_CONFLICTS` strips overlapping raw instrumentors.
- Deepagents wins over Langfuse when both are installed.
"""

from __future__ import annotations

import json
import logging
import os
from unittest.mock import MagicMock

import pytest

# Silence Langfuse's OTLP exporter errors during tests — it will try to POST
# to a real endpoint on flush and fail loudly otherwise.
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-test")
os.environ.setdefault("LANGFUSE_HOST", "http://127.0.0.1:1")
logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.exporter.otlp.proto.http").setLevel(
    logging.CRITICAL)

from lmnr import Laminar  # noqa: E402
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse import (  # noqa: E402
    LangfuseAttributeTranslator,
    LangfuseInstrumentor,
)
from lmnr.opentelemetry_lib.tracing import TracerWrapper, instruments as instruments_mod  # noqa: E402
from lmnr.opentelemetry_lib.tracing.attributes import (  # noqa: E402
    ASSOCIATION_PROPERTIES,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_TYPE,
)
from lmnr.opentelemetry_lib.tracing.instruments import (  # noqa: E402
    INSTRUMENTATION_INITIALIZERS,
    Instruments,
    _LANGFUSE_PROVIDER_CONFLICTS,
    init_instrumentations,
)


# ---------------------------------------------------------------------------
# init_instrumentations auto-enable tests (no real langfuse interaction)
# ---------------------------------------------------------------------------


@pytest.fixture
def track_initializers(monkeypatch):
    """Patch every initializer in the map to record which were invoked."""
    called: set[Instruments] = set()
    replacements: dict[Instruments, object] = {}

    for instrument, initializer in INSTRUMENTATION_INITIALIZERS.items():
        fake = MagicMock(spec=initializer)
        fake.init_instrumentor = MagicMock(
            side_effect=lambda *_a, _inst=instrument, **_kw: called.add(_inst)
            or None
        )
        replacements[instrument] = fake

    monkeypatch.setattr(
        instruments_mod,
        "INSTRUMENTATION_INITIALIZERS",
        replacements,
    )
    return called


@pytest.fixture
def langfuse_installed(monkeypatch):
    monkeypatch.setattr(instruments_mod, "_langfuse_installed", lambda: True)


@pytest.fixture
def langfuse_not_installed(monkeypatch):
    monkeypatch.setattr(instruments_mod, "_langfuse_installed", lambda: False)


@pytest.fixture
def deepagents_installed(monkeypatch):
    monkeypatch.setattr(instruments_mod, "_deepagents_installed", lambda: True)


@pytest.fixture
def deepagents_not_installed(monkeypatch):
    monkeypatch.setattr(
        instruments_mod,
        "_deepagents_installed",
        lambda: False)


@pytest.fixture
def pydantic_ai_not_installed(monkeypatch):
    monkeypatch.setattr(
        instruments_mod,
        "_pydantic_ai_installed",
        lambda: False)


def test_langfuse_not_installed_defaults_exclude_it(
    track_initializers, langfuse_not_installed, pydantic_ai_not_installed
):
    """When langfuse is absent, LANGFUSE is not in the default set and all
    overlapping raw-provider instrumentors remain enabled."""
    init_instrumentations(tracer_provider=MagicMock(), instruments=None)
    assert Instruments.LANGFUSE not in track_initializers
    for instrument in _LANGFUSE_PROVIDER_CONFLICTS:
        assert instrument in track_initializers, (
            f"{instrument} should remain when langfuse isn't installed"
        )


def test_langfuse_initializer_skips_on_unreadable_or_invalid_version(
        monkeypatch):
    """Regression: the initializer used to pass version guards when
    `get_package_version` returned `None` (the check
    `if version and parse(version) < parse("3.0.0")` short-circuits on None),
    silently installing the bridge. For an explicit
    `instruments={Instruments.LANGFUSE}` call this would flip `_installed=True`
    and block any later valid install. Same story for an unparseable version
    string. Both must return `None` from the initializer."""
    from lmnr.opentelemetry_lib.tracing import _instrument_initializers

    monkeypatch.setattr(
        _instrument_initializers,
        "is_package_installed",
        lambda name: True if name == "langfuse" else False,
    )

    # Unreadable version.
    monkeypatch.setattr(
        _instrument_initializers,
        "get_package_version",
        lambda name: None,
    )
    initializer = _instrument_initializers.LangfuseInstrumentorInitializer()
    assert initializer.init_instrumentor() is None

    # Unparseable version string.
    monkeypatch.setattr(
        _instrument_initializers,
        "get_package_version",
        lambda name: "not-a-version",
    )
    assert initializer.init_instrumentor() is None

    # Happy path — parseable, >= 3.0 → instrumentor returned.
    monkeypatch.setattr(
        _instrument_initializers,
        "get_package_version",
        lambda name: "3.14.6",
    )
    assert initializer.init_instrumentor() is not None


def test_langfuse_v2_reports_not_installed(monkeypatch):
    """langfuse 2.x is not OTel-native, so the bridge initializer returns None.
    `_langfuse_installed()` must report False in that case; otherwise the
    provider-conflict set strips OPENAI/ANTHROPIC/... with no replacement."""
    monkeypatch.setattr(
        instruments_mod,
        "is_package_installed",
        lambda name: True if name == "langfuse" else False,
    )
    monkeypatch.setattr(
        instruments_mod,
        "get_package_version",
        lambda name: "2.60.0" if name == "langfuse" else None,
    )
    assert instruments_mod._langfuse_installed() is False

    monkeypatch.setattr(
        instruments_mod,
        "get_package_version",
        lambda name: "3.0.0" if name == "langfuse" else None,
    )
    assert instruments_mod._langfuse_installed() is True


def test_langfuse_installed_auto_enables_and_strips_providers(
    track_initializers, langfuse_installed, pydantic_ai_not_installed, deepagents_not_installed
):
    """When langfuse is installed, LANGFUSE auto-enables and conflicting
    raw-provider / langchain instrumentors are auto-removed."""
    init_instrumentations(
        tracer_provider=MagicMock(),
        instruments=None,
        lmnr_span_processor=MagicMock(),
    )
    assert Instruments.LANGFUSE in track_initializers
    for instrument in _LANGFUSE_PROVIDER_CONFLICTS:
        assert instrument not in track_initializers, (
            f"{instrument} should be auto-removed to avoid duplicate spans"
        )


def test_block_langfuse_disables_auto_logic(
    track_initializers, langfuse_installed, pydantic_ai_not_installed, deepagents_not_installed
):
    """Blocking LANGFUSE suppresses both the auto-enable and the auto-removal."""
    init_instrumentations(
        tracer_provider=MagicMock(),
        instruments=None,
        block_instruments={Instruments.LANGFUSE},
    )
    assert Instruments.LANGFUSE not in track_initializers
    for instrument in _LANGFUSE_PROVIDER_CONFLICTS:
        assert instrument in track_initializers


def test_deepagents_wins_over_langfuse(
    track_initializers, langfuse_installed, deepagents_installed, pydantic_ai_not_installed
):
    """Deepagents wins over Langfuse (same reasoning as pydantic_ai): the
    deepagents instrumentation relies on raw-provider instrumentors to emit
    LLM spans inside each tool call."""
    init_instrumentations(
        tracer_provider=MagicMock(),
        instruments=None,
        lmnr_span_processor=MagicMock(),
    )
    assert Instruments.DEEPAGENTS in track_initializers
    assert Instruments.LANGFUSE in track_initializers
    # Raw provider instrumentors overlap between Langfuse and deepagents-needed
    # ones. Deepagents wins: these must stay enabled.
    provider_only_conflicts = _LANGFUSE_PROVIDER_CONFLICTS - {
        Instruments.LANGCHAIN  # LANGCHAIN is deepagents noise, always removed
    }
    for instrument in provider_only_conflicts:
        assert instrument in track_initializers, (
            f"{instrument} must stay on when deepagents is present"
        )


# ---------------------------------------------------------------------------
# Attribute translator unit tests (direct — no langfuse import needed)
# ---------------------------------------------------------------------------


class _FakeSpan:
    """Minimal stand-in for `opentelemetry.sdk.trace.Span` + `ReadableSpan`."""

    def __init__(self, attributes: dict, scope_name: str = "langfuse-sdk"):
        self._attributes = dict(attributes)
        scope = MagicMock()
        scope.name = scope_name
        self.instrumentation_scope = scope

    @property
    def attributes(self):
        return self._attributes

    def set_attribute(self, key, value):
        self._attributes[key] = value


def test_translator_ignores_non_langfuse_spans():
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan({"some.attr": "x"}, scope_name="openai")
    translator.on_end(span)
    assert span.attributes == {"some.attr": "x"}


def test_translator_maps_generation_to_llm_span():
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan({
        "langfuse.observation.type": "generation",
        "langfuse.observation.model.name": "gpt-4o",
        "langfuse.observation.usage_details": json.dumps(
            {"input": 10, "output": 20, "total": 30}
        ),
        "langfuse.observation.cost_details": json.dumps(
            {"input": 0.001, "output": 0.002, "total": 0.003}
        ),
        "langfuse.observation.input": '{"prompt": "hi"}',
        "langfuse.observation.output": '{"text": "hello"}',
    })
    translator.on_end(span)

    assert span.attributes[SPAN_TYPE] == "LLM"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert span.attributes["gen_ai.response.model"] == "gpt-4o"
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 20
    assert span.attributes["llm.usage.total_tokens"] == 30
    assert span.attributes["gen_ai.usage.input_cost"] == 0.001
    assert span.attributes["gen_ai.usage.output_cost"] == 0.002
    assert span.attributes["gen_ai.usage.cost"] == 0.003
    assert span.attributes[SPAN_INPUT] == '{"prompt": "hi"}'
    assert span.attributes[SPAN_OUTPUT] == '{"text": "hello"}'


def test_translator_maps_tool_observation():
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan({"langfuse.observation.type": "tool"})
    translator.on_end(span)
    assert span.attributes[SPAN_TYPE] == "TOOL"


def test_translator_promotes_trace_session_user_tags_metadata():
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan({
        "session.id": "s456",
        "user.id": "u123",
        "langfuse.trace.tags": ("t1", "t2"),
        "langfuse.trace.metadata.mk": "mv",
        "langfuse.observation.type": "span",
    })
    translator.on_end(span)
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.session_id"] == "s456"
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.user_id"] == "u123"
    assert span.attributes[f"{ASSOCIATION_PROPERTIES}.tags"] == ["t1", "t2"]
    assert (
        span.attributes[f"{ASSOCIATION_PROPERTIES}.metadata.mk"] == "mv"
    )


def test_translator_infers_total_tokens_when_absent():
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan({
        "langfuse.observation.type": "generation",
        "langfuse.observation.usage_details": json.dumps({"input": 7, "output": 3}),
    })
    translator.on_end(span)
    assert span.attributes["llm.usage.total_tokens"] == 10


def test_translator_handles_dict_usage_without_json_wrap():
    """Langfuse sometimes passes a dict rather than a JSON string."""
    translator = LangfuseAttributeTranslator()
    span = _FakeSpan({
        "langfuse.observation.type": "generation",
        # Directly a dict (not JSON-encoded) — the translator should still
        # handle it, but OTel's attribute type-check will have flattened it by
        # the time on_end runs. We keep the helper resilient.
        "langfuse.observation.usage_details": {"input": 1, "output": 2},
    })
    translator.on_end(span)
    assert span.attributes["gen_ai.usage.input_tokens"] == 1
    assert span.attributes["gen_ai.usage.output_tokens"] == 2


# ---------------------------------------------------------------------------
# End-to-end integration tests with real langfuse SDK
# ---------------------------------------------------------------------------


@pytest.fixture
def langfuse_client():
    """Fresh Langfuse client with an isolated resource manager.

    Langfuse caches instances in a module-level singleton keyed by
    public_key. We reset the cache each test so every test gets its own
    TracerProvider (otherwise earlier tests' bridges stack up and confuse the
    assertions).
    """
    # Delay import so tests that don't need langfuse can still be collected.
    from langfuse._client.resource_manager import LangfuseResourceManager

    # Clear the singleton cache so we get a fresh TracerProvider.
    LangfuseResourceManager._instances.clear()

    from langfuse import Langfuse

    client = Langfuse()
    yield client
    try:
        client.shutdown()
    except Exception:
        pass
    LangfuseResourceManager._instances.clear()


def test_connect_to_langfuse_dual_exports_observation(
    span_exporter, langfuse_client
):
    """After `connect_to_langfuse`, Langfuse `@observe` spans reach Laminar's
    in-memory exporter with translated attributes."""
    import contextlib

    assert Laminar.connect_to_langfuse() is True

    from langfuse import observe

    @observe
    def compute(x: int) -> int:
        return x * 2

    compute(21)

    # Langfuse's flush will fail against the fake host; silence its stderr.
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        langfuse_client.flush()

    TracerWrapper.instance.flush()

    spans = span_exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "compute" in names, f"langfuse span did not reach laminar exporter: {names}"
    span = next(s for s in spans if s.name == "compute")
    # Langfuse emits `langfuse.observation.input/output` with JSON encoding;
    # the translator rewrites them to lmnr.span.input / lmnr.span.output.
    assert SPAN_INPUT in span.attributes
    assert SPAN_OUTPUT in span.attributes
    assert "21" in span.attributes[SPAN_INPUT]
    assert span.attributes[SPAN_OUTPUT] == "42"


def test_connect_to_langfuse_translates_generation_attributes(
    span_exporter, langfuse_client
):
    import contextlib

    assert Laminar.connect_to_langfuse() is True

    with langfuse_client.start_as_current_observation(
        name="my-llm",
        as_type="generation",
        model="gpt-4o",
        input={"prompt": "hi"},
    ) as gen:
        gen.update(
            output={"text": "hello"},
            usage_details={"input": 10, "output": 20, "total": 30},
            cost_details={"input": 0.001, "output": 0.002, "total": 0.003},
        )

    with contextlib.redirect_stderr(open(os.devnull, "w")):
        langfuse_client.flush()
    TracerWrapper.instance.flush()

    span = next(
        s for s in span_exporter.get_finished_spans() if s.name == "my-llm"
    )
    assert span.attributes[SPAN_TYPE] == "LLM"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 20
    assert span.attributes["llm.usage.total_tokens"] == 30
    assert span.attributes["gen_ai.usage.input_cost"] == 0.001
    assert span.attributes["gen_ai.usage.output_cost"] == 0.002
    assert span.attributes["gen_ai.usage.cost"] == 0.003


def test_connect_to_langfuse_promotes_trace_session_and_user(
    span_exporter, langfuse_client
):
    import contextlib

    assert Laminar.connect_to_langfuse() is True

    with langfuse_client.start_as_current_span(name="root") as root:
        root.update_trace(
            user_id="user-42",
            session_id="session-7",
            tags=["env:test"],
            metadata={"feature": "beta"},
        )

    with contextlib.redirect_stderr(open(os.devnull, "w")):
        langfuse_client.flush()
    TracerWrapper.instance.flush()

    root = next(
        s for s in span_exporter.get_finished_spans() if s.name == "root"
    )
    assert root.attributes[f"{ASSOCIATION_PROPERTIES}.session_id"] == "session-7"
    assert root.attributes[f"{ASSOCIATION_PROPERTIES}.user_id"] == "user-42"
    tags = root.attributes[f"{ASSOCIATION_PROPERTIES}.tags"]
    assert "env:test" in tags
    assert (
        root.attributes[f"{ASSOCIATION_PROPERTIES}.metadata.feature"] == "beta"
    )


def test_connect_to_langfuse_is_idempotent(span_exporter, langfuse_client):
    """Calling the bridge twice must not duplicate spans on the Laminar side."""
    import contextlib

    assert Laminar.connect_to_langfuse() is True
    assert Laminar.connect_to_langfuse() is True  # second call: no-op

    from langfuse import observe

    @observe
    def once() -> str:
        return "done"

    once()
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        langfuse_client.flush()
    TracerWrapper.instance.flush()

    spans = [s for s in span_exporter.get_finished_spans() if s.name == "once"]
    assert len(spans) == 1, f"expected exactly one span, got {len(spans)}"


def test_translator_mutates_before_synchronous_exporter():
    """Regression: under `disable_batch=True` the Laminar exporter uses
    `SimpleSpanProcessor` and exports inside `on_end`. The translator MUST run
    first, otherwise the exporter ships the pre-translation `langfuse.*`
    shape. We verify this by installing a fake SimpleSpanProcessor-backed
    exporter on the same provider and confirming it sees translated attrs.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )

    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse import (
        _prepend_span_processor,
    )

    exported: list[dict] = []

    class CaptureExporter(SpanExporter):
        def export(self, spans):
            for s in spans:
                exported.append(dict(s.attributes or {}))
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(CaptureExporter()))

    # Install the translator AFTER the exporter (the real-world order) but
    # via `_prepend_span_processor`, which reorders it to the front.
    translator = LangfuseAttributeTranslator()
    assert _prepend_span_processor(provider, translator) is True

    tracer = provider.get_tracer("langfuse-sdk")
    span = tracer.start_span(
        "llm-call",
        attributes={"langfuse.observation.model.name": "gpt-4o"},
    )
    span.end()

    assert len(exported) == 1
    assert (
        exported[0].get("gen_ai.request.model") == "gpt-4o"
    ), f"translator must run before exporter, got {exported[0]}"


def test_connect_to_langfuse_swallows_install_exceptions(monkeypatch):
    """Regression: `LangfuseInstrumentor.instrument()` re-raises on
    attach-phase failures (e.g. `RuntimeError` from concurrent modification
    of `LangfuseResourceManager._instances`). `Laminar.connect_to_langfuse()`
    documents a `bool` return and callers should not have to wrap it in a
    try/except — so the helper must swallow the exception and return
    `False` on failure."""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation import langfuse as lf

    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._handled_providers = set()
    LangfuseInstrumentor._attached_providers = {}
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._lmnr_span_processor = None
    LangfuseInstrumentor._lmnr_tracer_provider = None
    LangfuseInstrumentor._original_initialize_instance = None

    def exploding_attach(self):  # noqa: ARG001
        raise RuntimeError("simulated attach failure")

    monkeypatch.setattr(
        lf.LangfuseInstrumentor,
        "_attach_to_existing_langfuse_providers",
        exploding_attach,
    )

    # Must not raise.
    assert Laminar.connect_to_langfuse() is False
    # `instrument()`'s rollback path must have cleaned up, leaving
    # `_installed=False`.
    assert LangfuseInstrumentor._installed is False


def test_connect_to_langfuse_before_initialize_does_not_crash(monkeypatch):
    """Regression: `connect_to_langfuse()` must return `False` (not raise)
    when called before `Laminar.initialize()`. The not-initialized branch
    previously accessed the private name-mangled `cls.__logger`, which is
    only set by `_initialize_logger()` during `initialize()`. Any caller
    that probed for Langfuse support before initializing Laminar would hit
    an `AttributeError` instead of the documented False return."""
    # Force the "not initialized" state: both the `__initialized` flag and
    # (crucially) the `__logger` attribute absent — the latter is what the
    # original bug stumbled on.
    monkeypatch.setattr(Laminar, "_Laminar__initialized", False)
    if hasattr(Laminar, "_Laminar__logger"):
        monkeypatch.delattr(Laminar, "_Laminar__logger")

    # Must not raise.
    assert Laminar.connect_to_langfuse() is False


def test_connect_to_langfuse_returns_false_on_install_failure(monkeypatch):
    """If `instrument()` bails out before setting `_installed=True` (e.g.
    translator install raises), the public helper must surface the failure
    as `False` — not claim success."""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation import langfuse as lf

    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._original_initialize_instance = None

    def failing_prepend(provider, processor):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr(lf, "_prepend_span_processor", failing_prepend)

    assert Laminar.connect_to_langfuse() is False
    assert LangfuseInstrumentor._installed is False


def test_connect_to_langfuse_returns_false_without_langfuse(monkeypatch):
    """If `langfuse` isn't importable (or is too old), the helper must return
    False and must NOT install the bridge (i.e. no translator added, no
    monkey-patch). The version-aware `_langfuse_installed` check is what
    guards this — see the companion 2.x-specific test below."""
    monkeypatch.setattr(
        instruments_mod, "_langfuse_installed", lambda: False,
    )

    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._original_initialize_instance = None

    assert Laminar.connect_to_langfuse() is False
    assert LangfuseInstrumentor._installed is False
    assert LangfuseInstrumentor._translator is None


def test_connect_to_langfuse_rejects_langfuse_v2(monkeypatch):
    """Regression: `connect_to_langfuse()` must also version-gate on
    langfuse >= 3.0. With 2.x installed, the bridge initializer returns
    None, so installing it would attach a useless translator and
    permanently flip `_installed=True` (blocking a later valid install)."""
    monkeypatch.setattr(
        instruments_mod,
        "is_package_installed",
        lambda name: True if name == "langfuse" else False,
    )
    monkeypatch.setattr(
        instruments_mod,
        "get_package_version",
        lambda name: "2.60.0" if name == "langfuse" else None,
    )

    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._original_initialize_instance = None

    assert Laminar.connect_to_langfuse() is False
    assert LangfuseInstrumentor._installed is False
    assert LangfuseInstrumentor._translator is None


def test_late_attach_patches_future_langfuse_clients(span_exporter):
    """The resource-manager patch means Langfuse clients created AFTER the
    bridge is installed still get dual-attached."""
    import contextlib

    from langfuse._client.resource_manager import LangfuseResourceManager

    # Install bridge with NO langfuse client yet.
    LangfuseResourceManager._instances.clear()
    assert Laminar.connect_to_langfuse() is True

    # Now create a Langfuse client — this triggers
    # `LangfuseResourceManager._initialize_instance`, which our bridge
    # has monkey-patched to attach Laminar's processor.
    from langfuse import Langfuse, observe

    client = Langfuse()
    try:
        @observe
        def late() -> int:
            return 7

        late()
        with contextlib.redirect_stderr(open(os.devnull, "w")):
            client.flush()
    finally:
        client.shutdown()
        LangfuseResourceManager._instances.clear()

    TracerWrapper.instance.flush()
    names = [s.name for s in span_exporter.get_finished_spans()]
    assert "late" in names, f"late-created Langfuse client was not bridged: {names}"


def test_instrumentor_skips_laminar_own_provider(span_exporter):
    """Guardrail: if Langfuse somehow ends up sharing Laminar's
    TracerProvider, `_attach_to_provider` must skip it (otherwise Laminar's
    processor would be attached to Laminar's provider twice → duplicate
    spans on every Laminar export).
    """
    instrumentor = LangfuseInstrumentor()
    wrapper = TracerWrapper.instance
    initial_count = len(
        wrapper._tracer_provider._active_span_processor._span_processors
    )
    instrumentor._lmnr_span_processor = wrapper._span_processor
    instrumentor._attach_to_provider(wrapper._tracer_provider)
    after_count = len(
        wrapper._tracer_provider._active_span_processor._span_processors
    )
    assert after_count == initial_count, (
        "attached to Laminar's own provider — would cause duplicate exports"
    )


def test_instrument_rolls_back_translator_if_attach_phase_raises(span_exporter):
    """Regression: if `_attach_to_existing_langfuse_providers` or
    `_patch_resource_manager` raises, the translator that was already
    prepended to Laminar's provider in step 1 must be removed and all
    class-level state cleared. Otherwise, a subsequent `instrument()` (e.g.
    via `Laminar.connect_to_langfuse()`) would pass the `_installed` guard
    and prepend a SECOND translator, double-translating every Langfuse span.
    """
    from opentelemetry.sdk.trace import TracerProvider

    # Clean state.
    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._handled_providers = set()
    LangfuseInstrumentor._attached_providers = {}
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._lmnr_span_processor = None
    LangfuseInstrumentor._lmnr_tracer_provider = None

    provider = TracerProvider()

    def count_translators() -> int:
        return sum(
            1
            for p in provider._active_span_processor._span_processors
            if isinstance(p, LangfuseAttributeTranslator)
        )

    baseline = count_translators()

    instrumentor = LangfuseInstrumentor()

    # Force `_attach_to_existing_langfuse_providers` to blow up on first
    # install.
    def exploding_attach(self):  # noqa: ARG001
        raise RuntimeError("simulated attach failure")

    original_attach = (
        LangfuseInstrumentor._attach_to_existing_langfuse_providers
    )
    LangfuseInstrumentor._attach_to_existing_langfuse_providers = (
        exploding_attach
    )
    try:
        with pytest.raises(RuntimeError, match="simulated attach failure"):
            instrumentor.instrument(
                lmnr_tracer_provider=provider,
                lmnr_span_processor=MagicMock(),
            )
    finally:
        LangfuseInstrumentor._attach_to_existing_langfuse_providers = (
            original_attach
        )

    # Translator must have been rolled back.
    assert count_translators() == baseline, (
        "translator must be removed on partial install failure"
    )
    assert LangfuseInstrumentor._installed is False
    assert LangfuseInstrumentor._translator is None

    # A subsequent successful install must attach exactly ONE translator —
    # not two, which is what would happen if the failed-install translator
    # were still around.
    instrumentor2 = LangfuseInstrumentor()
    instrumentor2.instrument(
        lmnr_tracer_provider=provider,
        lmnr_span_processor=MagicMock(),
    )
    assert count_translators() == baseline + 1
    instrumentor2.uninstrument()


def test_instrument_skips_shared_laminar_provider_without_tracerwrapper(span_exporter):
    """Regression: during auto-install via `init_instrumentations`,
    `TracerWrapper.instance` is assigned AFTER `init_instrumentations`
    returns. If a pre-existing Langfuse client happens to share Laminar's
    newly-created `TracerProvider`, `_attach_to_provider`'s
    `TracerWrapper.verify_initialized()` fallback guard returns False
    (the wrapper isn't set yet) and the translator + Laminar span processor
    would be double-attached.

    `instrument()` must pre-register `id(lmnr_tracer_provider)` in
    `_handled_providers` so the short-circuit works independently of the
    TracerWrapper lifecycle.
    """
    from opentelemetry.sdk.trace import TracerProvider

    # Start from a clean slate.
    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._handled_providers = set()
    LangfuseInstrumentor._attached_providers = {}
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._lmnr_span_processor = None
    LangfuseInstrumentor._lmnr_tracer_provider = None

    # A fresh provider standing in for Laminar's. We deliberately do NOT
    # reassign `TracerWrapper.instance` to point at this provider — that's
    # exactly the case the guard has to cover during auto-install.
    shared_provider = TracerProvider()

    mock_processor = MagicMock()
    instrumentor = LangfuseInstrumentor()
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
        1 for p in processors_after_install
        if isinstance(p, LangfuseAttributeTranslator)
    )
    assert translator_count == 1
    assert mock_processor not in processors_after_install, (
        "Laminar span processor should NOT be attached to Laminar's own "
        "provider — the translator is enough there"
    )

    # Now simulate Langfuse's resource manager calling `_attach_to_provider`
    # with the same provider. The pre-registered id must prevent a
    # double-attach.
    instrumentor._attach_to_provider(shared_provider)

    processors_after_simulated_langfuse = list(
        shared_provider._active_span_processor._span_processors
    )
    translator_count_after = sum(
        1 for p in processors_after_simulated_langfuse
        if isinstance(p, LangfuseAttributeTranslator)
    )
    assert translator_count_after == 1, (
        "translator should not be attached twice to Laminar's provider"
    )
    assert mock_processor not in processors_after_simulated_langfuse, (
        "Laminar span processor should not be appended to Laminar's own "
        "provider via the Langfuse attach path"
    )

    instrumentor.uninstrument()


def test_uninstrument_removes_translator_and_clears_state(span_exporter):
    """Regression: `uninstrument` must detach the translator from Laminar's
    provider and clear class-level state (`_handled_providers`, `_translator`,
    `_lmnr_span_processor`, `_attached_providers`, `_lmnr_tracer_provider`).
    Otherwise a subsequent `instrument()` call would prepend a SECOND
    translator onto Laminar's provider (the first was never removed), and
    `_handled_providers` would still hold ids from the previous session,
    making `_attach_to_existing_langfuse_providers` skip providers it saw
    last time around.
    """
    from langfuse._client.resource_manager import LangfuseResourceManager

    LangfuseResourceManager._instances.clear()
    wrapper = TracerWrapper.instance
    lmnr_provider = wrapper._tracer_provider

    def count_translators() -> int:
        return sum(
            1
            for p in lmnr_provider._active_span_processor._span_processors
            if isinstance(p, LangfuseAttributeTranslator)
        )

    # Start from a clean slate in case an earlier test left state behind.
    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._handled_providers = set()
    LangfuseInstrumentor._attached_providers = {}
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._lmnr_span_processor = None
    LangfuseInstrumentor._lmnr_tracer_provider = None

    baseline_translators = count_translators()

    instrumentor = LangfuseInstrumentor()
    instrumentor.instrument(
        lmnr_tracer_provider=lmnr_provider,
        lmnr_span_processor=wrapper._span_processor,
    )
    assert LangfuseInstrumentor._installed is True
    assert count_translators() == baseline_translators + 1

    instrumentor.uninstrument()

    # Translator must have been removed from Laminar's provider.
    assert count_translators() == baseline_translators
    # All class-level state must be cleared so a fresh install starts clean.
    assert LangfuseInstrumentor._installed is False
    assert LangfuseInstrumentor._translator is None
    assert LangfuseInstrumentor._lmnr_span_processor is None
    assert LangfuseInstrumentor._lmnr_tracer_provider is None
    assert LangfuseInstrumentor._handled_providers == set()
    assert LangfuseInstrumentor._attached_providers == {}

    # Re-install: the translator count should increase by exactly one again —
    # NOT two, which is what would happen if the previous translator was
    # still attached.
    instrumentor2 = LangfuseInstrumentor()
    instrumentor2.instrument(
        lmnr_tracer_provider=lmnr_provider,
        lmnr_span_processor=wrapper._span_processor,
    )
    assert count_translators() == baseline_translators + 1
    instrumentor2.uninstrument()
    assert count_translators() == baseline_translators


def test_uninstrument_detaches_processors_from_langfuse_providers(span_exporter):
    """Regression: after `uninstrument`, every provider we previously attached
    the translator / Laminar span processor to must lose those processors.
    Re-install must also succeed (stale `_handled_providers` must not
    short-circuit re-attachment).

    We simulate a Langfuse-owned TracerProvider with a plain
    `sdk.trace.TracerProvider`, directed at `_attach_to_provider`: the bridge
    treats any non-Laminar provider the same way, so this exercises the
    install/uninstall/reinstall flow deterministically without depending on
    whether the real Langfuse SDK reuses the global provider.
    """
    from opentelemetry.sdk.trace import TracerProvider

    wrapper = TracerWrapper.instance
    lmnr_provider = wrapper._tracer_provider
    lmnr_processor = wrapper._span_processor

    # Start from a known-clean state.
    LangfuseInstrumentor._installed = False
    LangfuseInstrumentor._handled_providers = set()
    LangfuseInstrumentor._attached_providers = {}
    LangfuseInstrumentor._translator = None
    LangfuseInstrumentor._lmnr_span_processor = None
    LangfuseInstrumentor._lmnr_tracer_provider = None

    fake_lf_provider = TracerProvider()
    baseline = list(fake_lf_provider._active_span_processor._span_processors)

    instrumentor = LangfuseInstrumentor()
    instrumentor.instrument(
        lmnr_tracer_provider=lmnr_provider,
        lmnr_span_processor=lmnr_processor,
    )
    # Manually attach to our fake Langfuse-owned provider the same way the
    # monkey-patched `_initialize_instance` would in production.
    instrumentor._attach_to_provider(fake_lf_provider)

    after_install = list(
        fake_lf_provider._active_span_processor._span_processors
    )
    assert any(
        isinstance(p, LangfuseAttributeTranslator) for p in after_install
    )
    assert lmnr_processor in after_install
    assert id(fake_lf_provider) in LangfuseInstrumentor._attached_providers

    instrumentor.uninstrument()

    after_uninstall = list(
        fake_lf_provider._active_span_processor._span_processors
    )
    assert not any(
        isinstance(p, LangfuseAttributeTranslator) for p in after_uninstall
    ), "translator should be removed from the Langfuse-owned provider"
    assert lmnr_processor not in after_uninstall, (
        "Laminar span processor should be removed from the Langfuse-owned provider"
    )
    # Ordering of unrelated processors should be preserved.
    assert after_uninstall == baseline

    # Re-install must work — stale `_handled_providers` was cleared so the
    # existing provider is seen again.
    instrumentor.instrument(
        lmnr_tracer_provider=lmnr_provider,
        lmnr_span_processor=lmnr_processor,
    )
    instrumentor._attach_to_provider(fake_lf_provider)
    reinstalled = list(
        fake_lf_provider._active_span_processor._span_processors
    )
    assert any(
        isinstance(p, LangfuseAttributeTranslator) for p in reinstalled
    )
    assert lmnr_processor in reinstalled
    instrumentor.uninstrument()

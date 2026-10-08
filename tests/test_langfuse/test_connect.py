"""End-to-end tests for `Laminar.connect_to_langfuse` against the real langfuse SDK."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast
from unittest.mock import MagicMock

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue
from typing_extensions import override

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse import (
    LangfuseAttributeTranslator,
    get_langfuse_instrumentor,
    prepend_span_processor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langfuse.provider_attachment import (
    ProviderAttachment,
)
from lmnr.opentelemetry_lib.tracing import instruments as instruments_mod
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_TYPE,
)

from .utils import (
    reset_langfuse_bridge_state,
)


def test_connect_to_langfuse_dual_exports_observation(span_exporter: InMemorySpanExporter):
    """After `connect_to_langfuse`, Langfuse `@observe` spans reach Laminar's
    in-memory exporter with translated attributes."""
    assert Laminar.connect_to_langfuse() is True

    from langfuse import observe  # pyright: ignore[reportUnknownVariableType]

    @observe
    def compute(x: int) -> int:
        return x * 2

    _result = compute(21)

    spans = span_exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "compute" in names, f"langfuse span did not reach laminar exporter: {names}"
    span = next(s for s in spans if s.name == "compute")
    # Langfuse emits `langfuse.observation.input/output` with JSON encoding;
    # the translator rewrites them to lmnr.span.input / lmnr.span.output.
    assert SPAN_INPUT in (span.attributes or {})
    assert SPAN_OUTPUT in (span.attributes or {})
    assert "21" in cast(str, (span.attributes or {})[SPAN_INPUT])
    assert (span.attributes or {})[SPAN_OUTPUT] == "42"


def test_connect_to_langfuse_translates_generation_attributes(
    span_exporter: InMemorySpanExporter, langfuse_client: Any,
):
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

    span = next(s for s in span_exporter.get_finished_spans() if s.name == "my-llm")
    attributes = span.attributes or {}
    assert attributes[SPAN_TYPE] == "LLM"
    assert attributes["gen_ai.request.model"] == "gpt-4o"
    assert attributes["gen_ai.usage.input_tokens"] == 10
    assert attributes["gen_ai.usage.output_tokens"] == 20
    assert attributes["llm.usage.total_tokens"] == 30
    assert attributes["gen_ai.usage.input_cost"] == 0.001
    assert attributes["gen_ai.usage.output_cost"] == 0.002
    assert attributes["gen_ai.usage.cost"] == 0.003


def test_connect_to_langfuse_splits_openai_generation_input(
    span_exporter: InMemorySpanExporter, langfuse_client: Any,
):
    """End-to-end: a generation whose input is the OpenAI {messages, tools}
    shape (what `langfuse.openai` ships) is split into gen_ai.input.messages +
    gen_ai.tool.definitions on the Laminar side."""
    assert Laminar.connect_to_langfuse() is True

    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    messages = [{"role": "user", "content": "weather in SF?"}]
    with langfuse_client.start_as_current_observation(
        name="openai-gen",
        as_type="generation",
        model="gpt-4o",
        input={"messages": messages, "tools": tools},
    ) as gen:
        gen.update(output={"role": "assistant", "content": "It's sunny"})

    span = next(s for s in span_exporter.get_finished_spans() if s.name == "openai-gen")
    attributes = span.attributes or {}
    assert json.loads(cast(str, attributes["gen_ai.input.messages"])) == messages
    assert json.loads(cast(str, attributes["gen_ai.tool.definitions"])) == tools
    assert SPAN_INPUT not in attributes
    assert json.loads(cast(str, attributes["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "It's sunny"}
    ]


def test_connect_to_langfuse_promotes_trace_session_and_user(
     span_exporter: InMemorySpanExporter, langfuse_client: Any,
):
    assert Laminar.connect_to_langfuse() is True

    from langfuse import propagate_attributes

    with langfuse_client.start_as_current_observation(name="root") as root, propagate_attributes(
        user_id="user-42",
        session_id="session-7",
        tags=["env:test"],
        metadata={"feature": "beta"},
    ):
        pass

    root = next(s for s in span_exporter.get_finished_spans() if s.name == "root")
    attributes = root.attributes or {}
    assert attributes[f"{ASSOCIATION_PROPERTIES}.session_id"] == "session-7"
    assert attributes[f"{ASSOCIATION_PROPERTIES}.user_id"] == "user-42"
    tags = cast(list[str], attributes[f"{ASSOCIATION_PROPERTIES}.tags"])
    assert "env:test" in tags
    assert attributes[f"{ASSOCIATION_PROPERTIES}.metadata.feature"] == "beta"


def test_connect_to_langfuse_is_idempotent(span_exporter: InMemorySpanExporter):
    """Calling the bridge twice must not duplicate spans on the Laminar side."""
    assert Laminar.connect_to_langfuse() is True
    assert Laminar.connect_to_langfuse() is True  # second call: no-op

    from langfuse import observe  # pyright: ignore[reportUnknownVariableType]

    @observe
    def once() -> str:
        return "done"

    _done = once()

    spans = [s for s in span_exporter.get_finished_spans() if s.name == "once"]
    assert len(spans) == 1, f"expected exactly one span, got {len(spans)}"


def test_translator_mutates_before_synchronous_exporter():
    """Regression: under `disable_batch=True` the Laminar exporter uses
    `SimpleSpanProcessor` and exports inside `on_end`. The translator MUST run
    first, otherwise the exporter ships the pre-translation `langfuse.*`
    shape. We verify this by installing a fake SimpleSpanProcessor-backed
    exporter on the same provider and confirming it sees translated attrs.
    """
    exported: list[dict[str, AttributeValue]] = []

    class CaptureExporter(SpanExporter):
        @override
        def export(self, spans: Sequence[ReadableSpan]):
            for s in spans:
                exported.append(dict(s.attributes or {}))
            return SpanExportResult.SUCCESS

        @override
        def shutdown(self):
            pass

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(CaptureExporter()))

    # Install the translator AFTER the exporter (the real-world order) but
    # via `_prepend_span_processor`, which reorders it to the front.
    translator = LangfuseAttributeTranslator()
    assert prepend_span_processor(provider, translator) is True

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


def test_connect_to_langfuse_swallows_install_exceptions(monkeypatch: MagicMock):
    """Regression: `LangfuseInstrumentor.instrument()` re-raises on
    attach-phase failures (e.g. `RuntimeError` from concurrent modification
    of `LangfuseResourceManager._instances`). `Laminar.connect_to_langfuse()`
    documents a `bool` return and callers should not have to wrap it in a
    try/except — so the helper must swallow the exception and return
    `False` on failure."""

    reset_langfuse_bridge_state()

    def exploding_attach(self):
        raise RuntimeError("simulated attach failure")

    monkeypatch.setattr(
        ProviderAttachment,
        "attach_to_existing_langfuse_providers",
        exploding_attach,
    )

    # Must not raise.
    assert Laminar.connect_to_langfuse() is False
    # `instrument()`'s rollback path must have cleaned up, leaving
    # not instrumented.
    assert get_langfuse_instrumentor().is_instrumented_by_opentelemetry is False


def test_connect_to_langfuse_before_initialize_does_not_crash(monkeypatch: MagicMock):
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


def test_connect_to_langfuse_returns_false_on_install_failure(monkeypatch: MagicMock):
    """If `instrument()` bails out before setting `_installed=True` (e.g.
    translator install raises), the public helper must surface the failure
    as `False` — not claim success."""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation import langfuse as lf

    reset_langfuse_bridge_state()

    def failing_prepend(provider: TracerProvider, processor: SpanProcessor):
        raise RuntimeError("boom")

    monkeypatch.setattr(lf, "prepend_span_processor", failing_prepend)

    assert Laminar.connect_to_langfuse() is False
    assert get_langfuse_instrumentor().is_instrumented_by_opentelemetry is False


def test_connect_to_langfuse_returns_false_without_langfuse(monkeypatch: MagicMock):
    """If `langfuse` isn't importable (or is too old), the helper must return
    False and must NOT install the bridge (i.e. no translator added, no
    monkey-patch). The version-aware `langfuse_installed` check is what
    guards this — see the companion 2.x-specific test below."""
    monkeypatch.setattr(
        instruments_mod,
        "langfuse_installed",
        lambda: False,
    )

    reset_langfuse_bridge_state()

    assert Laminar.connect_to_langfuse() is False
    instrumentor = get_langfuse_instrumentor()
    assert instrumentor.is_instrumented_by_opentelemetry is False
    assert instrumentor._translator is None


def test_connect_to_langfuse_returns_false_when_sdk_unimportable(monkeypatch: MagicMock):
    """If `langfuse` is present per metadata but cannot actually be imported
    (the pydantic-v1 failure on Python 3.14), the bridge's resource-manager
    attach/patch path would silently no-op while `instrument()` still flips
    `_installed=True`. `connect_to_langfuse()` must NOT report success there —
    SDK spans (`@observe`, `langfuse.openai`, ...) would never reach Laminar.
    Guard on a real import probe and return False without installing."""
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation import langfuse as lf

    # Metadata says installed and modern, but the SDK can't be imported.
    monkeypatch.setattr(instruments_mod, "langfuse_installed", lambda: True)
    monkeypatch.setattr(lf, "langfuse_sdk_importable", lambda: False)

    reset_langfuse_bridge_state()

    assert Laminar.connect_to_langfuse() is False
    # The bridge must NOT have been installed (no orphaned translator that a
    # later valid install would have to clean up / could stack a second one).
    instrumentor = get_langfuse_instrumentor()
    assert instrumentor.is_instrumented_by_opentelemetry is False
    assert instrumentor._translator is None


def test_connect_to_langfuse_rejects_langfuse_v2(monkeypatch: MagicMock):
    """Regression: `connect_to_langfuse()` must also version-gate on
    langfuse >= 3.0. With 2.x installed, the bridge initializer returns
    None, so installing it would attach a useless translator and
    permanently flip `_installed=True` (blocking a later valid install)."""
    monkeypatch.setattr(
        instruments_mod,
        "is_package_installed",
        lambda name: name == "langfuse",  # pyright: ignore[reportUnknownLambdaType]
    )
    monkeypatch.setattr(
        instruments_mod,
        "get_package_version",
        lambda name: "2.60.0" if name == "langfuse" else None,  # pyright: ignore[reportUnknownLambdaType]
    )

    reset_langfuse_bridge_state()

    assert Laminar.connect_to_langfuse() is False
    instrumentor = get_langfuse_instrumentor()
    assert instrumentor.is_instrumented_by_opentelemetry is False
    assert instrumentor._translator is None


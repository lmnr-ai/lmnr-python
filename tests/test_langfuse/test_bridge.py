"""The Langfuse bridge against a real Langfuse client.

Each test gets its own Laminar span processor (`laminar` fixture) and a
Langfuse client on its own provider whose export is recorded
(`langfuse_export`).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langfuse import Langfuse, observe, propagate_attributes
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.bridges import BridgeContext
from lmnr.opentelemetry_lib.bridges.langfuse import LangfuseBridge
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SPAN_BRIDGE_NAME,
    SPAN_BRIDGE_VERSION,
    SPAN_IDS_PATH,
    SPAN_PATH,
    SPAN_TYPE,
)
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from lmnr.opentelemetry_lib.utils.package_check import get_package_version

from .conftest import Laminar, RecordingExporter, _replace_langfuse_exporter

# Values that differ between two otherwise identical runs.
_VOLATILE = {"langfuse.observation.completion_start_time"}


def _new_client(exporter: RecordingExporter) -> Any:
    LangfuseResourceManager._instances.clear()
    client = Langfuse(tracer_provider=TracerProvider())
    _replace_langfuse_exporter(client, exporter)
    return client


def _run_scenario(client: Any) -> None:
    @observe(as_type="generation")
    def generate(question: str) -> str:
        client.update_current_generation(
            model="gpt-4o",
            usage_details={"input": 9, "input_cached_tokens": 2, "output": 5},
            cost_details={"input": 0.01, "output": 0.02},
            input=[{"role": "user", "content": question}],
            output={"role": "assistant", "content": "hi"},
        )
        return "hi"

    @observe(as_type="tool")
    def lookup(x: int) -> dict[str, int]:
        return {"result": x}

    @observe()
    def fail() -> None:
        raise ValueError("kaboom")

    @observe()
    def agent(question: str) -> str:
        with propagate_attributes(
            user_id="user-1", session_id="session-1", tags=["tag-1"], metadata={"team": "a"}
        ):
            generate(question)
            lookup(1)
            try:
                fail()
            except ValueError:
                pass
        return "done"

    agent("hello")
    client.flush()


def _normalized(exporter: RecordingExporter) -> list[dict[str, Any]]:
    spans = [
        {**s, "attributes": {k: v for k, v in s["attributes"].items() if k not in _VOLATILE}}
        for s in exporter.spans
    ]
    return sorted(spans, key=lambda s: s["name"])


def test_langfuse_export_is_unchanged_by_the_bridge(laminar: Laminar, bridge: LangfuseBridge):
    without_bridge = RecordingExporter()
    client = _new_client(without_bridge)
    _run_scenario(client)
    client.shutdown()

    with_bridge = RecordingExporter()
    client = _new_client(with_bridge)
    assert bridge.install(laminar.context)
    _run_scenario(client)
    client.shutdown()

    assert laminar.spans(), "the bridge must forward spans to Laminar"
    assert _normalized(with_bridge) == _normalized(without_bridge)
    for span in with_bridge.spans:
        assert not any(k.startswith(("lmnr.", "gen_ai.")) for k in span["attributes"])
    # Only Laminar's copy drops the converted `langfuse.*` attributes.
    exported = next(s for s in with_bridge.spans if s["name"] == "generate")["attributes"]
    assert "langfuse.observation.input" in exported
    assert "langfuse.observation.usage_details" in exported


def test_bridged_spans_are_translated(
    laminar: Laminar, bridge: LangfuseBridge, langfuse_client: Any
):
    assert bridge.install(laminar.context)
    _run_scenario(langfuse_client)
    spans = laminar.spans()

    generation = dict(spans["generate"].attributes or {})
    assert generation[SPAN_TYPE] == "LLM"
    assert generation["gen_ai.request.model"] == "gpt-4o"
    # Langfuse's `input` excludes cached tokens; Laminar's input count includes them.
    assert generation["gen_ai.usage.input_tokens"] == 11
    assert generation["gen_ai.usage.cache_read.input_tokens"] == 2
    assert generation["gen_ai.usage.output_tokens"] == 5
    assert generation["gen_ai.usage.cost"] == pytest.approx(0.03)
    assert json.loads(str(generation["gen_ai.input.messages"])) == [
        {"role": "user", "content": "hello"}
    ]
    assert generation[f"{ASSOCIATION_PROPERTIES}.user_id"] == "user-1"
    assert generation[f"{ASSOCIATION_PROPERTIES}.session_id"] == "session-1"
    assert list(generation[f"{ASSOCIATION_PROPERTIES}.tags"]) == ["tag-1"]  # pyright: ignore[reportArgumentType]
    assert generation[f"{ASSOCIATION_PROPERTIES}.metadata.team"] == "a"
    assert generation[SPAN_BRIDGE_NAME] == "langfuse"
    # Converted attributes aren't stored twice.
    for key in (
        "langfuse.observation.input",
        "langfuse.observation.output",
        "langfuse.observation.usage_details",
        "langfuse.observation.cost_details",
        "langfuse.observation.model.name",
    ):
        assert key not in generation, key
    assert generation[SPAN_BRIDGE_VERSION] == get_package_version("langfuse")

    assert (spans["lookup"].attributes or {})[SPAN_TYPE] == "TOOL"
    assert SPAN_TYPE not in (spans["agent"].attributes or {})

    failed = spans["fail"]
    exception = [e for e in failed.events if e.name == "exception"]
    assert len(exception) == 1
    assert "kaboom" in str((exception[0].attributes or {})["exception.message"])

    # Laminar's own path bookkeeping nests the bridged spans.
    assert list(generation[SPAN_PATH]) == ["agent", "generate"]  # pyright: ignore[reportArgumentType]
    assert len(list(generation[SPAN_IDS_PATH])) == 2  # pyright: ignore[reportArgumentType]


def test_langfuse_openai_metadata_keys_reach_laminar(
    laminar: Laminar, bridge: LangfuseBridge, langfuse_client: Any
):
    """What `langfuse.openai` records for `metadata={"langfuse_session_id": ...}`:
    plain observation metadata on a root generation."""
    assert bridge.install(laminar.context)
    with langfuse_client.start_as_current_observation(
        name="OpenAI-generation",
        as_type="generation",
        metadata={
            "langfuse_session_id": "session-2",
            "langfuse_user_id": "user-2",
            "langfuse_tags": ["tag-2"],
            "customer": "acme",
        },
    ):
        pass
    langfuse_client.flush()

    generation = dict(laminar.spans()["OpenAI-generation"].attributes or {})
    assert generation[f"{ASSOCIATION_PROPERTIES}.session_id"] == "session-2"
    assert generation[f"{ASSOCIATION_PROPERTIES}.user_id"] == "user-2"
    assert list(generation[f"{ASSOCIATION_PROPERTIES}.tags"]) == ["tag-2"]  # pyright: ignore[reportArgumentType]
    # A root observation's metadata is trace metadata in Langfuse.
    assert generation[f"{ASSOCIATION_PROPERTIES}.metadata.customer"] == "acme"
    assert f"{ASSOCIATION_PROPERTIES}.metadata.langfuse_session_id" not in generation


def test_clients_created_after_install_are_bridged(laminar: Laminar, bridge: LangfuseBridge):
    assert bridge.install(laminar.context)
    client = _new_client(RecordingExporter())
    _run_scenario(client)
    client.shutdown()
    assert "generate" in laminar.spans()


def test_uninstall_stops_forwarding(laminar: Laminar, bridge: LangfuseBridge, langfuse_client: Any):
    original_init = LangfuseResourceManager._initialize_instance
    assert bridge.install(laminar.context)
    assert LangfuseResourceManager._initialize_instance is not original_init
    bridge.uninstall()
    assert LangfuseResourceManager._initialize_instance is original_init

    _run_scenario(langfuse_client)
    assert laminar.spans() == {}


def test_only_spans_langfuse_exports_are_forwarded(
    laminar: Laminar,
    bridge: LangfuseBridge,
    langfuse_client: Any,
    langfuse_export: RecordingExporter,
):
    """Spans from other instrumentations on Langfuse's provider reach Laminar
    exactly when Langfuse exports them (its scope filter, langfuse >= 4)."""
    assert bridge.install(laminar.context)
    provider = langfuse_client._resources.tracer_provider
    with provider.get_tracer("some.http.library").start_as_current_span("http-call"):
        pass
    langfuse_client.flush()

    exported_by_langfuse = any(s["name"] == "http-call" for s in langfuse_export.spans)
    assert ("http-call" in laminar.spans()) == exported_by_langfuse


def test_langfuse_flush_flushes_laminars_batch(
    bridge: LangfuseBridge, langfuse_client: Any, monkeypatch: pytest.MonkeyPatch
):
    # A batching processor with a long delay: copies stay queued until flushed.
    monkeypatch.setenv("OTEL_BSP_SCHEDULE_DELAY", "600000")
    exporter = InMemorySpanExporter()
    processor = LaminarSpanProcessor(api_key="test", exporter=exporter)
    try:
        provider = TracerProvider()
        provider.add_span_processor(processor)
        assert bridge.install(BridgeContext(span_processor=processor, tracer_provider=provider))
        _run_scenario(langfuse_client)  # ends with `client.flush()`
        assert {s.name for s in exporter.get_finished_spans()} >= {"generate", "agent"}
    finally:
        processor.shutdown()

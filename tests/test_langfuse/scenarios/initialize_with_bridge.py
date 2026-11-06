"""Runs in a fresh interpreter (global OTel state can only be set once):

    python initialize_with_bridge.py MODE ORDER

MODE: `none` (no Laminar), `bridge`, `bridge_reinit` (initialize, shutdown,
initialize again) or `bridge_explicit_openai` (bridge + instruments={OPENAI}).
ORDER: `laminar_first` or `langfuse_first`. Prints one JSON document.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any
from unittest.mock import patch

os.environ.update(
    LANGFUSE_PUBLIC_KEY="pk-lf-test",
    LANGFUSE_SECRET_KEY="sk-lf-test",
    LANGFUSE_HOST="http://127.0.0.1:1",
    LANGFUSE_MEDIA_UPLOAD_ENABLED="False",
)

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

mode, order = sys.argv[1], sys.argv[2]

# Langfuse exports over OTLP/HTTP; Laminar's exporter is replaced below, so
# everything reaching this exporter is what Langfuse would have sent.
langfuse_spans: list[ReadableSpan] = []


def _record(_self: Any, spans: Any) -> SpanExportResult:
    langfuse_spans.extend(spans)
    return SpanExportResult.SUCCESS


OTLPSpanExporter.export = _record  # pyright: ignore[reportAttributeAccessIssue]

# One per `init_tracing`: `Laminar.shutdown` shuts the previous one down.
laminar_exporters: list[InMemorySpanExporter] = []
warnings: list[str] = []


class _Capture(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        warnings.append(record.getMessage())


def init_laminar() -> None:
    if mode == "none":
        return
    import lmnr.sdk.laminar as laminar_module
    from lmnr import Instruments, Laminar

    original = laminar_module.init_tracing

    def init_tracing(*args: Any, **kwargs: Any) -> Any:
        laminar_exporters.append(InMemorySpanExporter())
        kwargs["exporter"] = laminar_exporters[-1]
        return original(*args, **kwargs)

    logging.getLogger("lmnr.sdk.laminar").addHandler(_Capture())
    instruments = {Instruments.OPENAI} if mode == "bridge_explicit_openai" else None
    with patch.object(laminar_module, "init_tracing", side_effect=init_tracing):
        Laminar.initialize(
            project_api_key="test",
            disable_batch=True,
            bridges=["langfuse"],
            instruments=instruments,
        )
        if mode == "bridge_reinit":
            Laminar.shutdown()
            Laminar.initialize(project_api_key="test", disable_batch=True, bridges=["langfuse"])


from langfuse import get_client, observe, propagate_attributes

if order == "laminar_first":
    init_laminar()
    langfuse = get_client()
else:
    langfuse = get_client()
    init_laminar()

from lmnr import Laminar
from lmnr import observe as laminar_observe


@observe(as_type="generation")
def generate(question: str) -> str:
    langfuse.update_current_generation(
        model="gpt-4o",
        usage_details={"input": 3, "output": 5},
        input=[{"role": "user", "content": question}],
        output={"role": "assistant", "content": "hi"},
    )
    return "hi"


@observe(as_type="tool")
def lookup(x: int) -> int:
    return x


@laminar_observe(name="laminar_inner")
def laminar_inner() -> int:
    return lookup(2)


@observe()
def agent(question: str) -> str:
    with propagate_attributes(user_id="user-1", session_id="session-1"):
        generate(question)
        lookup(1)
        laminar_inner()
    return "done"


@laminar_observe(name="laminar_outer")
def outer() -> str:
    return agent("hello")


outer()
langfuse.flush()

openai_instrumented = None
laminar_is_global = None
if mode != "none":
    from lmnr.opentelemetry_lib.tracing.wrapper import get_tracer_wrapper

    wrapper = get_tracer_wrapper()
    laminar_is_global = (
        wrapper is not None and trace.get_tracer_provider() is wrapper.tracer_provider
    )
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import OpenAIInstrumentor

    openai_instrumented = OpenAIInstrumentor().is_instrumented_by_opentelemetry
    _flushed = Laminar.flush()


def dump(spans: Any, volatile: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    out = [
        {
            "name": s.name,
            "attributes": {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in sorted((s.attributes or {}).items())
                if k not in volatile
            },
            "events": [e.name for e in s.events],
            "status": s.status.status_code.name,
        }
        for s in spans
    ]
    return sorted(out, key=lambda s: (s["name"], json.dumps(s["attributes"], default=str)))


print(
    json.dumps(
        {
            "langfuse": dump(langfuse_spans, ("langfuse.observation.completion_start_time",)),
            "laminar": dump(
                laminar_exporters[-1].get_finished_spans() if laminar_exporters else []
            ),
            "laminar_is_global": laminar_is_global,
            "openai_instrumented": openai_instrumented,
            "warnings": warnings,
        },
        default=str,
    )
)

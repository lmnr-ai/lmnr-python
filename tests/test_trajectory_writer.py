"""The trajectory writer (`LMNR_TRAJECTORY_PATH`): live spans to lmnr-atif files."""

from __future__ import annotations

import base64
import importlib
import json
import os
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import jsonschema
import pytest
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from lmnr import observe
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from lmnr.opentelemetry_lib.tracing.trajectory import TrajectoryCollector, span_type
from lmnr.opentelemetry_lib.tracing.wrapper import get_tracer_wrapper
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.trajectory import reader
from lmnr.sdk.trajectory.normalize import normalize_call
from lmnr.sdk.trajectory.reader import TrajectoryExtension
from lmnr.sdk.trajectory.writer import media_sink, render_path, write_json_atomic

SPEC = Path(__file__).resolve().parents[1] / "spec" / "lmnr-atif"
SCHEMA = json.loads((SPEC / "lmnr-atif.schema.json").read_text())
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n-not-really-a-png").decode()


def assert_valid(trajectory: dict[str, Any]) -> None:
    jsonschema.validate(trajectory, SCHEMA)
    try:
        trajectories: Any = importlib.import_module("harbor.models.trajectories")
    except ImportError:  # harbor needs Python 3.12+; not a dev dependency
        return
    _ = trajectories.Trajectory.model_validate(trajectory)


# -- writer ---------------------------------------------------------------------


def test_render_path_fills_placeholders(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", "/home/agent")
    assert render_path("~/runs/{trace_id}/{pid}.json", "abc", pid=7) == Path(
        "/home/agent/runs/abc/7.json"
    )
    assert render_path("run.json", "abc") == Path("run.json")


def test_atomic_write_creates_directories_and_leaves_no_temp_files(tmp_path: Path):
    path = tmp_path / "a" / "b" / "trajectory.json"
    write_json_atomic(path, {"x": 1})
    write_json_atomic(path, {"x": 2})
    assert json.loads(path.read_text()) == {"x": 2}
    assert [p.name for p in path.parent.iterdir()] == ["trajectory.json"]


def test_media_sink_stores_images_by_content_hash(tmp_path: Path):
    store = media_sink(tmp_path)
    first = store(PNG, "image/png")
    assert first is not None and first.startswith("images/") and first.endswith(".png")
    assert (tmp_path / first).read_bytes() == base64.b64decode(PNG)
    assert store(PNG, "image/png") == first
    assert store(PNG, "image/tiff") is None  # not an ATIF image type
    assert store("not base64!", "image/png") is None


# -- collector ------------------------------------------------------------------


class Harness:
    """A private provider with a `LaminarSpanProcessor` writing a trajectory."""

    def __init__(self, path: Path, export: bool):
        self.path: Path = path
        self.exporter: InMemorySpanExporter = InMemorySpanExporter()
        self.processor: LaminarSpanProcessor = LaminarSpanProcessor(
            exporter=self.exporter,
            disable_batch=True,
            trajectory_path=str(path),
            export=export,
        )
        self.provider: TracerProvider = TracerProvider()
        self.provider.add_span_processor(self.processor)
        self.tracer: trace.Tracer = self.provider.get_tracer("test")

    @property
    def collector(self) -> TrajectoryCollector:
        assert self.processor.trajectory is not None
        return self.processor.trajectory

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Generator[trace.Span, None, None]:
        with self.tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span

    def read(self, path: Path | None = None) -> dict[str, Any]:
        return json.loads((path or self.path).read_text())


@pytest.fixture
def harness(tmp_path: Path) -> Generator[Harness, None, None]:
    h = Harness(tmp_path / "out" / "trajectory.json", export=False)
    yield h
    h.provider.shutdown()


def llm_call(h: Harness, messages: list[Any], output: list[Any], **attributes: Any) -> None:
    with h.span(
        "openai.chat",
        **{
            "lmnr.span.type": "LLM",
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-5.5",
            "gen_ai.input.messages": json.dumps(messages),
            "gen_ai.output.messages": json.dumps(output),
            "lmnr.span.input": json.dumps(messages),
            **attributes,
        },
    ):
        pass


TOOL_CALL = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"x"}'}}
    ],
}


def tool_loop(h: Harness, root: str = "agent", fail_tool: bool = False) -> None:
    ask = {"role": "user", "content": "Look up x"}
    with h.span(root):
        llm_call(h, [ask], [TOOL_CALL])
        with h.span(
            "lookup",
            **{"lmnr.span.type": "TOOL", "lmnr.span.input": '{"q":"x"}'},
        ) as tool:
            if fail_tool:
                tool.record_exception(RuntimeError("no x"))
                tool.set_status(Status(StatusCode.ERROR))
            else:
                tool.set_attribute("lmnr.span.output", '"found x"')
        result = {"role": "tool", "tool_call_id": "c1", "content": "found x"}
        llm_call(h, [ask, TOOL_CALL, result], [{"role": "assistant", "content": "x is found"}])


def test_local_only_writes_the_file_and_exports_nothing(harness: Harness):
    tool_loop(harness)
    assert not harness.path.exists()  # nothing until a flush or a snapshot
    assert harness.processor.force_flush()

    trajectory = harness.read()
    assert_valid(trajectory)
    assert harness.exporter.get_finished_spans() == ()
    assert harness.processor.instance is None
    assert harness.processor.exporter is None
    lmnr = trajectory["extra"]["lmnr"]
    assert lmnr["exported"] is False
    assert lmnr["complete"] is True
    assert trajectory["agent"]["name"] == "agent"
    assert [s["source"] for s in trajectory["steps"]] == ["user", "agent", "agent"]
    call = trajectory["steps"][1]
    assert call["tool_calls"][0]["extra"]["lmnr"]["span_id"]
    assert call["observation"]["results"][0]["content"] == "found x"
    assert call["observation"]["results"][0]["extra"]["lmnr"]["status"] == "ok"


def test_dual_mode_exports_and_marks_the_file_exported(tmp_path: Path):
    h = Harness(tmp_path / "t.json", export=True)
    tool_loop(h)
    assert h.processor.force_flush()
    assert len(h.exporter.get_finished_spans()) == 4
    assert h.read()["extra"]["lmnr"]["exported"] is True
    h.provider.shutdown()


def test_tool_errors_come_from_exception_events(harness: Harness):
    tool_loop(harness, fail_tool=True)
    _ = harness.processor.force_flush()
    result = harness.read()["steps"][1]["observation"]["results"][0]
    assert result["extra"]["lmnr"]["status"] == "error"
    assert result["extra"]["lmnr"]["error"]["type"] == "RuntimeError"
    assert result["extra"]["lmnr"]["error"]["message"] == "no x"


def test_snapshot_while_spans_are_open_is_incomplete(harness: Harness):
    with harness.span("agent"):
        llm_call(harness, [{"role": "user", "content": "hi"}], [TOOL_CALL])
        with harness.span("lookup", **{"lmnr.span.type": "TOOL"}):
            assert harness.collector.write()
            snapshot = harness.read()
    assert_valid(snapshot)
    assert snapshot["extra"]["lmnr"]["complete"] is False
    # The open root still names the agent.
    assert snapshot["agent"]["name"] == "agent"
    _ = harness.processor.force_flush()
    assert harness.read()["extra"]["lmnr"]["complete"] is True


def test_snapshots_are_written_in_the_background(harness: Harness):
    harness.collector.snapshot_interval = 0.05
    with harness.span("agent"):
        llm_call(
            harness, [{"role": "user", "content": "hi"}], [{"role": "assistant", "content": "yo"}]
        )
        deadline = time.monotonic() + 5
        while not harness.path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert harness.path.exists()
        assert harness.read()["extra"]["lmnr"]["complete"] is False


def test_failed_write_is_retried_at_the_next_snapshot(tmp_path: Path):
    blocker = tmp_path / "not-a-dir"
    _ = blocker.write_text("")
    h = Harness(blocker / "trajectory.json", export=False)
    h.collector.snapshot_interval = 0
    tool_loop(h)
    assert not h.collector.write()
    assert h.collector._dirty  # the snapshot thread will try again
    blocker.unlink()
    assert h.collector.write()
    assert not h.collector._dirty
    assert h.read()["extra"]["lmnr"]["complete"] is True
    h.provider.shutdown()


def test_collector_works_without_fork_support(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Windows has no `os.register_at_fork`.
    monkeypatch.delattr(os, "register_at_fork")
    collector = TrajectoryCollector(str(tmp_path / "t.json"), exported=False)
    assert collector.write()


def test_shutdown_writes_the_final_file(harness: Harness):
    tool_loop(harness)
    harness.provider.shutdown()
    assert harness.read()["extra"]["lmnr"]["complete"] is True


def test_one_file_per_trace_with_a_trace_id_placeholder(tmp_path: Path):
    h = Harness(tmp_path / "{trace_id}.json", export=False)
    tool_loop(h, root="first")
    tool_loop(h, root="second")
    _ = h.processor.force_flush()
    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 2
    names = {h.read(f)["agent"]["name"] for f in files}
    assert names == {"first", "second"}
    for file in files:
        trajectory = h.read(file)
        assert (
            file.stem == trajectory["trajectory_id"] == trajectory["extra"]["lmnr"]["trace_ids"][0]
        )
    h.provider.shutdown()


def test_a_finished_trace_stays_complete_while_another_runs(tmp_path: Path):
    h = Harness(tmp_path / "{trace_id}.json", export=False)
    h.collector.snapshot_interval = 0
    tool_loop(h, root="first")
    with h.span("second") as second:
        llm_call(h, [{"role": "user", "content": "hi"}], [TOOL_CALL])
        assert h.collector.write()
        running = f"{uuid.UUID(int=second.get_span_context().trace_id)}.json"
        complete = {f.name: h.read(f)["extra"]["lmnr"]["complete"] for f in tmp_path.glob("*")}
    assert complete.pop(running) is False
    assert list(complete.values()) == [True]
    h.provider.shutdown()


def test_force_reinit_starts_a_new_run(harness: Harness):
    # `Laminar.force_flush` between Lambda invocations: flush, then reinit.
    tool_loop(harness, root="first")
    stale = harness.tracer.start_span("never-ended")
    _ = harness.processor.force_flush()
    assert harness.read()["agent"]["name"] == "first"
    assert harness.processor.force_reinit()
    tool_loop(harness, root="second")
    _ = harness.processor.force_flush()
    lmnr = harness.read()["extra"]["lmnr"]
    assert harness.read()["agent"]["name"] == "second"
    assert len(lmnr["trace_ids"]) == 1
    assert lmnr["complete"] is True
    stale.end()


def test_a_span_open_at_force_reinit_stays_out_of_the_next_run(harness: Harness):
    # An `@observe` Lambda handler that calls `Laminar.force_flush` before
    # returning: its span ends after the reinit.
    with harness.span("first"):
        tool_loop(harness, root="first-agent")
        _ = harness.processor.force_flush()
        assert harness.processor.force_reinit()
    tool_loop(harness, root="second")
    _ = harness.processor.force_flush()
    trajectory = harness.read()
    assert trajectory["agent"]["name"] == "second"
    assert len(trajectory["extra"]["lmnr"]["trace_ids"]) == 1
    assert trajectory["extra"]["lmnr"]["complete"] is True


def test_without_a_placeholder_every_trace_goes_into_one_file(harness: Harness):
    tool_loop(harness, root="first")
    tool_loop(harness, root="second")
    _ = harness.processor.force_flush()
    trajectory = harness.read()
    assert len(trajectory["extra"]["lmnr"]["trace_ids"]) == 2
    assert [s["source"] for s in trajectory["steps"]].count("agent") == 4


def test_disabled_spans_are_not_written(harness: Harness):
    with harness.span("agent"):
        llm_call(
            harness, [{"role": "user", "content": "hi"}], [{"role": "assistant", "content": "yo"}]
        )
        llm_call(
            harness,
            [{"role": "user", "content": "native duplicate"}],
            [{"role": "assistant", "content": "no"}],
            **{"lmnr.internal.disabled": True},
        )
    _ = harness.processor.force_flush()
    trajectory = harness.read()
    assert [s["message"] for s in trajectory["steps"]] == ["hi", "yo"]
    assert trajectory["extra"]["lmnr"]["complete"] is True


def test_inline_images_are_stored_next_to_the_file(harness: Harness):
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}
    with harness.span("agent"):
        llm_call(
            harness,
            [{"role": "user", "content": [{"type": "text", "text": "what is it?"}, image]}],
            [{"role": "assistant", "content": "a png"}],
        )
    _ = harness.processor.force_flush()
    part = harness.read()["steps"][0]["message"][1]
    assert part["type"] == "image"
    assert (harness.path.parent / part["source"]["path"]).read_bytes() == base64.b64decode(PNG)


def test_records_keep_only_what_the_builder_reads(harness: Harness):
    big = json.dumps([{"role": "user", "content": "x" * 1000}])
    with harness.span("agent"):
        llm_call(
            harness,
            json.loads(big),
            [{"role": "assistant", "content": "y"}],
            **{"gen_ai.prompt.3.content": "dup", "lmnr.span.path": ("agent", "openai.chat")},
        )
    records = list(harness.collector._records.values())
    llm = next(r for r in records if r.span_type == "LLM")
    assert "gen_ai.input.messages" in llm.attributes
    assert "lmnr.span.input" not in llm.attributes  # same as gen_ai.input.messages
    assert "gen_ai.prompt.3.content" not in llm.attributes
    assert "lmnr.span.path" not in llm.attributes


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ({"lmnr.span.type": "TOOL"}, "TOOL"),
        ({"gen_ai.operation.name": "chat"}, "LLM"),
        ({"gen_ai.operation.name": "execute_tool"}, "TOOL"),
        ({"gen_ai.operation.name": "invoke_agent", "gen_ai.request.model": "m"}, "DEFAULT"),
        ({"gen_ai.tool.call.arguments": "{}"}, "TOOL"),
        ({"gen_ai.system": "openai"}, "LLM"),
        ({}, "DEFAULT"),
    ],
)
def test_span_type_matches_app_server_inference(attributes: dict[str, Any], expected: str):
    assert span_type(attributes) == expected


# -- Laminar.initialize ---------------------------------------------------------


@pytest.fixture
def uninitialized(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Laminar as if never initialized, with `init_tracing` stubbed out."""
    for name in ("LMNR_PROJECT_API_KEY", "LMNR_TRAJECTORY_PATH", "LMNR_TRAJECTORY_FORCE_EXPORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(Path(__file__).parent)  # no .env with a key
    saved = {
        name: getattr(Laminar, f"_Laminar__{name}")
        for name in ("initialized", "project_api_key", "base_http_url")
    }
    Laminar._Laminar__initialized = False  # pyright: ignore[reportAttributeAccessIssue]
    Laminar._Laminar__project_api_key = None  # pyright: ignore[reportAttributeAccessIssue]
    with (
        patch("lmnr.sdk.laminar.init_tracing") as init_tracing,
        patch("lmnr.sdk.laminar.get_otel_env_var", return_value=None),
        patch.object(Laminar, "_init_debug_runtime"),
        patch.object(Laminar, "_initialize_context_from_env"),
    ):
        yield init_tracing
    for name, value in saved.items():
        setattr(Laminar, f"_Laminar__{name}", value)


def test_trajectory_path_means_local_only_without_an_api_key(
    uninitialized: Any, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LMNR_TRAJECTORY_PATH", "/tmp/run/trajectory.json")
    Laminar.initialize()
    kwargs = uninitialized.call_args.kwargs
    assert kwargs["trajectory_path"] == "/tmp/run/trajectory.json"
    assert kwargs["export"] is False


def test_force_export_needs_an_api_key(uninitialized: Any, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LMNR_TRAJECTORY_PATH", "/tmp/run/trajectory.json")
    monkeypatch.setenv("LMNR_TRAJECTORY_FORCE_EXPORT", "true")
    with pytest.raises(ValueError):
        Laminar.initialize()
    Laminar.initialize(project_api_key="key")
    kwargs = uninitialized.call_args.kwargs
    assert kwargs["trajectory_path"] == "/tmp/run/trajectory.json"
    assert kwargs["export"] is True


def test_write_trajectory_false_ignores_the_path_even_from_dotenv(
    uninitialized: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A host process (the harbor plugin) whose env is meant for its agent."""
    _ = (tmp_path / ".env").write_text("LMNR_TRAJECTORY_PATH=/tmp/run/trajectory.json\n")
    monkeypatch.chdir(tmp_path)
    Laminar.initialize(project_api_key="key", write_trajectory=False)
    kwargs = uninitialized.call_args.kwargs
    assert kwargs["trajectory_path"] is None
    assert kwargs["export"] is True


def test_dotenv_trajectory_path_applies_by_default(
    uninitialized: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _ = (tmp_path / ".env").write_text("LMNR_TRAJECTORY_PATH=/tmp/run/trajectory.json\n")
    monkeypatch.chdir(tmp_path)
    Laminar.initialize()
    kwargs = uninitialized.call_args.kwargs
    assert kwargs["trajectory_path"] == "/tmp/run/trajectory.json"
    assert kwargs["export"] is False


def test_without_a_trajectory_path_nothing_changes(uninitialized: Any):
    with pytest.raises(ValueError):
        Laminar.initialize()
    Laminar.initialize(project_api_key="key")
    kwargs = uninitialized.call_args.kwargs
    assert kwargs["trajectory_path"] is None
    assert kwargs["export"] is True


# -- end to end -----------------------------------------------------------------


@pytest.fixture
def global_trajectory(tmp_path: Path) -> Generator[Path, None, None]:
    """Attach a collector to the session's Laminar processor."""
    wrapper = get_tracer_wrapper()
    assert wrapper is not None
    processor = cast(LaminarSpanProcessor, wrapper.span_processor)
    path = tmp_path / "trajectory.json"
    processor.trajectory = TrajectoryCollector(str(path), exported=True, snapshot_interval=0)
    yield path
    processor.trajectory = None


WEATHER_TOOL: Any = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@pytest.mark.vcr
def test_openai_tool_loop_end_to_end(span_exporter: InMemorySpanExporter, global_trajectory: Path):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "test-key"))

    @observe(span_type="TOOL")
    def get_weather(city: str) -> str:
        return f"Sunny, 21C in {city}"

    @observe(name="weather-agent")
    def agent(question: str) -> str | None:
        messages: list[Any] = [
            {"role": "system", "content": "Answer with the get_weather tool."},
            {"role": "user", "content": question},
        ]
        while True:
            response = client.chat.completions.create(
                model="gpt-5.4-nano", messages=messages, tools=[WEATHER_TOOL]
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                return message.content
            for call in message.tool_calls:
                arguments = json.loads(cast(Any, call).function.arguments)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": get_weather(**arguments)}
                )

    answer = agent("What's the weather in Paris?")
    assert Laminar.flush()

    trajectory = json.loads(global_trajectory.read_text())
    assert_valid(trajectory)
    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    root = spans["weather-agent"]
    assert root.context is not None
    trace_id = f"{root.context.trace_id:032x}"
    assert trajectory["extra"]["lmnr"]["trace_ids"][0].replace("-", "") == trace_id
    assert trajectory["agent"]["name"] == "weather-agent"
    assert trajectory["agent"]["model_name"].startswith("openai/gpt-5.4-nano")

    steps = trajectory["steps"]
    assert [s["source"] for s in steps] == ["system", "user", "agent", "agent"]
    first, last = steps[2], steps[3]
    call = first["tool_calls"][0]
    assert call["function_name"] == "get_weather"
    assert call["arguments"] == {"city": "Paris"}
    tool_span = spans["get_weather"]
    assert tool_span.context is not None
    assert call["extra"]["lmnr"]["span_id"].replace("-", "")[-16:] == (
        f"{tool_span.context.span_id:016x}"
    )
    assert first["observation"]["results"][0]["content"] == "Sunny, 21C in Paris"
    assert last["message"] == answer
    assert first["metrics"]["prompt_tokens"] > 0

    # Reading the file back gives each call the exact input it had.
    ext = TrajectoryExtension.read(trajectory)
    llm_spans = [s for s in span_exporter.get_finished_spans() if s.name == "openai.chat"]
    for index, span in zip((2, 3), sorted(llm_spans, key=_start), strict=True):
        actual, _output, _reason = normalize_call(
            json.loads(cast(str, (span.attributes or {})["gen_ai.input.messages"])), None
        )
        rebuilt = reader.input_messages(steps, index, ext)
        assert _comparable(rebuilt) == _comparable(actual)


def _start(span: ReadableSpan) -> int:
    return span.start_time or 0


def _comparable(messages: list[dict[str, Any]]) -> list[Any]:
    """What the model saw: writer metadata on tool calls aside."""
    return [
        [
            m.get("role"),
            m.get("content") or "",
            [
                [c.get("tool_call_id"), c.get("function_name"), c.get("arguments")]
                for c in cast(list[dict[str, Any]], m.get("tool_calls") or [])
            ],
            m.get("tool_call_id"),
        ]
        for m in messages
    ]

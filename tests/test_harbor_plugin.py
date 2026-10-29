"""Tests for the Harbor plugin (`lmnr.integrations.harbor`).

Harbor requires Python 3.12+, so these tests drive the plugin with lightweight
fakes of Harbor's `Job` and `TrialHookEvent` instead of importing `harbor`.
"""

from __future__ import annotations

import datetime
import itertools
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext, StatusCode

from lmnr import Laminar
from lmnr.integrations.harbor import LaminarPlugin
from lmnr.integrations.harbor.plugin import AGENT_TRAJECTORY_PATH, _request_agent_trajectories
from lmnr.integrations.harbor.trajectory import (
    ORDER_GAP_NS,
    content_to_text,
    parse_timestamp_ns,
    split_model_name,
)
from lmnr.sdk.evaluations.models import EvaluationResultDatapoint, PartialEvaluationDatapoint
from lmnr.sdk.evaluations.utils import get_evaluation_url

T0 = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)


def _ctx(span: ReadableSpan) -> SpanContext:
    ctx = span.get_span_context()
    assert ctx is not None
    return ctx


def ts(seconds: float) -> datetime.datetime:
    return T0 + datetime.timedelta(seconds=seconds)


def iso(seconds: float) -> str:
    return ts(seconds).isoformat().replace("+00:00", "Z")


class FakeJob:
    def __init__(self, job_dir: Path, dataset_name: str | None = "hello-world"):
        self.id: uuid.UUID = uuid.uuid4()
        self.config: SimpleNamespace = SimpleNamespace(
            job_name="2026-01-01__12-00-00",
            datasets=(
                [SimpleNamespace(name=dataset_name, path=None, version="1.0")]
                if dataset_name
                else []
            ),
            tasks=[],
            agents=[
                SimpleNamespace(
                    name="terminus-2",
                    import_path=None,
                    model_name="anthropic/claude-sonnet-4-5",
                    env={},
                )
            ],
        )
        self.job_dir: Path = job_dir
        self.hooks: dict[str, list[Any]] = {}

    def _add(self, name: str, callback: Callable[..., Any]) -> FakeJob:
        self.hooks.setdefault(name, []).append(callback)
        return self

    def on_trial_started(self, cb: Callable[..., Any]) -> FakeJob:
        return self._add("start", cb)

    def on_agent_started(self, cb: Callable[..., Any]) -> FakeJob:
        return self._add("agent_start", cb)

    def on_verification_started(self, cb: Callable[..., Any]) -> FakeJob:
        return self._add("verification_start", cb)

    def on_trial_cancelled(self, cb: Callable[..., Any]) -> FakeJob:
        return self._add("cancel", cb)

    def on_trial_ended(self, cb: Callable[..., Any]) -> FakeJob:
        return self._add("end", cb)

    async def emit(self, name: str, event: Any) -> None:
        for cb in self.hooks.get(name, []):
            await cb(event)


def make_task_dir(root: Path, name: str) -> Path:
    task_dir = root / "tasks" / name
    task_dir.mkdir(parents=True)
    _bytes_written = (task_dir / "instruction.md").write_text(f"Solve {name}.")
    return task_dir


def make_event(
    job: FakeJob,
    task_dir: Path,
    trial_name: str,
    *,
    rewards: dict[str, Any] | None = None,
    exception: Any = None,
    timestamp: datetime.datetime | None = None,
):
    timing = SimpleNamespace
    result = SimpleNamespace(
        id=uuid.uuid4(),
        trial_uri=f"file://{job.job_dir / trial_name}",
        agent_info=SimpleNamespace(name="terminus-2", version="2.0.0"),
        agent_result=SimpleNamespace(
            n_input_tokens=300,
            n_output_tokens=60,
            n_cache_tokens=100,
            cost_usd=0.01,
            metadata={"n_episodes": 2},
        ),
        verifier_result=(SimpleNamespace(rewards=rewards) if rewards is not None else None),
        exception_info=exception,
        started_at=ts(0),
        finished_at=ts(30),
        environment_setup=timing(started_at=ts(1), finished_at=ts(5)),
        agent_setup=timing(started_at=ts(5), finished_at=ts(6)),
        agent_execution=timing(started_at=ts(6), finished_at=ts(20)),
        verifier=timing(started_at=ts(20), finished_at=ts(29)),
    )
    config = SimpleNamespace(
        trial_name=trial_name,
        trials_dir=job.job_dir,
        task=SimpleNamespace(get_local_path=lambda: task_dir),
        agent=job.config.agents[0],
        environment=SimpleNamespace(type="docker", import_path=None, override_cpus=4),
    )
    return SimpleNamespace(
        event="start",
        task_name=task_dir.name,
        trial_name=trial_name,
        trial_id=result.id,
        config=config,
        result=result,
        timestamp=timestamp or ts(0),
    )


TRAJECTORY = {
    "schema_version": "ATIF-v1.6",
    "session_id": "s1",
    "agent": {
        "name": "terminus-2",
        "version": "2.0.0",
        "model_name": "anthropic/claude-sonnet-4-5",
    },
    "steps": [
        {"step_id": 1, "timestamp": iso(6), "source": "system", "message": "sys"},
        {"step_id": 2, "timestamp": iso(6), "source": "user", "message": "task"},
        {
            "step_id": 3,
            "timestamp": iso(10),
            "source": "agent",
            "message": "Listing files",
            "reasoning_content": "I should look around",
            "tool_calls": [
                {
                    "tool_call_id": "call_1",
                    "function_name": "bash",
                    "arguments": {"command": "ls"},
                }
            ],
            "observation": {"results": [{"source_call_id": "call_1", "content": "hello.txt"}]},
            "metrics": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cached_tokens": 50,
                "cost_usd": 0.004,
            },
        },
        {
            "step_id": 4,
            "timestamp": iso(15),
            "source": "agent",
            "message": [{"type": "text", "text": "Done"}],
            "metrics": {"prompt_tokens": 200, "completion_tokens": 40},
        },
    ],
}


def write_trajectory(job: FakeJob, trial_name: str, trajectory: Any = TRAJECTORY) -> None:
    agent_dir = job.job_dir / trial_name / "agent"
    agent_dir.mkdir(parents=True)
    _bytes_written = (agent_dir / "trajectory.json").write_text(json.dumps(trajectory))


@pytest.fixture(autouse=True)
def initialized_project_key():
    # Other test modules re-initialize Laminar with their own keys, and the
    # plugin rejects a key that differs from the initialized one.
    with patch.object(Laminar, "get_project_api_key", return_value="test_key"):
        yield


@pytest.fixture
def client():
    client = MagicMock()
    client.evals.init = AsyncMock(return_value={"id": uuid.uuid4(), "projectId": uuid.uuid4()})
    client.evals.save_datapoints = AsyncMock()
    client.close = AsyncMock()
    with patch("lmnr.integrations.harbor.plugin.AsyncLaminarClient", return_value=client):
        yield client


def saved_datapoints(client) -> list[Any]:
    return [dp for call in client.evals.save_datapoints.call_args_list for dp in call[0][1]]


def spans_by_name(exporter: InMemorySpanExporter) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for span in exporter.get_finished_spans():
        result.setdefault(span.name, []).append(span)
    return result


@pytest.mark.asyncio
async def test_job_reports_trials_as_evaluation(
    span_exporter: InMemorySpanExporter, client: MagicMock, tmp_path: Path
):
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "hello-world")
    plugin = LaminarPlugin(project_api_key="test_key")

    await plugin.on_job_start(job)
    client.evals.init.assert_awaited_once()
    init_kwargs = client.evals.init.call_args.kwargs
    assert init_kwargs["name"] == job.config.job_name
    assert init_kwargs["group_name"] == "hello-world@1.0"
    assert init_kwargs["metadata"]["source"] == "harbor"

    event = make_event(job, task_dir, "hello-world__abc", rewards={"reward": 1.0})
    write_trajectory(job, "hello-world__abc")
    await job.emit("start", event)
    await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())

    partial, full = saved_datapoints(client)
    assert isinstance(partial, PartialEvaluationDatapoint)
    assert isinstance(full, EvaluationResultDatapoint)
    assert partial.id == full.id
    assert partial.trace_id == full.trace_id
    assert full.index == 0
    assert full.data == {
        "task_name": "hello-world",
        "instruction": "Solve hello-world.",
    }
    assert full.scores == {"reward": 1.0}
    assert full.executor_output == "Done"
    assert full.metadata is not None
    assert full.metadata["agent"] == "terminus-2"
    client.close.assert_awaited_once()

    spans = spans_by_name(span_exporter)
    root = spans["hello-world__abc"][0]
    assert root.attributes["lmnr.span.type"] == "EVALUATION"
    assert root.parent is None
    assert root.start_time == parse_timestamp_ns(ts(0))
    assert root.end_time == parse_timestamp_ns(ts(30))
    assert json.loads(root.attributes["lmnr.span.output"]) == {"rewards": {"reward": 1.0}}

    agent = spans["agent"][0]
    assert agent.attributes["lmnr.span.type"] == "EXECUTOR"
    assert agent.parent.span_id == root.context.span_id
    assert uuid.UUID(int=agent.context.span_id) == full.executor_span_id
    assert spans["verifier"][0].attributes["lmnr.span.type"] == "EVALUATOR"
    assert spans["environment_setup"][0].parent.span_id == root.context.span_id

    # Every span belongs to the evaluation trace and carries the evaluation id.
    trace_ids = {_ctx(s).trace_id for s in span_exporter.get_finished_spans()}
    assert trace_ids == {root.context.trace_id}
    eval_id = str(client.evals.init.return_value["id"])
    for span in span_exporter.get_finished_spans():
        assert (span.attributes or {})[
            "lmnr.association.properties.metadata.evaluation_id"
        ] == eval_id
        assert (span.attributes or {})["lmnr.association.properties.trace_type"] == ("EVALUATION")


@pytest.mark.asyncio
async def test_trajectory_becomes_llm_and_tool_spans(
    span_exporter: InMemorySpanExporter, client: MagicMock, tmp_path: Path
):
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "hello-world")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)

    event = make_event(job, task_dir, "t1", rewards={"reward": 1})
    write_trajectory(job, "t1")
    await job.emit("start", event)
    await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())

    spans = spans_by_name(span_exporter)
    agent = spans["agent"][0]
    llm_spans = sorted(spans["claude-sonnet-4-5"], key=lambda s: s.start_time or 0)
    assert len(llm_spans) == 2
    first, second = llm_spans
    for span in llm_spans:
        assert span.attributes["lmnr.span.type"] == "LLM"
        assert span.parent.span_id == agent.context.span_id
        assert span.attributes["gen_ai.system"] == "anthropic"
        assert span.attributes["gen_ai.request.model"] == "claude-sonnet-4-5"

    # The first call runs from the previous step to this step.
    assert first.start_time == parse_timestamp_ns(ts(6))
    assert first.end_time == parse_timestamp_ns(ts(10))
    assert first.attributes["gen_ai.usage.input_tokens"] == 100
    assert first.attributes["gen_ai.usage.output_tokens"] == 20
    assert first.attributes["llm.usage.total_tokens"] == 120
    assert first.attributes["gen_ai.usage.cache_read_input_tokens"] == 50
    assert first.attributes["gen_ai.usage.cost"] == 0.004
    assert json.loads(first.attributes["lmnr.span.input"]) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    output = json.loads(first.attributes["lmnr.span.output"])
    assert output[0]["role"] == "assistant"
    assert output[0]["reasoning_content"] == "I should look around"
    assert output[0]["tool_calls"][0]["function"] == {
        "name": "bash",
        "arguments": '{"command": "ls"}',
    }

    # The second call sees the tool result in its history.
    second_input = json.loads(second.attributes["lmnr.span.input"])
    assert second_input[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "hello.txt",
    }
    assert json.loads(second.attributes["lmnr.span.output"])[0]["content"] == "Done"
    # It starts once the previous step's tool call is done, not at the same time.
    ts_10 = parse_timestamp_ns(ts(10))
    assert ts_10 is not None
    assert second.start_time == ts_10 + ORDER_GAP_NS
    assert second.end_time == parse_timestamp_ns(ts(15))

    tool = spans["bash"][0]
    assert tool.attributes["lmnr.span.type"] == "TOOL"
    assert tool.parent.span_id == agent.context.span_id
    assert json.loads(tool.attributes["lmnr.span.input"]) == {"command": "ls"}
    assert json.loads(tool.attributes["lmnr.span.output"]) == "hello.txt"
    assert tool.start_time == parse_timestamp_ns(ts(10))
    # ATIF has no tool durations, so tool spans are instants at the step.
    assert tool.end_time == tool.start_time


@pytest.mark.asyncio
async def test_embedded_subagent_is_nested_under_tool(
    span_exporter: InMemorySpanExporter, client: MagicMock, tmp_path: Path
):
    trajectory = {
        "agent": {"name": "main", "model_name": "openai/gpt-5"},
        "steps": [
            {
                "step_id": 1,
                "timestamp": iso(7),
                "source": "agent",
                "message": "delegating",
                "tool_calls": [
                    {
                        "tool_call_id": "c1",
                        "function_name": "task",
                        "arguments": {"prompt": "x"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "c1",
                            "content": "sub done",
                            "subagent_trajectory_ref": [{"trajectory_id": "sub-1"}],
                        }
                    ]
                },
            },
            # A copied-context step isn't a new LLM call.
            {
                "step_id": 2,
                "timestamp": iso(8),
                "source": "agent",
                "message": "copied",
                "is_copied_context": True,
            },
            # A deterministic dispatch step isn't an LLM call either.
            {
                "step_id": 3,
                "timestamp": iso(9),
                "source": "agent",
                "message": "dispatch",
                "llm_call_count": 0,
            },
        ],
        "subagent_trajectories": [
            {
                "trajectory_id": "sub-1",
                "agent": {"name": "explorer", "model_name": "openai/gpt-5-mini"},
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": iso(7.5),
                        "source": "agent",
                        "message": "explored",
                    }
                ],
            }
        ],
    }
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    event = make_event(job, task_dir, "t1", rewards={"reward": 0})
    write_trajectory(job, "t1", trajectory)
    await job.emit("start", event)
    await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())

    spans = spans_by_name(span_exporter)
    assert len(spans["gpt-5"]) == 1
    tool = spans["task"][0]
    subagent = spans["explorer"][0]
    assert subagent.parent.span_id == tool.context.span_id
    sub_llm = spans["gpt-5-mini"][0]
    assert sub_llm.parent.span_id == subagent.context.span_id
    # The tool span stretches to cover the subagent's own steps.
    assert subagent.start_time == subagent.end_time == parse_timestamp_ns(ts(7.5))
    assert tool.start_time == parse_timestamp_ns(ts(7))
    assert tool.end_time == parse_timestamp_ns(ts(7.5))


@pytest.mark.asyncio
async def test_spans_start_in_trajectory_order(
    span_exporter: InMemorySpanExporter, client: MagicMock, tmp_path: Path
):
    def agent_step(step_id: int, seconds: float, calls: list[str]) -> dict[str, Any]:
        return {
            "step_id": step_id,
            "timestamp": iso(seconds),
            "source": "agent",
            "message": f"step {step_id}",
            "tool_calls": [{"tool_call_id": c, "function_name": c, "arguments": {}} for c in calls],
            "observation": {"results": [{"source_call_id": c, "content": "ok"} for c in calls]},
        }

    trajectory = {
        "agent": {"name": "main", "model_name": "openai/gpt-5"},
        "steps": [
            {"step_id": 1, "timestamp": iso(6), "source": "user", "message": "go"},
            agent_step(2, 8, ["read", "grep"]),
            # Same timestamp as the previous step: still ordered after its tools.
            agent_step(3, 8, ["edit"]),
            agent_step(4, 9, []),
        ],
    }
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    event = make_event(job, task_dir, "t1", rewards={"reward": 0})
    write_trajectory(job, "t1", trajectory)
    await job.emit("start", event)
    await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())

    agent = spans_by_name(span_exporter)["agent"][0]
    children = [
        s
        for s in span_exporter.get_finished_spans()
        if s.parent is not None and s.parent.span_id == agent.context.span_id
    ]
    emitted = [s.name for s in children]
    by_start = [s.name for s in sorted(children, key=lambda s: s.start_time or 0)]
    assert emitted == ["gpt-5", "read", "grep", "gpt-5", "edit", "gpt-5"]
    assert by_start == emitted
    starts = [s.start_time or 0 for s in children]
    assert all(b - a >= ORDER_GAP_NS for a, b in itertools.pairwise(starts))
    assert all((s.end_time or 0) >= (s.start_time or 0) for s in children)


@pytest.mark.asyncio
async def test_bare_step_model_takes_agent_provider(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    # terminus-2 records the model the API returned, without the provider.
    trajectory = {
        "agent": {"name": "terminus-2", "model_name": "anthropic/claude-sonnet-4-5"},
        "steps": [
            {"step_id": 1, "timestamp": iso(6), "source": "user", "message": "go"},
            {
                "step_id": 2,
                "timestamp": iso(7),
                "source": "agent",
                "model_name": "claude-sonnet-4-5-20250929",
                "message": "done",
            },
            {
                "step_id": 3,
                "timestamp": iso(8),
                "source": "agent",
                "model_name": "openai/gpt-5-mini",
                "message": "checked",
            },
        ],
    }
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    event = make_event(job, task_dir, "t1", rewards={"reward": 0})
    write_trajectory(job, "t1", trajectory)
    await job.emit("start", event)
    await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())

    spans = spans_by_name(span_exporter)
    bare = spans["claude-sonnet-4-5-20250929"][0].attributes
    assert bare["gen_ai.system"] == "anthropic"
    assert bare["gen_ai.request.model"] == "claude-sonnet-4-5-20250929"
    assert spans["gpt-5-mini"][0].attributes["gen_ai.system"] == "openai"


@pytest.mark.asyncio
async def test_unattributed_observation_goes_on_last_tool(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    # terminus-2 records one terminal observation per step, not one per call.
    trajectory = {
        "agent": {"name": "terminus-2", "model_name": "anthropic/claude"},
        "steps": [
            {"step_id": 1, "timestamp": iso(6), "source": "user", "message": "go"},
            {
                "step_id": 2,
                "timestamp": iso(8),
                "source": "agent",
                "message": "run two commands",
                "tool_calls": [
                    {"tool_call_id": "a", "function_name": "first", "arguments": {}},
                    {"tool_call_id": "b", "function_name": "second", "arguments": {}},
                ],
                "observation": {"results": [{"content": "$ ls\nfile.txt"}]},
            },
        ],
    }
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    event = make_event(job, task_dir, "t1", rewards={"reward": 0})
    write_trajectory(job, "t1", trajectory)
    await job.emit("start", event)
    await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())

    spans = spans_by_name(span_exporter)
    assert "lmnr.span.output" not in spans["first"][0].attributes
    assert json.loads(spans["second"][0].attributes["lmnr.span.output"]) == ("$ ls\nfile.txt")


# -- lmnr-atif trajectories (spec/lmnr-atif) -----------------------------------

LMNR_ATIF_FIXTURES = Path(__file__).resolve().parents[1] / "spec" / "lmnr-atif" / "fixtures"


def lmnr_atif_fixture(name: str, exported: bool = False) -> dict[str, Any]:
    """A spec fixture. Not exported by default: an exported file's spans are
    already in Laminar, so the plugin doesn't rebuild them."""
    trajectory = json.loads((LMNR_ATIF_FIXTURES / name).read_text())
    trajectory["extra"]["lmnr"]["exported"] = exported
    return trajectory


async def run_trial_with_trajectory(
    client: MagicMock, tmp_path: Path, trajectory: dict[str, Any]
) -> None:
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    event = make_event(job, task_dir, "t1", rewards={"reward": 1})
    write_trajectory(job, "t1", trajectory)
    await job.emit("start", event)
    await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())


def exception_events(span: ReadableSpan) -> list[dict[str, Any]]:
    return [dict(e.attributes or {}) for e in span.events if e.name == "exception"]


@pytest.mark.asyncio
async def test_lmnr_atif_trajectory_uses_exact_timings_and_call_details(
    span_exporter: InMemorySpanExporter, client: MagicMock, tmp_path: Path
):
    trajectory = lmnr_atif_fixture("basic-agent-loop.json")
    await run_trial_with_trajectory(client, tmp_path, trajectory)

    spans = spans_by_name(span_exporter)
    first, second = sorted(spans["claude-sonnet-4-5-20250929"], key=lambda s: s.start_time or 0)
    step3, step4 = trajectory["steps"][2], trajectory["steps"][3]
    # Exact call bounds, not "previous step to this step".
    assert first.start_time == parse_timestamp_ns(step3["extra"]["lmnr"]["start_time"])
    assert first.end_time == parse_timestamp_ns(step3["extra"]["lmnr"]["end_time"])
    assert second.start_time == parse_timestamp_ns(step4["extra"]["lmnr"]["start_time"])

    attrs = first.attributes or {}
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.request.max_tokens"] == 4096
    assert attrs["gen_ai.request.temperature"] == 0
    assert tuple(attrs["gen_ai.response.finish_reasons"]) == ("tool_use",)
    assert attrs["gen_ai.response.id"] == "msg_01"
    assert attrs["gen_ai.usage.cache_creation_input_tokens"] == 300
    assert attrs["gen_ai.usage.input_cost"] == 0.00132
    assert attrs["gen_ai.usage.output_cost"] == 0.00096
    assert attrs["lmnr.trajectory.source_span_id"] == step3["extra"]["lmnr"]["span_id"]
    tools = json.loads(attrs["gen_ai.tool.definitions"])
    assert [t["function"]["name"] for t in tools] == ["get_order", "refund_order"]
    assert exception_events(first) == []

    # Tools get their real duration.
    tool = spans["get_order"][0]
    tool_lmnr = step3["tool_calls"][0]["extra"]["lmnr"]
    assert tool.start_time == parse_timestamp_ns(tool_lmnr["start_time"])
    assert tool.end_time == parse_timestamp_ns(tool_lmnr["end_time"])
    assert (tool.attributes or {})["lmnr.trajectory.source_span_id"] == tool_lmnr["span_id"]

    agent = (spans["agent"][0].attributes) or {}
    assert agent["lmnr.trajectory.spec_version"] == "0.1.0"
    assert tuple(agent["lmnr.trajectory.trace_ids"]) == tuple(
        trajectory["extra"]["lmnr"]["trace_ids"]
    )
    assert agent["lmnr.trajectory.exported"] is False
    assert agent["lmnr.trajectory.complete"] is True
    assert agent["lmnr.trajectory.llm_calls"] == 2
    assert agent["lmnr.trajectory.duration_ms"] == 3790


@pytest.mark.asyncio
async def test_exported_lmnr_atif_trajectory_is_linked_not_rebuilt(
    span_exporter: InMemorySpanExporter, client: MagicMock, tmp_path: Path
):
    """The SDK exported these spans itself (dual mode): rebuilding them would
    record every call twice, so the agent span only links to the traces."""
    trajectory = lmnr_atif_fixture("subagents.json", exported=True)
    await run_trial_with_trajectory(client, tmp_path, trajectory)

    spans = span_exporter.get_finished_spans()
    assert not [s for s in spans if (s.attributes or {}).get("lmnr.trajectory.source_span_id")]
    assert not [s for s in spans if (s.attributes or {}).get("lmnr.span.type") in ("LLM", "TOOL")]
    agent = spans_by_name(span_exporter)["agent"][0].attributes or {}
    assert agent["lmnr.trajectory.exported"] is True
    assert tuple(agent["lmnr.trajectory.trace_ids"]) == tuple(
        trajectory["extra"]["lmnr"]["trace_ids"]
    )
    assert (
        agent["lmnr.trajectory.llm_calls"]
        == trajectory["final_metrics"]["extra"]["lmnr"]["llm_calls"]
    )


@pytest.mark.asyncio
async def test_lmnr_atif_overrides_shape_each_call_input(
    span_exporter: InMemorySpanExporter, client: MagicMock, tmp_path: Path
):
    trajectory = lmnr_atif_fixture("overrides.json")
    await run_trial_with_trajectory(client, tmp_path, trajectory)

    calls = sorted(spans_by_name(span_exporter)["gpt-5.5"], key=lambda s: s.start_time or 0)
    assert len(calls) == 5
    inputs = [json.loads((c.attributes or {})["lmnr.span.input"]) for c in calls]
    default_prompt = [
        {"role": "system", "content": "You are a travel agent."},
        {"role": "developer", "content": "Always quote prices in EUR."},
    ]
    assert inputs[0] == [*default_prompt, {"role": "user", "content": "Plan a trip to Lisbon."}]

    # Call 2: its own system prompt and no tools.
    assert inputs[1][0]["role"] == "system"
    assert "premium member" in inputs[1][0]["content"]
    assert [m["role"] for m in inputs[1]] == ["system", "user", "assistant", "tool"]
    assert "gen_ai.tool.definitions" not in (calls[1].attributes or {})

    # Call 3: back to the default prompt, with a narrowed tool set.
    assert inputs[2][:2] == default_prompt
    tools = json.loads((calls[2].attributes or {})["gen_ai.tool.definitions"])
    assert [t["function"]["name"] for t in tools] == ["search_hotels"]

    # Call 4: after compaction, only the compacted conversation.
    compacted = [
        *default_prompt,
        {"role": "system", "content": "Earlier turns were compacted into a summary."},
        {"role": "user", "content": trajectory["steps"][8]["message"]},
    ]
    assert inputs[3] == compacted
    # Call 5 extends call 4's input rather than the whole history.
    assert inputs[4] == [
        *compacted,
        {"role": "assistant", "content": trajectory["steps"][9]["message"]},
        {"role": "user", "content": "Any cheaper one?"},
    ]


@pytest.mark.asyncio
async def test_lmnr_atif_errors_and_deterministic_dispatch(
    span_exporter: InMemorySpanExporter, client: MagicMock, tmp_path: Path
):
    trajectory = lmnr_atif_fixture("errors-and-dispatch.json")
    await run_trial_with_trajectory(client, tmp_path, trajectory)

    spans = spans_by_name(span_exporter)
    ok_call, failed_call = sorted(spans["claude-opus-4-1"], key=lambda s: s.start_time or 0)
    assert exception_events(ok_call) == []
    assert ok_call.status.status_code != StatusCode.ERROR
    [event] = exception_events(failed_call)
    assert event["exception.type"] == "RateLimitError"
    assert event["exception.message"] == "Error code: 429 - rate_limit_error"
    assert failed_call.status.status_code == StatusCode.ERROR
    # The dispatch step is not part of what the model saw.
    failed_input = json.loads((failed_call.attributes or {})["lmnr.span.input"])
    assert [m["role"] for m in failed_input] == ["user", "assistant", "tool", "tool"]
    assert failed_input[1]["tool_calls"][1]["function"]["arguments"] == '{"path": "src/app.py"'

    [event] = exception_events(spans["run_tests"][0])
    assert event["exception.type"] == "TimeoutError"
    assert spans["run_tests"][0].status.status_code == StatusCode.ERROR
    # Arguments that weren't a JSON object are shown as the model wrote them.
    read_file = spans["read_file"][0]
    assert json.loads((read_file.attributes or {})["lmnr.span.input"]) == '{"path": "src/app.py"'
    assert exception_events(read_file)[0]["exception.type"] == "JSONDecodeError"

    # The deterministic dispatch is a tool span without an LLM call.
    notify = spans["notify_slack"][0]
    assert exception_events(notify) == []
    assert notify.end_time - notify.start_time == 400_000_000

    agent = spans["agent"][0].attributes or {}
    assert agent["lmnr.trajectory.complete"] is False
    assert agent["lmnr.trajectory.errors"] == 3
    assert len(agent["lmnr.trajectory.trace_ids"]) == 2


@pytest.mark.asyncio
async def test_lmnr_atif_subagents_nest_under_their_tool_calls(
    span_exporter: InMemorySpanExporter, client: MagicMock, tmp_path: Path
):
    await run_trial_with_trajectory(client, tmp_path, lmnr_atif_fixture("subagents.json"))

    spans = spans_by_name(span_exporter)
    for tool_name, agent_name in (
        ("delegate_research", "researcher"),
        ("summarize_sources", "summarize_sources"),
        ("fact_checker", "fact_checker"),
    ):
        tool = next(s for s in spans[tool_name] if (s.attributes or {})["lmnr.span.type"] == "TOOL")
        subagent = next(
            s for s in spans[agent_name] if (s.attributes or {})["lmnr.span.type"] != "TOOL"
        )
        assert subagent.parent.span_id == tool.context.span_id, agent_name
        assert "lmnr.trajectory.source_span_id" in (subagent.attributes or {})

    researcher = next(
        s for s in spans["researcher"] if (s.attributes or {})["lmnr.span.type"] != "TOOL"
    )
    researcher_calls = [
        s for s in spans["gpt-5.5-mini"] if s.parent.span_id == researcher.context.span_id
    ]
    assert len(researcher_calls) == 2
    # The researcher's tool set is a `$ref` into the root trajectory's defs.
    for call in researcher_calls:
        tools = json.loads((call.attributes or {})["gen_ai.tool.definitions"])
        assert [t["function"]["name"] for t in tools] == ["web_search"]
    # Its first call's input is its own conversation, not the parent's.
    first_input = json.loads(
        (min(researcher_calls, key=lambda s: s.start_time or 0).attributes or {})["lmnr.span.input"]
    )
    assert first_input == [
        {"role": "system", "content": "You research databases. Cite sources."},
        {"role": "user", "content": "Research: ClickHouse"},
    ]
    assert len(spans["claude-sonnet-4-5"]) == 2


@pytest.mark.asyncio
async def test_incompatible_lmnr_atif_version_is_read_as_plain_atif(
    span_exporter: InMemorySpanExporter, client: MagicMock, tmp_path: Path
):
    trajectory = lmnr_atif_fixture("basic-agent-loop.json")
    trajectory["extra"]["lmnr"]["spec_version"] = "0.2.0"
    await run_trial_with_trajectory(client, tmp_path, trajectory)

    spans = spans_by_name(span_exporter)
    first, _second = sorted(spans["claude-sonnet-4-5-20250929"], key=lambda s: s.start_time or 0)
    attrs = first.attributes or {}
    assert "gen_ai.request.max_tokens" not in attrs
    assert "lmnr.trajectory.source_span_id" not in attrs
    # Plain ATIF: the call ends at its step's timestamp; the tool is an instant.
    assert first.end_time == parse_timestamp_ns(trajectory["steps"][2]["timestamp"])
    tool = spans["get_order"][0]
    assert tool.start_time == tool.end_time
    agent = spans["agent"][0].attributes or {}
    assert not any(key.startswith("lmnr.trajectory.") for key in agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["tool_result_is_error", "is_error"])
async def test_plain_atif_tool_error_flags_mark_tool_spans_failed(
    span_exporter: InMemorySpanExporter, client: MagicMock, tmp_path: Path, flag: str
):
    # Harbor's claude-code adapter writes `tool_result_is_error`; kimi writes `is_error`.
    trajectory = {
        "agent": {"name": "claude-code", "model_name": "anthropic/claude-sonnet-4-5"},
        "steps": [
            {"step_id": 1, "timestamp": iso(6), "source": "user", "message": "go"},
            {
                "step_id": 2,
                "timestamp": iso(8),
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {"tool_call_id": "a", "function_name": "Bash", "arguments": {"cmd": "x"}},
                    {"tool_call_id": "b", "function_name": "Read", "arguments": {"path": "y"}},
                ],
                "observation": {
                    "results": [
                        {"source_call_id": "a", "content": "not found", "extra": {flag: True}},
                        {"source_call_id": "b", "content": "ok", "extra": {flag: False}},
                    ]
                },
            },
        ],
    }
    await run_trial_with_trajectory(client, tmp_path, trajectory)

    spans = spans_by_name(span_exporter)
    [event] = exception_events(spans["Bash"][0])
    assert event["exception.type"] == "ToolError"
    assert spans["Bash"][0].status.status_code == StatusCode.ERROR
    assert exception_events(spans["Read"][0]) == []


@pytest.mark.asyncio
async def test_plain_atif_tool_definitions_and_reasoning_effort_are_recorded(
    span_exporter: InMemorySpanExporter, client: MagicMock, tmp_path: Path
):
    tool = {"type": "function", "function": {"name": "bash", "parameters": {}}}
    trajectory = {
        "agent": {"name": "a", "model_name": "openai/gpt-5", "tool_definitions": [tool]},
        "steps": [
            {"step_id": 1, "timestamp": iso(6), "source": "user", "message": "go"},
            {
                "step_id": 2,
                "timestamp": iso(8),
                "source": "agent",
                "message": "done",
                "reasoning_effort": "high",
            },
        ],
    }
    await run_trial_with_trajectory(client, tmp_path, trajectory)

    attrs = spans_by_name(span_exporter)["gpt-5"][0].attributes or {}
    assert json.loads(attrs["gen_ai.tool.definitions"]) == [tool]
    assert attrs["gen_ai.request.reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_failed_and_retried_trials(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    job = FakeJob(tmp_path / "job", dataset_name=None)
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key", group_name="my-group")
    await plugin.on_job_start(job)
    assert client.evals.init.call_args.kwargs["group_name"] == "my-group"

    ok = make_event(job, task_dir, "ok", rewards={"reward": 1, "passed": True})
    await job.emit("start", ok)
    await job.emit("end", ok)

    exception = SimpleNamespace(
        exception_type="AgentTimeoutError",
        exception_message="timed out",
        exception_traceback="Traceback (most recent call last): ...",
        occurred_at=ts(19),
    )
    first = make_event(job, task_dir, "flaky", exception=exception)
    await job.emit("start", first)
    await job.emit("end", first)
    retry = make_event(job, task_dir, "flaky", rewards={"reward": 0.5})
    await job.emit("start", retry)
    await job.emit("end", retry)
    await plugin.on_job_end(SimpleNamespace())

    full = [dp for dp in saved_datapoints(client) if isinstance(dp, EvaluationResultDatapoint)]
    ok_dp, failed_dp, retried_dp = full
    assert ok_dp.scores == {"reward": 1, "passed": 1}
    # Harbor counts a trial without rewards as 0.
    assert failed_dp.scores == {"reward": 0, "passed": 0}
    assert (failed_dp.metadata or {})["exception_type"] == "AgentTimeoutError"
    # A retry overwrites the same datapoint with a new trace.
    assert retried_dp.id == failed_dp.id
    assert retried_dp.index == failed_dp.index == 1
    assert retried_dp.trace_id != failed_dp.trace_id
    assert retried_dp.scores == {"reward": 0.5}

    roots = spans_by_name(span_exporter)["flaky"]
    assert [r.status.status_code for r in roots] == [StatusCode.ERROR, StatusCode.UNSET]
    # The error lands on the executor span the datapoint links to.
    failed_agent, retried_agent = [
        s
        for s in spans_by_name(span_exporter)["agent"]
        if s.context.trace_id in {r.context.trace_id for r in roots}
    ]
    assert failed_agent.status.status_code == StatusCode.ERROR
    assert retried_agent.status.status_code == StatusCode.UNSET
    assert not retried_agent.events
    (event,) = failed_agent.events
    assert event.name == "exception"
    assert event.timestamp == parse_timestamp_ns(ts(19))
    assert event.attributes["exception.type"] == "AgentTimeoutError"
    assert event.attributes["exception.message"] == "timed out"
    assert event.attributes["exception.stacktrace"].startswith("Traceback")
    assert failed_dp.executor_span_id == uuid.UUID(int=failed_agent.context.span_id)


@pytest.mark.asyncio
async def test_unscored_trials_are_rescored_when_reward_keys_appear(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)

    crashed = make_event(job, task_dir, "crashed")
    await job.emit("start", crashed)
    await job.emit("end", crashed)
    passed = make_event(job, task_dir, "passed", rewards={"accuracy": 1})
    await job.emit("start", passed)
    await job.emit("end", passed)
    both = make_event(job, task_dir, "both", rewards={"accuracy": 0, "f1": 0.5})
    await job.emit("start", both)
    await job.emit("end", both)
    await plugin.on_job_end(SimpleNamespace())

    full = [dp for dp in saved_datapoints(client) if isinstance(dp, EvaluationResultDatapoint)]
    crashed_saves = [dp for dp in full if (dp.metadata or {})["trial_name"] == "crashed"]
    # Saved before any reward key was known, then re-saved with each new key.
    assert [dp.scores for dp in crashed_saves] == [
        {"reward": 0},
        {"accuracy": 0},
        {"accuracy": 0, "f1": 0},
    ]
    assert len({dp.id for dp in crashed_saves}) == 1
    assert len({dp.trace_id for dp in crashed_saves}) == 1
    # Scored trials are not re-saved.
    assert [(dp.metadata or {})["trial_name"] for dp in full].count("passed") == 1


@pytest.mark.asyncio
async def test_cancelled_trial_with_rewards_has_no_scores(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    event = make_event(job, task_dir, "t1", rewards={"reward": 1})
    await job.emit("start", event)
    await job.emit("cancel", event)
    await job.emit("end", event)
    # A cancelled trial's keys don't trigger re-saves either.
    later = make_event(job, task_dir, "t2")
    await job.emit("start", later)
    await job.emit("end", later)
    await plugin.on_job_end(SimpleNamespace())

    cancelled, later_dp = [
        dp for dp in saved_datapoints(client) if isinstance(dp, EvaluationResultDatapoint)
    ]
    assert cancelled.scores == {}
    assert (cancelled.metadata or {})["cancelled"] is True
    assert later_dp.scores == {"reward": 0}


@pytest.mark.asyncio
async def test_cancelled_trial_has_no_scores(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    event = make_event(job, task_dir, "t1")
    await job.emit("start", event)
    await job.emit("cancel", event)
    await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())

    full = saved_datapoints(client)[-1]
    assert full.scores == {}
    assert full.metadata["cancelled"] is True
    root = spans_by_name(span_exporter)["t1"][0]
    assert root.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_laminar_errors_do_not_break_the_job(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    client.evals.save_datapoints.side_effect = RuntimeError("network down")
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    event = make_event(job, task_dir, "t1", rewards={"reward": 1})
    await job.emit("start", event)
    await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())

    job = FakeJob(tmp_path / "job2")
    strict = LaminarPlugin(project_api_key="test_key", fail_fast="true")
    await strict.on_job_start(job)
    await job.emit("start", event)
    await job.emit("end", event)
    with pytest.raises(RuntimeError):
        await strict.on_job_end(SimpleNamespace())


@pytest.mark.asyncio
async def test_span_emission_failure_still_ends_root(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    event = make_event(job, task_dir, "t1", rewards={"reward": 1})
    await job.emit("start", event)
    with patch.object(plugin, "_emit_agent", side_effect=RuntimeError("bad trajectory")):
        await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())

    root = spans_by_name(span_exporter)["t1"][0]
    assert root.end_time is not None
    assert root.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_spans_are_exported_while_trial_runs(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    exception = SimpleNamespace(
        exception_type="AgentTimeoutError",
        exception_message="timed out",
        exception_traceback="Traceback ...",
        occurred_at=ts(19),
    )
    event = make_event(job, task_dir, "t1", exception=exception)
    write_trajectory(job, "t1")

    await job.emit("start", event)
    (marker,) = span_exporter.get_finished_spans()
    assert marker.name == "trial_started"
    assert marker.start_time == marker.end_time == parse_timestamp_ns(ts(0))
    await job.emit("agent_start", event)
    assert set(spans_by_name(span_exporter)) == {
        "trial_started",
        "environment_setup",
        "agent_setup",
    }
    await job.emit("verification_start", event)
    spans = spans_by_name(span_exporter)
    assert "agent" in spans and "t1" not in spans and "verifier" not in spans
    (agent,) = spans["agent"]
    assert agent.status.status_code == StatusCode.ERROR

    # The verifier then fails too: it gets its own error.
    event.result.verifier_result = None
    event.result.exception_info = SimpleNamespace(
        exception_type="VerifierTimeoutError",
        exception_message="verifier timed out",
        exception_traceback="Traceback ...",
        occurred_at=ts(28),
    )
    await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())

    spans = spans_by_name(span_exporter)
    # Nothing is emitted twice.
    assert {name: len(s) for name, s in spans.items()} == {
        "trial_started": 1,
        "environment_setup": 1,
        "agent_setup": 1,
        "agent": 1,
        "claude-sonnet-4-5": 2,
        "bash": 1,
        "verifier": 1,
        "t1": 1,
    }
    (verifier,) = spans["verifier"]
    assert verifier.status.status_code == StatusCode.ERROR
    assert verifier.events[0].attributes["exception.type"] == "VerifierTimeoutError"
    (root,) = spans["t1"]
    assert root.status.status_code == StatusCode.ERROR
    assert not root.events
    full = saved_datapoints(client)[-1]
    assert full.executor_span_id == uuid.UUID(int=agent.context.span_id)


@pytest.mark.asyncio
async def test_verifier_exception_without_verifier_span_goes_on_root(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    event = make_event(job, task_dir, "t1")
    event.result.verifier = None
    await job.emit("start", event)
    await job.emit("agent_start", event)
    await job.emit("verification_start", event)
    event.result.exception_info = SimpleNamespace(
        exception_type="RewardFileNotFoundError",
        exception_message="no reward",
        exception_traceback="Traceback ...",
        occurred_at=ts(28),
    )
    await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())

    spans = spans_by_name(span_exporter)
    (agent,) = spans["agent"]
    assert agent.status.status_code == StatusCode.UNSET
    (root,) = spans["t1"]
    assert root.events[0].attributes["exception.type"] == "RewardFileNotFoundError"


@pytest.mark.asyncio
async def test_retry_emits_setup_and_agent_again(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    event = make_event(job, task_dir, "t1", rewards={"reward": 1})
    for _ in range(2):
        await job.emit("start", event)
        await job.emit("agent_start", event)
        await job.emit("verification_start", event)
        await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())

    spans = spans_by_name(span_exporter)
    for name in ("trial_started", "environment_setup", "agent", "verifier", "t1"):
        assert len(spans[name]) == 2
    roots = {r.context.trace_id for r in spans["t1"]}
    assert {s.context.trace_id for s in spans["agent"]} == roots


@pytest.mark.asyncio
async def test_phase_spans_carry_setup_and_test_details(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    (task_dir / "task.toml").write_text(
        '[environment]\ndocker_image = "ubuntu:24.04"\ncpus = 1\nmemory_mb = 2048\n'
    )
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("pytest /tests")
    trial_dir = job.job_dir / "t1"
    (trial_dir / "agent" / "setup").mkdir(parents=True)
    (trial_dir / "agent" / "setup" / "install.log").write_text("installed")
    (trial_dir / "verifier").mkdir()
    (trial_dir / "verifier" / "test-stdout.txt").write_text("1 passed")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    event = make_event(job, task_dir, "t1", rewards={"reward": 1})
    await job.emit("start", event)
    await job.emit("end", event)
    await plugin.on_job_end(SimpleNamespace())

    spans = spans_by_name(span_exporter)
    attrs = spans["environment_setup"][0].attributes
    # A CLI resource override wins over task.toml.
    assert json.loads(attrs["lmnr.span.input"]) == {
        "type": "docker",
        "docker_image": "ubuntu:24.04",
        "cpus": 4,
        "memory_mb": 2048,
    }
    attrs = spans["agent_setup"][0].attributes
    assert json.loads(attrs["lmnr.span.input"]) == {
        "agent": "terminus-2",
        "model": "anthropic/claude-sonnet-4-5",
    }
    assert json.loads(attrs["lmnr.span.output"]) == {
        "version": "2.0.0",
        "logs": {"install.log": "installed"},
    }
    attrs = spans["verifier"][0].attributes
    assert json.loads(attrs["lmnr.span.input"]) == "pytest /tests"
    assert json.loads(attrs["lmnr.span.output"]) == {
        "rewards": {"reward": 1},
        "test_output": "1 passed",
    }


def test_fail_fast_reads_dotenv():
    with patch(
        "lmnr.integrations.harbor.plugin.from_env",
        side_effect=lambda key: "true" if key == "HARBOR_LAMINAR_FAIL_FAST" else None,
    ):
        assert LaminarPlugin(project_api_key="test_key").fail_fast is True


@pytest.mark.asyncio
async def test_missing_api_key_skips_reporting(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("LMNR_PROJECT_API_KEY", raising=False)
    with patch("lmnr.integrations.harbor.plugin.from_env", return_value=None):
        plugin = LaminarPlugin()
    job = FakeJob(tmp_path / "job")
    with patch.object(Laminar, "is_initialized", return_value=False):
        await plugin.on_job_start(job)
    assert job.hooks == {}
    client.evals.init.assert_not_awaited()
    await plugin.on_job_end(SimpleNamespace())


@pytest.mark.asyncio
async def test_already_initialized_laminar_with_another_key_is_rejected(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    # Spans would go to the initialized project, the evaluation to another one.
    job = FakeJob(tmp_path / "job")
    plugin = LaminarPlugin(project_api_key="other_project_key")
    await plugin.on_job_start(job)
    assert job.hooks == {}
    client.evals.init.assert_not_awaited()
    await plugin.on_job_end(SimpleNamespace())

    strict = LaminarPlugin(project_api_key="other_project_key", fail_fast=True)
    with pytest.raises(ValueError, match="different project API key"):
        await strict.on_job_start(FakeJob(tmp_path / "job2"))


@pytest.mark.asyncio
async def test_already_initialized_laminar_sets_evaluation_destination(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    with patch("lmnr.integrations.harbor.plugin.from_env", return_value=None):
        plugin = LaminarPlugin(base_url="https://elsewhere.example.com")
    with patch(
        "lmnr.integrations.harbor.plugin.AsyncLaminarClient", return_value=client
    ) as client_cls:
        await plugin.on_job_start(FakeJob(tmp_path / "job"))
    client_cls.assert_called_once_with(
        base_url=Laminar.get_base_http_url(),
        project_api_key=Laminar.get_project_api_key(),
        port=None,
    )
    # The logged link points at the host that received the evaluation.
    eval_id = client.evals.init.return_value["id"]
    project_id = client.evals.init.return_value["projectId"]
    http_url = Laminar.get_base_http_url()
    assert http_url is not None
    assert plugin.evaluation_url == get_evaluation_url(
        project_id, eval_id, http_url.rsplit(":", 1)[0]
    )
    assert "elsewhere.example.com" not in (plugin.evaluation_url or "")
    await plugin.on_job_end(SimpleNamespace())


@pytest.mark.asyncio
async def test_already_initialized_cloud_laminar_links_to_laminar_sh(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    with (
        patch.object(Laminar, "get_base_http_url", return_value="https://api.lmnr.ai:443"),
        patch("lmnr.integrations.harbor.plugin.from_env", return_value=None),
    ):
        plugin = LaminarPlugin()
        await plugin.on_job_start(FakeJob(tmp_path / "job"))
    assert (plugin.evaluation_url or "").startswith("https://www.laminar.sh/project/")
    await plugin.on_job_end(SimpleNamespace())


def test_entry_point_is_registered():
    from importlib.metadata import entry_points

    (ep,) = [ep for ep in entry_points(group="harbor.plugins") if ep.name == "laminar"]
    assert ep.load() is LaminarPlugin


def test_helpers():
    assert split_model_name("anthropic/claude-x") == ("anthropic", "claude-x")
    assert split_model_name("openrouter/openai/gpt-5") == ("openrouter", "openai/gpt-5")
    assert split_model_name("gpt-5") == (None, "gpt-5")
    assert split_model_name(None) == (None, None)
    assert (
        content_to_text(
            [
                {"type": "text", "text": "look"},
                {
                    "type": "image",
                    "source": {"media_type": "image/png", "path": "a.png"},
                },
            ]
        )
        == "look\n[image: a.png]"
    )
    assert parse_timestamp_ns("2026-01-01T12:00:00Z") == parse_timestamp_ns(T0)
    assert parse_timestamp_ns("not a date") is None


@pytest.mark.asyncio
async def test_retry_uploads_wait_for_earlier_attempt(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    import asyncio

    saved = []
    calls = []
    release_first = asyncio.Event()

    async def save(eval_id, datapoints, group_name):
        calls.append(datapoints)
        # The first attempt's partial upload is slow.
        if len(calls) == 1:
            await release_first.wait()
        saved.extend(datapoints)

    client.evals.save_datapoints.side_effect = save
    job = FakeJob(tmp_path / "job")
    task_dir = make_task_dir(tmp_path, "t")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)

    first = make_event(job, task_dir, "flaky", rewards={"reward": 0})
    await job.emit("start", first)
    await job.emit("end", first)
    retry = make_event(job, task_dir, "flaky", rewards={"reward": 1})
    await job.emit("start", retry)
    await job.emit("end", retry)
    await asyncio.sleep(0)
    release_first.set()
    await plugin.on_job_end(SimpleNamespace())

    assert len(saved) == 4
    assert isinstance(saved[-1], EvaluationResultDatapoint)
    assert saved[-1].scores == {"reward": 1}


@pytest.mark.asyncio
async def test_failed_evaluation_init_closes_client(
    span_exporter: InMemorySpanExporter,
    client: MagicMock,
    tmp_path: Path,
):
    client.evals.init.side_effect = RuntimeError("unauthorized")
    job = FakeJob(tmp_path / "job")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    assert job.hooks == {}
    client.close.assert_awaited_once()
    await plugin.on_job_end(SimpleNamespace())
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_does_not_take_over_global_tracer_provider(client, tmp_path: Path):
    # Libraries in the Harbor process that trace with the global provider (the
    # Daytona SDK) must not export a trace per call into the project.
    plugin = LaminarPlugin(project_api_key="test_key")
    with (
        patch.object(Laminar, "is_initialized", return_value=False),
        patch.object(Laminar, "initialize") as initialize,
    ):
        await plugin.on_job_start(FakeJob(tmp_path / "job"))
    assert initialize.call_args.kwargs["set_global_tracer_provider"] is False
    assert initialize.call_args.kwargs["instruments"] == set()
    await plugin.on_job_end(SimpleNamespace())


@pytest.mark.asyncio
async def test_agent_trajectory_path_does_not_make_the_plugin_local_only(client, tmp_path: Path):
    # LMNR_TRAJECTORY_PATH (env or .env) is for the agent; in the Harbor
    # process it would stop the plugin's own spans from being exported.
    plugin = LaminarPlugin(project_api_key="test_key")
    with (
        patch.object(Laminar, "is_initialized", return_value=False),
        patch.object(Laminar, "initialize") as initialize,
    ):
        await plugin.on_job_start(FakeJob(tmp_path / "job"))
    assert initialize.call_args.kwargs["write_trajectory"] is False
    await plugin.on_job_end(SimpleNamespace())


# -- agent trajectories (LMNR_TRAJECTORY_PATH) ------------------------------------


@pytest.mark.asyncio
async def test_agents_are_asked_to_write_their_trajectory(client, tmp_path: Path):
    job = FakeJob(tmp_path / "job")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    assert job.config.agents[0].env == {"LMNR_TRAJECTORY_PATH": AGENT_TRAJECTORY_PATH}
    await plugin.on_job_end(SimpleNamespace())


@pytest.mark.asyncio
async def test_agent_trajectory_path_set_by_the_user_wins(client, tmp_path: Path):
    job = FakeJob(tmp_path / "job")
    job.config.agents[0].env["LMNR_TRAJECTORY_PATH"] = "/logs/agent/mine.json"
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    assert job.config.agents[0].env["LMNR_TRAJECTORY_PATH"] == "/logs/agent/mine.json"
    await plugin.on_job_end(SimpleNamespace())


@pytest.mark.asyncio
async def test_no_agent_trajectory_without_trajectory_spans(client, tmp_path: Path):
    job = FakeJob(tmp_path / "job")
    plugin = LaminarPlugin(project_api_key="test_key", trajectory_spans="false")
    await plugin.on_job_start(job)
    assert job.config.agents[0].env == {}
    await plugin.on_job_end(SimpleNamespace())


@pytest.mark.asyncio
async def test_no_agent_trajectory_when_the_evaluation_fails_to_start(client, tmp_path: Path):
    # A local-only agent whose trajectory nobody reads would lose its spans.
    client.evals.init.side_effect = RuntimeError("down")
    job = FakeJob(tmp_path / "job")
    plugin = LaminarPlugin(project_api_key="test_key")
    await plugin.on_job_start(job)
    assert job.config.agents[0].env == {}


def test_agent_trajectory_reaches_harbor_agents(tmp_path: Path):
    """Against real Harbor: trial configs are built before `on_job_start` and
    the agent copies its env when the trial is created, after it."""
    pytest.importorskip("harbor")
    import importlib
    import logging

    def harbor(module: str) -> Any:
        return importlib.import_module(f"harbor.{module}")

    AgentFactory = harbor("agents.factory").AgentFactory
    JobPlan = harbor("job_plan").JobPlan
    JobConfig = harbor("models.job.config").JobConfig
    trial_config = harbor("models.trial.config")
    EnvironmentPaths = harbor("models.trial.paths").EnvironmentPaths

    assert AGENT_TRAJECTORY_PATH == str(EnvironmentPaths.agent_dir / "trajectory.json")
    config = JobConfig(agents=[trial_config.AgentConfig(name="nop")])
    trial_configs = JobPlan.build_trial_configs(
        config, [trial_config.TaskConfig(path=tmp_path)], job_id=uuid.uuid4()
    )
    _request_agent_trajectories(cast(Any, SimpleNamespace(config=config)))
    agent = AgentFactory.create_agent_from_config(
        trial_configs[0].agent, logs_dir=tmp_path / "logs", logger=logging.getLogger(__name__)
    )
    assert agent.extra_env["LMNR_TRAJECTORY_PATH"] == AGENT_TRAJECTORY_PATH

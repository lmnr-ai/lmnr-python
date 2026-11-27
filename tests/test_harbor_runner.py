"""Tests for the `EvalRunner` adapters (`lmnr.integrations.harbor.runner`).

They subclass Harbor's classes, so these tests need Harbor (Python 3.12+), which
isn't a dev dependency: they are skipped without it. Run them with
`uv run --python 3.13 --with harbor==0.25.0 pytest tests/test_harbor_runner.py`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typing_extensions import override

pytest.importorskip("harbor")

from harbor.agents.factory import AgentFactory  # pyright: ignore[reportMissingImports]
from harbor.models.agent.context import AgentContext  # pyright: ignore[reportMissingImports]
from harbor.models.task.config import TaskOS  # pyright: ignore[reportMissingImports]
from harbor.models.task.task import Task  # pyright: ignore[reportMissingImports]
from harbor.models.trial.config import (  # pyright: ignore[reportMissingImports]
    AgentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.models.trial.paths import TrialPaths  # pyright: ignore[reportMissingImports]
from harbor.verifier.factory import VerifierFactory  # pyright: ignore[reportMissingImports]

from lmnr import Laminar
from lmnr.evals import relay
from lmnr.integrations.harbor import runner as runner_module
from lmnr.integrations.harbor.runner import LaminarAgent, LaminarVerifier
from lmnr.integrations.harbor.spans import (
    LAMINAR_AGENT_IMPORT_PATH,
    LAMINAR_VERIFIER_IMPORT_PATH,
    verifier_spans,
)

EVAL_PY = """
from lmnr.evals import Artifact, EvalRunner, Image, secret

class SupportEval(EvalRunner):
    image = Image.from_registry("python:3.13")
    model = "anthropic/claude-haiku-4-5"
    secrets = ["APP_KEY", "JUDGE_KEY"]
    env = {"APP_KEY": secret("APP_KEY"), "TONE": "formal"}
    artifacts = [Artifact("/srv/app/state", destination="state")]

    async def setup(self, ctx):
        await ctx.container.run_file("seed.py")

    async def run(self, ctx):
        r = await ctx.container.sh("python -m app", env={"AGENT_MODEL": ctx.model})
        if ctx.case.metadata.get("fail"):
            raise RuntimeError("agent crashed")
        ctx.set_result(reply=r.stdout, model=ctx.model, tone=ctx.env["TONE"])

    async def collect(self, ctx):
        await ctx.container.sh("collect-state")

    async def verify(self, ctx):
        if ctx.case.metadata.get("fail"):
            raise KeyError("reply")
        await ctx.container.sh("check-db")
        ctx.logger.info(f"judge key: {ctx.secrets['JUDGE_KEY']}")
        if ctx.artifacts.exists("/srv/app/state/db.txt"):
            ctx.logger.info(f"state: {ctx.artifacts.read_text('/srv/app/state/db.txt')}")
        return {"replied": ctx.result().get("reply") == "hi", "model_ok": ctx.model == "openai/gpt-5"}
"""


# How `ctx.container.sh` runs a command: `relay.py exec` reading its payload file.
RELAY_EXEC = re.compile(r"^\{ rm -f (\S+) 2>/dev/null; .+ exec; \} < \S+$")


class FakeEnvironment:
    """Records commands; a `relay.py exec` is answered as the relay would and
    recorded as the command it runs."""

    os: TaskOS = TaskOS.LINUX

    def __init__(self) -> None:
        self.execs: list[dict[str, Any]] = []
        self.uploads: dict[str, str] = {}

    async def exec(self, command: str, cwd=None, env=None, timeout_sec=None, user=None) -> Any:
        relayed = RELAY_EXEC.match(command)
        if relayed:
            command = json.loads(self.uploads[relayed[1]])["cmd"]
        stdout = await self.run(command)
        self.execs.append({"command": command, "env": env, "user": user})
        if not relayed:
            return SimpleNamespace(stdout=stdout, stderr="", return_code=0)
        result = {"stdout": stdout, "stderr": "", "return_code": 0, "timed_out": False}
        return SimpleNamespace(
            stdout=relay.format_result({**result, "stdin_unlinked": True}),
            stderr="",
            return_code=0,
        )

    async def run(self, command: str) -> str:
        return "hi" if command == "python -m app" else ""

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploads[target_path] = Path(source_path).read_text()

    def is_path_mounted(self, environment_path: Any, host_path: Path) -> bool:
        return False

    def commands(self) -> list[str]:
        return [e["command"] for e in self.execs]


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "support"
    (root / "agent").mkdir(parents=True)
    _ = (root / "agent" / "eval.py").write_text(textwrap.dedent(EVAL_PY))
    _ = (root / "agent" / "seed.py").write_text("print('seed')")
    for name, metadata in (("ok", ""), ("broken", "[metadata]\nfail = true\n")):
        task = root / "tasks" / name
        (task / "tests").mkdir(parents=True)
        _ = (task / "task.toml").write_text(
            f'[task]\nname = "acme/{name}"\n\n[environment]\ndocker_image = "support-evals"\n\n'
            + metadata
        )
        _ = (task / "instruction.md").write_text("Handle the case in this task.")
        _ = (task / "tests" / "test.sh").write_text('#!/bin/bash\necho "graded by agent/eval.py"\n')
    # Harbor runs from the dataset folder; the runner path is relative to it.
    monkeypatch.chdir(root)
    monkeypatch.setenv("APP_KEY", "app-secret")
    monkeypatch.setenv("JUDGE_KEY", "judge-secret")
    return root


def make_trial(dataset: Path, task: str, model: str = "openai/gpt-5") -> TrialPaths:
    trial_paths = TrialPaths(dataset / "jobs" / "job" / f"{task}__1")
    trial_paths.mkdir()
    config = TrialConfig(
        task=TaskConfig(path=dataset / "tasks" / task),
        trial_name=trial_paths.trial_dir.name,
        trials_dir=trial_paths.trial_dir.parent,
        agent=AgentConfig(import_path=LAMINAR_AGENT_IMPORT_PATH, model_name=model),
    )
    _ = trial_paths.config_path.write_text(config.model_dump_json())
    return trial_paths


def make_agent(trial_paths: TrialPaths, env: dict[str, str]) -> LaminarAgent:
    agent = AgentFactory.create_agent_from_config(
        AgentConfig(
            import_path=LAMINAR_AGENT_IMPORT_PATH,
            kwargs={"runner": "agent/eval.py:SupportEval"},
            model_name="openai/gpt-5",
            env=env,
        ),
        logs_dir=trial_paths.agent_dir,
        logger=logging.getLogger(__name__),
    )
    assert isinstance(agent, LaminarAgent)
    return agent


def make_verifier(dataset: Path, task: str, trial_paths: TrialPaths, env: FakeEnvironment):
    verifier = VerifierFactory.create_verifier_from_config(
        VerifierConfig(
            import_path=LAMINAR_VERIFIER_IMPORT_PATH,
            kwargs={"runner": "agent/eval.py:SupportEval"},
        ),
        task=Task(dataset / "tasks" / task),
        trial_paths=trial_paths,
        environment=env,  # pyright: ignore[reportArgumentType]
        logger=logging.getLogger(__name__),
    )
    assert isinstance(verifier, LaminarVerifier)
    return verifier


@pytest.mark.asyncio
async def test_agent_runs_setup_run_collect(dataset: Path):
    trial_paths = make_trial(dataset, "ok")
    # lmnr-cli writes the runner's env into agents[].env; Harbor sets it on
    # every agent-phase command, so the runner must not add it again.
    agent = make_agent(trial_paths, env={"APP_KEY": "${APP_KEY}", "TONE": "formal"})
    assert agent.name() == "laminar-eval-runner"
    env = FakeEnvironment()
    await agent.install(env)
    assert env.execs == []

    await agent.run("Handle the case in this task.", env, AgentContext())

    commands = env.commands()
    seed = next(c for c in commands if c.startswith("python3 /tmp/lmnr-"))
    assert commands.index(seed) < commands.index("python -m app") < commands.index("collect-state")
    run = next(e for e in env.execs if e["command"] == "python -m app")
    assert run["env"] == {"AGENT_MODEL": "openai/gpt-5"}
    result = json.loads((trial_paths.agent_dir / "result.json").read_text())
    assert result == {"reply": "hi", "model": "openai/gpt-5", "tone": "formal"}

    context = AgentContext()
    agent.populate_context_post_run(context)
    assert context.metadata == result


@pytest.mark.asyncio
async def test_agent_adds_runner_env_harbor_does_not_set(dataset: Path):
    trial_paths = make_trial(dataset, "ok")
    agent = make_agent(trial_paths, env={})  # e.g. a hand-written harbor.yaml
    env = FakeEnvironment()
    await agent.run("", env, AgentContext())
    run = next(e for e in env.execs if e["command"] == "python -m app")
    assert run["env"] == {"APP_KEY": "app-secret", "TONE": "formal", "AGENT_MODEL": "openai/gpt-5"}


@pytest.mark.asyncio
async def test_collect_runs_when_run_fails(dataset: Path):
    trial_paths = make_trial(dataset, "broken")
    agent = make_agent(trial_paths, env={})
    env = FakeEnvironment()
    with pytest.raises(RuntimeError, match="agent crashed"):
        await agent.run("", env, AgentContext())
    assert env.commands()[-1] == "collect-state"


class HangingCollectEnvironment(FakeEnvironment):
    @override
    async def run(self, command: str) -> str:
        if command == "collect-state":
            await asyncio.Event().wait()
        return await super().run(command)


@pytest.mark.asyncio
async def test_run_error_wins_over_a_timeout_during_collect(dataset: Path):
    trial_paths = make_trial(dataset, "broken")
    agent = make_agent(trial_paths, env={})
    # Harbor bounds the agent phase with wait_for: here it fires during collect.
    with pytest.raises(RuntimeError, match="agent crashed"):
        await asyncio.wait_for(
            agent.run("", HangingCollectEnvironment(), AgentContext()), timeout=0.5
        )


@pytest.mark.asyncio
async def test_verifier_grades_in_the_harbor_process(dataset: Path):
    trial_paths = make_trial(dataset, "ok")
    trial_paths.agent_dir.mkdir(parents=True, exist_ok=True)
    _ = (trial_paths.agent_dir / "result.json").write_text('{"reply": "hi"}')
    # Harbor collected the declared artifact under its destination.
    (trial_paths.artifacts_dir / "state").mkdir(parents=True)
    _ = (trial_paths.artifacts_dir / "state" / "db.txt").write_text("refunded")
    env = FakeEnvironment()
    verifier = make_verifier(dataset, "ok", trial_paths, env)

    result = await verifier.verify()

    assert result.rewards == {"replied": 1, "model_ok": 1}
    check = next(e for e in env.execs if e["command"] == "check-db")
    # The verifier container gets the runner's env too, but no undeclared values.
    assert check["env"] == {"APP_KEY": "app-secret", "TONE": "formal"}
    assert json.loads(trial_paths.reward_json_path.read_text()) == result.rewards
    assert json.loads(env.uploads["/logs/verifier/reward.json"]) == result.rewards
    log = trial_paths.test_stdout_path.read_text()
    assert "graded by SupportEval.verify() in agent/eval.py:SupportEval" in log
    assert "judge key: judge-secret" in log
    assert "state: refunded" in log
    assert 'rewards: {"replied": 1, "model_ok": 1}' in log


@pytest.mark.asyncio
async def test_verify_errors_propagate_with_a_log(dataset: Path):
    trial_paths = make_trial(dataset, "broken")
    verifier = make_verifier(dataset, "broken", trial_paths, FakeEnvironment())
    with pytest.raises(KeyError, match="reply"):
        _ = await verifier.verify()
    assert "KeyError: 'reply'" in trial_paths.test_stdout_path.read_text()
    assert not trial_paths.reward_json_path.exists()


@pytest.mark.asyncio
async def test_verify_is_traced_under_the_plugin_verifier_span(
    dataset: Path, span_exporter: InMemorySpanExporter
):
    trial_paths = make_trial(dataset, "ok")
    verifier = make_verifier(dataset, "ok", trial_paths, FakeEnvironment())
    parent = Laminar.start_span("verifier")
    verifier_spans[trial_paths.trial_dir.name] = parent.get_laminar_span_context()
    try:
        _ = await verifier.verify()
    finally:
        del verifier_spans[trial_paths.trial_dir.name]
        parent.end()
    (verify,) = [s for s in span_exporter.get_finished_spans() if s.name == "verify"]
    assert verify.parent is not None
    assert verify.parent.span_id == parent.get_span_context().span_id
    assert json.loads(str((verify.attributes or {})["lmnr.span.output"])) == {
        "replied": 0,
        "model_ok": 1,
    }


SERVICES_EVAL_PY = """
from lmnr.evals import EvalRunner, Image, Service

class Recorder(Service):
    # Runs marker commands instead of a real server.
    async def start(self, ctx):
        await ctx.container.sh(f"start {self.name}", user="root")

    async def migrate(self, ctx):
        await ctx.container.sh(f"migrate {self.name}", user="root")

    async def seed(self, ctx):
        await ctx.container.sh(f"seed {self.name}", user="root")

    async def snapshot(self, ctx, out_dir):
        await ctx.container.sh(f"snapshot {self.name} {out_dir}", user="root")

class ServicesEval(EvalRunner):
    image = Image.from_registry("python:3.13")
    env = {"TONE": "formal"}
    services = [Recorder("db", "1", env={"DB_URL": "db://{host}"})]

    async def setup(self, ctx):
        await ctx.container.sh("setup")

    async def run(self, ctx):
        await ctx.container.sh("python -m app")
        ctx.set_result(db=ctx.env["DB_URL"])

    async def verify(self, ctx):
        await ctx.container.sh("check-db")
        return {"mode": ctx.verifier_mode == "separate"}
"""


@pytest.fixture
def services_dataset(dataset: Path) -> Path:
    _ = (dataset / "agent" / "eval.py").write_text(textwrap.dedent(SERVICES_EVAL_PY))
    return dataset


def services_agent(trial_paths: TrialPaths) -> LaminarAgent:
    agent = AgentFactory.create_agent_from_config(
        AgentConfig(
            import_path=LAMINAR_AGENT_IMPORT_PATH,
            kwargs={"runner": "agent/eval.py:ServicesEval"},
            env={"TONE": "formal"},
        ),
        logs_dir=trial_paths.agent_dir,
        logger=logging.getLogger(__name__),
    )
    assert isinstance(agent, LaminarAgent)
    return agent


@pytest.mark.asyncio
async def test_services_start_before_setup_and_snapshot_after_collect(services_dataset: Path):
    trial_paths = make_trial(services_dataset, "ok")
    env = FakeEnvironment()
    await services_agent(trial_paths).run("", env, AgentContext())
    assert env.commands() == [
        "start db",
        "migrate db",
        "seed db",
        "setup",
        "python -m app",
        "snapshot db /logs/artifacts/state/db",
    ]
    # Every later command gets the services' env (Harbor sets TONE itself).
    run = next(e for e in env.execs if e["command"] == "python -m app")
    assert run["env"] == {"DB_URL": "db://127.0.0.1"}
    result = json.loads((trial_paths.agent_dir / "result.json").read_text())
    assert result == {"db": "db://127.0.0.1"}


@pytest.mark.asyncio
async def test_snapshots_run_when_run_fails(services_dataset: Path):
    trial_paths = make_trial(services_dataset, "broken")
    _ = (services_dataset / "agent" / "eval.py").write_text(
        textwrap.dedent(SERVICES_EVAL_PY).replace(
            'ctx.set_result(db=ctx.env["DB_URL"])', 'raise RuntimeError("agent crashed")'
        )
    )
    env = FakeEnvironment()
    with pytest.raises(RuntimeError, match="agent crashed"):
        await services_agent(trial_paths).run("", env, AgentContext())
    assert env.commands()[-1] == "snapshot db /logs/artifacts/state/db"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["shared", "separate"])
async def test_verifier_mode_decides_service_env(services_dataset: Path, mode: str):
    task_toml = services_dataset / "tasks" / "ok" / "task.toml"
    _ = task_toml.write_text(task_toml.read_text() + f'\n[verifier]\nenvironment_mode = "{mode}"\n')
    trial_paths = make_trial(services_dataset, "ok")
    env = FakeEnvironment()
    verifier = VerifierFactory.create_verifier_from_config(
        VerifierConfig(
            import_path=LAMINAR_VERIFIER_IMPORT_PATH,
            kwargs={"runner": "agent/eval.py:ServicesEval"},
        ),
        task=Task(services_dataset / "tasks" / "ok"),
        trial_paths=trial_paths,
        environment=env,  # pyright: ignore[reportArgumentType]
        logger=logging.getLogger(__name__),
    )
    result = await verifier.verify()
    assert result.rewards == {"mode": int(mode == "separate")}
    check = next(e for e in env.execs if e["command"] == "check-db")
    # Services only run in the agent container, which a shared verifier is.
    expected = (
        {"TONE": "formal", "DB_URL": "db://127.0.0.1"} if mode == "shared" else {"TONE": "formal"}
    )
    assert check["env"] == expected


MOCKS_EVAL_PY = (
    SERVICES_EVAL_PY.replace(
        "from lmnr.evals import EvalRunner, Image, Service",
        """from lmnr.evals import EvalRunner, Image, Service, mock

def get_customer(call, ctx):
    return {"id": 1}

def zendesk(req, ctx):
    return None
""",
    )
    .replace(
        '    services = [Recorder("db", "1", env={"DB_URL": "db://{host}"})]',
        """    services = [Recorder("db", "1", env={"DB_URL": "db://{host}"})]
    mocks = {
        "crm.get_customer": get_customer,
        "https://api.zendesk.com": mock.http(zendesk, base_url_env="ZENDESK_URL"),
    }""",
    )
    .replace(
        '        return {"mode": ctx.verifier_mode == "separate"}',
        '        return {"lookups": len(ctx.mocks.calls("crm.get_customer"))}',
    )
)


@pytest.fixture
def mocks_dataset(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The mock server's start and collect replaced by marker commands (the
    lifecycle itself: `tests/test_evals_mocks.py`)."""
    _ = (dataset / "agent" / "eval.py").write_text(textwrap.dedent(MOCKS_EVAL_PY))

    async def start(ctx: Any, specs: Any, *, eval_path: Path) -> Any:
        assert [s.ref for s in specs] == ["__runner__:get_customer", "__runner__:zendesk"]
        assert eval_path == dataset / "agent" / "eval.py"
        # Services are up: their env reaches the server's config.
        assert ctx.env["DB_URL"] == "db://127.0.0.1"
        await ctx.container.sh("start mocks", user="root")
        return runner_module.RunningMocks(
            pid=7,
            url="http://127.0.0.1:1/_lmnr/v1/token",
            env={"LMNR_MOCK_URL": "http://127.0.0.1:1/_lmnr/v1/token", "ZENDESK_URL": "z"},
        )

    async def collect(ctx: Any, running: Any) -> None:
        await ctx.container.sh(f"collect mocks {running.pid}", user="root")

    monkeypatch.setattr(runner_module, "start_mock_server", start)
    monkeypatch.setattr(runner_module, "collect_mocks", collect)
    return dataset


@pytest.mark.asyncio
async def test_mock_server_starts_after_services_and_is_collected_last(mocks_dataset: Path):
    trial_paths = make_trial(mocks_dataset, "ok")
    env = FakeEnvironment()
    await services_agent(trial_paths).run("", env, AgentContext())
    assert env.commands() == [
        "start db",
        "migrate db",
        "seed db",
        "start mocks",
        "setup",
        "python -m app",
        "snapshot db /logs/artifacts/state/db",
        "collect mocks 7",
    ]
    run = next(e for e in env.execs if e["command"] == "python -m app")
    assert run["env"] == {
        "DB_URL": "db://127.0.0.1",
        "LMNR_MOCK_URL": "http://127.0.0.1:1/_lmnr/v1/token",
        "ZENDESK_URL": "z",
    }


@pytest.mark.asyncio
async def test_mock_calls_are_collected_when_run_fails(mocks_dataset: Path):
    trial_paths = make_trial(mocks_dataset, "broken")
    eval_py = mocks_dataset / "agent" / "eval.py"
    _ = eval_py.write_text(
        eval_py.read_text().replace(
            'ctx.set_result(db=ctx.env["DB_URL"])', 'raise RuntimeError("agent crashed")'
        )
    )
    env = FakeEnvironment()
    with pytest.raises(RuntimeError, match="agent crashed"):
        await services_agent(trial_paths).run("", env, AgentContext())
    assert env.commands()[-2:] == ["snapshot db /logs/artifacts/state/db", "collect mocks 7"]


def _with_setup(dataset: Path, body: str) -> None:
    """eval.py's setup replaced by `body`, plus a collect() that leaves a marker."""
    eval_py = dataset / "agent" / "eval.py"
    _ = eval_py.write_text(
        eval_py.read_text().replace(
            '        await ctx.container.sh("setup")\n',
            f"        {body}\n\n"
            + "    async def collect(self, ctx):\n"
            + '        await ctx.container.sh("user collect")\n',
        )
    )


@pytest.mark.asyncio
async def test_mock_server_is_collected_when_setup_fails(mocks_dataset: Path):
    """collect() only follows a finished setup; the runner's own cleanup always runs."""
    trial_paths = make_trial(mocks_dataset, "broken")
    _with_setup(mocks_dataset, 'raise RuntimeError("seed failed")')
    env = FakeEnvironment()
    with pytest.raises(RuntimeError, match="seed failed"):
        await services_agent(trial_paths).run("", env, AgentContext())
    assert env.commands()[-3:] == [
        "start mocks",
        "snapshot db /logs/artifacts/state/db",
        "collect mocks 7",
    ]


@pytest.mark.asyncio
async def test_mock_server_is_collected_when_setup_times_out(mocks_dataset: Path):
    """Harbor cancels the agent on timeout and still runs the verifier, which
    reads the call log."""
    trial_paths = make_trial(mocks_dataset, "slow")
    _with_setup(mocks_dataset, "await asyncio.sleep(30)")
    eval_py = mocks_dataset / "agent" / "eval.py"
    _ = eval_py.write_text("import asyncio\n" + eval_py.read_text())
    env = FakeEnvironment()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(services_agent(trial_paths).run("", env, AgentContext()), 0.5)
    assert "user collect" not in env.commands()
    assert env.commands()[-2:] == ["snapshot db /logs/artifacts/state/db", "collect mocks 7"]


@pytest.mark.asyncio
async def test_verifier_reads_the_mock_call_log(mocks_dataset: Path):
    trial_paths = make_trial(mocks_dataset, "ok")
    log = trial_paths.artifacts_dir / "logs" / "artifacts" / "mocks.jsonl"
    log.parent.mkdir(parents=True)
    _ = log.write_text(
        '{"key": "crm.get_customer", "outcome": "respond"}\n'
        + '{"key": "https://api.zendesk.com", "outcome": "unmatched"}\n'
    )
    verifier = VerifierFactory.create_verifier_from_config(
        VerifierConfig(
            import_path=LAMINAR_VERIFIER_IMPORT_PATH,
            kwargs={"runner": "agent/eval.py:ServicesEval"},
        ),
        task=Task(mocks_dataset / "tasks" / "ok"),
        trial_paths=trial_paths,
        environment=FakeEnvironment(),  # pyright: ignore[reportArgumentType]
        logger=logging.getLogger(__name__),
    )
    assert (await verifier.verify()).rewards == {"lookups": 1}

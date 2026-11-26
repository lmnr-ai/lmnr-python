"""Tests for `lmnr.evals` (EvalRunner, ctx, describe). No Harbor needed: a fake
environment records the `exec` / `upload` calls `ctx` makes."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, patch

import pytest

from lmnr.evals import (
    AgentContext,
    Case,
    CommandError,
    EvalConfigError,
    EvalRunner,
    Image,
    VerifyContext,
    http,
    port,
    secret,
)
from lmnr.evals.context import Secrets, normalize_rewards
from lmnr.evals.describe import describe
from lmnr.evals.runner import find_runner, load_module, load_runner


class FakeEnvironment:
    """Records calls; `responder(command)` returns `(return_code, stdout)`."""

    def __init__(self, responder: Callable[[str], tuple[int, str]] | None = None):
        self.calls: list[tuple[str, Any]] = []
        self.files: dict[str, bytes] = {}
        self.responder: Callable[[str], tuple[int, str]] = responder or (lambda _: (0, ""))

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> Any:
        self.calls.append(
            (
                "exec",
                {"command": command, "cwd": cwd, "env": env, "timeout": timeout_sec, "user": user},
            )
        )
        code, stdout = self.responder(command)
        return SimpleNamespace(stdout=stdout, stderr="boom" if code else None, return_code=code)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.calls.append(("upload_file", (Path(source_path).name, target_path)))
        self.files[target_path] = Path(source_path).read_bytes()

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        self.calls.append(("upload_dir", (str(source_dir), target_dir)))

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        self.calls.append(("download_file", source_path))
        _ = Path(target_path).write_bytes(self.files[source_path])

    def commands(self) -> list[str]:
        return [args["command"] for kind, args in self.calls if kind == "exec"]

    def execs(self) -> list[dict[str, Any]]:
        return [args for kind, args in self.calls if kind == "exec"]


def make_case(tmp_path: Path, name: str = "refund") -> Case:
    task_dir = tmp_path / "tasks" / name
    (task_dir / "tests" / "fixtures").mkdir(parents=True)
    (task_dir / "environment").mkdir()
    _ = (task_dir / "task.toml").write_text(
        f'[task]\nname = "acme/{name}"\n\n[metadata]\ndifficulty = "easy"\n'
    )
    _ = (task_dir / "instruction.md").write_text("Handle it.")
    _ = (task_dir / "tests" / "expected.json").write_text('{"refund_count": 1}')
    _ = (task_dir / "tests" / "fixtures" / "crm.json").write_text('{"a": 1}')
    return Case.from_dir(task_dir)


def agent_ctx(tmp_path: Path, env: FakeEnvironment, **kwargs: Any) -> AgentContext:
    runner_dir = tmp_path / "agent"
    runner_dir.mkdir(exist_ok=True)
    return AgentContext(
        environment=env,
        case=make_case(tmp_path),
        runner_dir=runner_dir,
        host_logs_dir=tmp_path / "trial" / "agent",
        **kwargs,
    )


def write_eval(tmp_path: Path, body: str, dataset: str = "support") -> Path:
    path = tmp_path / dataset / "agent" / "eval.py"
    path.parent.mkdir(parents=True)
    _ = path.write_text(textwrap.dedent(body))
    return path


SUPPORT_EVAL = """
    from lmnr.evals import EvalRunner, Image, secret

    print("module output must not reach the JSON")

    class SupportEval(EvalRunner):
        image = Image.from_dockerfile("Dockerfile", context="../..")
        model = "anthropic/claude-sonnet-5-5"
        secrets = ["ANTHROPIC_API_KEY", "JUDGE_API_KEY"]
        env = {"ANTHROPIC_API_KEY": secret("ANTHROPIC_API_KEY"), "SUPPORT_TONE": "formal"}
        task_template = {"agent": {"timeout_sec": 300}}

        async def run(self, ctx):
            ctx.result(reply="hi")

        async def verify(self, ctx):
            return 1.0
"""


# Configuration


def runner_class(**attrs: Any) -> type[EvalRunner]:
    async def run(self: Any, ctx: Any) -> None:
        pass

    async def verify(self: Any, ctx: Any) -> float:
        return 1.0

    namespace: dict[str, Any] = {"image": Image.from_registry("python:3.13"), "run": run}
    namespace["verify"] = verify
    namespace.update(attrs)
    return type("MyEval", (EvalRunner,), namespace)


def test_valid_runner_has_no_warnings():
    assert runner_class().validate() == []


@pytest.mark.parametrize(
    ("attrs", "message"),
    [
        ({"image": None}, "has no image"),
        ({"image": {}}, "empty dict"),
        ({"image": "python:3.13"}, "must be an Image"),
        ({"run": EvalRunner.run}, "must define `async def run"),
        ({"verify": EvalRunner.verify}, "must define `async def verify"),
        ({"verify": "tests"}, 'must be a method or "task"'),
        ({"env": {"KEY": secret("KEY")}}, "not in MyEval.secrets"),
        ({"secrets": ["KEY=value"]}, "env var names"),
        ({"env": {"KEY": 1}}, "must be a string or secret"),
        ({"services": ["postgres"]}, "services is not supported yet"),
        ({"mocks": {"crm.get": print}}, "mocks is not supported yet"),
    ],
)
def test_invalid_runner_raises_actionable_error(attrs: dict[str, Any], message: str):
    with pytest.raises(EvalConfigError, match=message):
        _ = runner_class(**attrs).validate()


def test_entrypoint_replaces_run_and_task_verify_replaces_verify():
    runner = runner_class(run=EvalRunner.run, entrypoint="python app.py", verify="task")
    assert runner.validate() == []
    assert runner.verify_mode() == "task"
    assert runner_class(entrypoint="python app.py").validate() == [
        "MyEval defines run(), so `entrypoint` is ignored."
    ]


def test_resolved_env_reads_secrets_and_skips_unset_ones():
    runner = runner_class(
        secrets=["A", "B"], env={"A_KEY": secret("A"), "B_KEY": secret("B"), "TONE": "formal"}
    )
    assert runner.resolved_env({"A": "a-value"}) == {"A_KEY": "a-value", "TONE": "formal"}


def test_describe_outputs_json_for_lmnr_cli(tmp_path: Path):
    path = write_eval(tmp_path, SUPPORT_EVAL)
    assert describe(path) == {
        "version": 1,
        "runner": "SupportEval",
        "images": {
            "support-evals": {
                "kind": "dockerfile",
                "dockerfile": str((path.parent / "Dockerfile").resolve()),
                "context": str(tmp_path.resolve()),
            }
        },
        "default_image": "support-evals",
        "model": "anthropic/claude-sonnet-5-5",
        "env": {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}", "SUPPORT_TONE": "formal"},
        "secrets": ["ANTHROPIC_API_KEY", "JUDGE_API_KEY"],
        "task_template": {"agent": {"timeout_sec": 300}},
        "verify": "runner",
        "warnings": [],
        "modules": ["support/agent/eval.py"],
    }


def test_describe_named_images(tmp_path: Path):
    path = write_eval(
        tmp_path,
        """
        from lmnr.evals import EvalRunner, Image

        class E(EvalRunner):
            image = {"new": Image.from_registry("python:3.14"), "old": Image.from_dockerfile("old/Dockerfile")}
            entrypoint = "true"
            verify = "task"
        """,
    )
    result = describe(path)
    assert result["images"] == {
        "new": {"kind": "registry", "ref": "python:3.14"},
        "old": {
            "kind": "dockerfile",
            "dockerfile": str((path.parent / "old" / "Dockerfile").resolve()),
            "context": str((path.parent / "old").resolve()),
        },
    }
    assert result["default_image"] == "new"
    assert result["verify"] == "task"


def test_describe_cli_prints_json_and_reports_errors(tmp_path: Path):
    path = write_eval(tmp_path, SUPPORT_EVAL)
    ok = subprocess.run(
        [sys.executable, "-m", "lmnr.evals.describe", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["runner"] == "SupportEval"
    assert "module output" in ok.stderr

    bad = write_eval(
        tmp_path,
        """
        from lmnr.evals import EvalRunner
        class E(EvalRunner):
            pass
        """,
        dataset="bad",
    )
    failed = subprocess.run(
        [sys.executable, "-m", "lmnr.evals.describe", str(bad)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 2
    assert "has no image" in failed.stderr
    assert failed.stdout == ""


# Shared runners: evals/_shared/ (main doc 2.1.2)

SHARED_RUNNER = """
    from lmnr.evals import EvalRunner, Image

    class SupportRunner(EvalRunner):
        image = Image.from_dockerfile("Dockerfile", context="..")
        env = {"TONE": "formal"}

        async def setup(self, ctx):
            await ctx.run_file("start.sh")
            await ctx.copy("fixtures", "/srv/fixtures")

        async def run(self, ctx):
            pass

        async def verify(self, ctx):
            return 1
"""


@pytest.fixture
def shared_evals(tmp_path: Path):
    """`evals/_shared/support.py` and a dataset whose eval.py subclasses it."""
    root = tmp_path / "evals"
    shared = root / "_shared"
    (shared / "fixtures").mkdir(parents=True)
    _ = (shared / "support.py").write_text(textwrap.dedent(SHARED_RUNNER))
    _ = (shared / "start.sh").write_text("echo start")
    _ = (shared / "fixtures" / "crm.json").write_text("{}")
    path = write_eval(
        root,
        """
        from _shared.support import SupportRunner

        class RefundsEval(SupportRunner):
            env = {**SupportRunner.env, "QUEUE": "refunds"}
        """,
        dataset="refunds",
    )
    saved_path, saved_modules = list(sys.path), set(sys.modules)
    yield path
    # Another test's evals root has its own `_shared`.
    sys.path[:] = saved_path
    for name in set(sys.modules) - saved_modules:
        del sys.modules[name]


def test_shared_runner_paths_resolve_from_its_own_file(shared_evals: Path):
    root = shared_evals.parent.parent.parent
    result = describe(shared_evals)
    assert result["runner"] == "RefundsEval"
    # The shared module's Dockerfile, not refunds/agent/Dockerfile.
    assert result["images"] == {
        "refunds-evals": {
            "kind": "dockerfile",
            "dockerfile": str((root / "_shared" / "Dockerfile").resolve()),
            "context": str(root.resolve()),
        }
    }
    assert result["env"] == {"TONE": "formal", "QUEUE": "refunds"}
    assert result["modules"] == ["_shared/support.py", "refunds/agent/eval.py"]


@pytest.mark.asyncio
async def test_shared_runner_files_resolve_from_its_own_file(shared_evals: Path, tmp_path: Path):
    runner_cls, eval_path = load_runner(str(shared_evals))
    env = FakeEnvironment()
    ctx = AgentContext(
        environment=env,
        case=make_case(tmp_path),
        runner_dir=eval_path.parent,
        host_logs_dir=tmp_path / "trial" / "agent",
    )
    await runner_cls().setup(ctx)
    shared = eval_path.parent.parent.parent / "_shared"
    assert [c for c in env.calls if c[0] != "exec"] == [
        ("upload_file", ("start.sh", ANY)),
        ("upload_dir", (str(shared / "fixtures"), "/srv/fixtures")),
    ]


@pytest.mark.asyncio
async def test_run_file_from_eval_py_resolves_from_its_dir(tmp_path: Path):
    path = write_eval(
        tmp_path,
        """
        from lmnr.evals import EvalRunner, Image

        class E(EvalRunner):
            image = Image.from_registry("python:3.13")

            async def setup(self, ctx):
                await ctx.run_file("seed.py")

            async def run(self, ctx):
                pass

            async def verify(self, ctx):
                return 1
        """,
    )
    _ = (path.parent / "seed.py").write_text("print('seed')")
    runner_cls, eval_path = load_runner(str(path))
    env = FakeEnvironment()
    ctx = AgentContext(
        environment=env,
        case=make_case(tmp_path),
        runner_dir=eval_path.parent,
        host_logs_dir=tmp_path / "trial" / "agent",
    )
    await runner_cls().setup(ctx)
    assert [args[0] for kind, args in env.calls if kind == "upload_file"] == ["seed.py"]


def test_find_runner_needs_exactly_one_subclass(tmp_path: Path):
    path = write_eval(
        tmp_path,
        """
        from lmnr.evals import EvalRunner
        class A(EvalRunner): pass
        class B(EvalRunner): pass
        """,
    )
    module = load_module(path)
    with pytest.raises(EvalConfigError, match=r"several EvalRunner subclasses \(A, B\)"):
        _ = find_runner(module)
    assert find_runner(module, "B").__name__ == "B"
    runner, resolved = load_runner("support/agent/eval.py:A", base_dir=tmp_path)
    assert runner.__name__ == "A"
    assert resolved == path.resolve()
    assert load_module(path) is module  # cached per process


def test_case_reads_task_dir(tmp_path: Path):
    case = make_case(tmp_path)
    assert case.name == "acme/refund"
    assert case.metadata == {"difficulty": "easy"}
    assert case.instruction == "Handle it."
    assert case.expected() == {"refund_count": 1}
    assert case.fixture_json("crm.json") == {"a": 1}
    assert case.file("ticket.json") == case.dir / "environment" / "ticket.json"
    # Data files are not seed scripts.
    _ = case.file("seed.json").write_text("{}")
    assert case.seed_file() is None
    _ = case.file("seed.sh").write_text("echo seeded")
    assert case.seed_file() == case.file("seed.sh")


def test_normalize_rewards():
    assert normalize_rewards(0.5) == {"reward": 0.5}
    assert normalize_rewards(True) == {"reward": 1}
    assert normalize_rewards({"refund": True, "reply": 0}) == {"refund": 1, "reply": 0}
    with pytest.raises(EvalConfigError):
        _ = normalize_rewards({"reply": "yes"})
    with pytest.raises(EvalConfigError):
        _ = normalize_rewards(float("nan"))


def test_secrets_only_expose_declared_names():
    secrets = Secrets(["JUDGE"], {"JUDGE": "j", "OTHER": "o"})
    assert secrets["JUDGE"] == "j"
    assert dict(secrets) == {"JUDGE": "j"}
    with pytest.raises(KeyError, match="not in the runner's `secrets`"):
        _ = secrets["OTHER"]
    with pytest.raises(KeyError, match="not set"):
        _ = Secrets(["MISSING"], {})["MISSING"]


# Commands


@pytest.mark.asyncio
async def test_sh_adds_runner_env_and_maps_users(tmp_path: Path):
    env = FakeEnvironment()
    ctx = agent_ctx(tmp_path, env, exec_env={"TONE": "formal"})
    result = await ctx.sh("echo hi", env={"EXTRA": "1"}, timeout=2.5)
    _ = await ctx.sh("whoami", user="root")
    first, second = env.execs()
    assert first == {
        "command": "echo hi",
        "cwd": None,
        "env": {"TONE": "formal", "EXTRA": "1"},
        "timeout": 3,
        "user": None,  # the task's agent user
    }
    assert second["user"] == "root"
    assert result.return_code == 0
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_sh_check_raises_with_output_tail(tmp_path: Path):
    ctx = agent_ctx(tmp_path, FakeEnvironment(lambda _: (3, "partial output")))
    with pytest.raises(CommandError, match="exit code 3: false") as info:
        _ = await ctx.sh("false")
    assert "partial output" in str(info.value)
    assert "boom" in str(info.value)
    result = await ctx.sh("false", check=False)
    assert result.return_code == 3


@pytest.mark.asyncio
async def test_run_file_uploads_runs_and_deletes(tmp_path: Path):
    env = FakeEnvironment()
    ctx = agent_ctx(tmp_path, env)
    _ = (ctx.runner_dir / "seed.py").write_text("print('seed')")
    _ = await ctx.run_file(ctx.runner_dir / "seed.py", "--case", "a b")

    uploads = [args for kind, args in env.calls if kind == "upload_file"]
    ((name, remote),) = uploads
    remote_dir = remote.rsplit("/", 1)[0]
    assert name == "seed.py"
    assert remote_dir.startswith("/tmp/lmnr-")
    commands = env.commands()
    assert commands[0] == f"mkdir -p {remote_dir}"
    assert f"python3 {remote} --case 'a b'" in commands
    assert commands[-1] == f"rm -rf {remote_dir}"


@pytest.mark.asyncio
async def test_copy_puts_host_files_and_dirs_in_the_container(tmp_path: Path):
    env = FakeEnvironment()
    ctx = agent_ctx(tmp_path, env)
    _ = (ctx.runner_dir / "fixtures").mkdir()
    _ = (ctx.runner_dir / "fixtures" / "rows.json").write_text("[]")
    await ctx.copy(ctx.runner_dir / "fixtures" / "rows.json", "/srv/data/rows.json")
    await ctx.copy(ctx.runner_dir / "fixtures", "/srv/fixtures")
    assert env.commands() == ["mkdir -p /srv/data", "mkdir -p /srv/fixtures"]
    assert [c for c in env.calls if c[0] != "exec"] == [
        ("upload_file", ("rows.json", "/srv/data/rows.json")),
        ("upload_dir", (str(ctx.runner_dir / "fixtures"), "/srv/fixtures")),
    ]


@pytest.mark.asyncio
async def test_run_file_deletes_even_when_the_script_fails(tmp_path: Path):
    env = FakeEnvironment(lambda command: (1, "") if command.startswith("bash /tmp") else (0, ""))
    ctx = agent_ctx(tmp_path, env)
    _ = (ctx.runner_dir / "start").write_text("echo no shebang")
    with pytest.raises(CommandError, match=r"run_file\('.*/start'\) failed"):
        _ = await ctx.run_file(ctx.runner_dir / "start")
    assert env.commands()[-1].startswith("rm -rf /tmp/lmnr-")


@pytest.mark.asyncio
async def test_run_file_runs_shebang_scripts_directly(tmp_path: Path):
    env = FakeEnvironment()
    ctx = agent_ctx(tmp_path, env)
    _ = (ctx.runner_dir / "start").write_text("#!/usr/bin/env node\n")
    _ = await ctx.run_file(ctx.runner_dir / "start")
    remote = next(args[1] for kind, args in env.calls if kind == "upload_file")
    assert f"chmod a+x {remote}" in env.commands()
    assert remote in env.commands()


@pytest.mark.asyncio
async def test_service_starts_in_background_and_waits(tmp_path: Path):
    env = FakeEnvironment(lambda command: (0, "4242\n") if "nohup" in command else (0, ""))
    ctx = agent_ctx(tmp_path, env)
    process = await ctx.service("api", "python -m http.server 8000", ready=port(8000))
    assert (process.name, process.pid, process.log_path) == ("api", 4242, "/logs/agent/api.log")
    start, wait = env.commands()
    assert "nohup sh -c 'python -m http.server 8000' > /logs/agent/api.log 2>&1" in start
    assert wait.startswith("bash -c ")
    assert "/dev/tcp/127.0.0.1/8000" in wait
    assert "kill -0 4242" in wait


@pytest.mark.asyncio
async def test_service_that_exits_reports_its_log(tmp_path: Path):
    def respond(command: str) -> tuple[int, str]:
        if "nohup" in command:
            return 0, "7\n"
        if command.startswith("bash -c"):
            return 2, ""  # the process exited
        if command.startswith("tail"):
            return 0, "Error: port in use"
        return 0, ""

    ctx = agent_ctx(tmp_path, FakeEnvironment(respond))
    with pytest.raises(RuntimeError, match="'api' exited") as info:
        _ = await ctx.service("api", "serve", ready=http("http://127.0.0.1:8000/health"))
    assert "Error: port in use" in str(info.value)


@pytest.mark.asyncio
async def test_wait_for_times_out(tmp_path: Path):
    ctx = agent_ctx(tmp_path, FakeEnvironment(lambda _: (1, "")))
    with pytest.raises(TimeoutError, match="`pg_isready` was not ready after 5s"):
        await ctx.wait_for("pg_isready", timeout=5)


@pytest.mark.asyncio
async def test_files_round_trip(tmp_path: Path):
    env = FakeEnvironment()
    ctx = agent_ctx(tmp_path, env)
    await ctx.write_file("/case/data.json", '{"x": 1}')
    assert await ctx.read_json("/case/data.json") == {"x": 1}
    assert "mkdir -p /case" in env.commands()


# Results, judge


def test_result_is_written_to_the_trial_agent_dir(tmp_path: Path):
    ctx = agent_ctx(tmp_path, FakeEnvironment())
    assert ctx.result() == {}
    _ = ctx.result(outcome="completed")
    assert ctx.result(reply="hi") == {"outcome": "completed", "reply": "hi"}
    path = tmp_path / "trial" / "agent" / "result.json"
    assert json.loads(path.read_text()) == {"outcome": "completed", "reply": "hi"}

    verify_ctx = VerifyContext(
        environment=FakeEnvironment(),
        case=ctx.case,
        runner_dir=ctx.runner_dir,
        host_agent_dir=path.parent,
        secrets=Secrets([], {}),
    )
    assert verify_ctx.result()["reply"] == "hi"
    with pytest.raises(EvalConfigError):
        _ = verify_ctx.result(reply="no")


@pytest.mark.asyncio
async def test_judge_asks_for_a_structured_verdict(tmp_path: Path):
    ctx = VerifyContext(
        environment=FakeEnvironment(),
        case=make_case(tmp_path),
        runner_dir=tmp_path,
        host_agent_dir=tmp_path,
        secrets=Secrets([], {}),
        judge_model="anthropic/claude-haiku-4-5",
    )
    message = SimpleNamespace(content='{"pass": true, "reason": "polite and correct"}')
    completion = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)]))
    with patch("litellm.acompletion", completion):
        assert await ctx.judge("Refunded $49.", "Confirm the refund.", api_key="k") == 1
    kwargs = completion.call_args.kwargs
    assert kwargs["model"] == "anthropic/claude-haiku-4-5"
    assert kwargs["api_key"] == "k"
    assert kwargs["response_format"]["type"] == "json_schema"
    assert "Confirm the refund." in kwargs["messages"][0]["content"]
    assert ctx.logger.lines == ["judge (anthropic/claude-haiku-4-5): pass: polite and correct"]


@pytest.mark.asyncio
async def test_judge_needs_a_model(tmp_path: Path):
    ctx = VerifyContext(
        environment=FakeEnvironment(),
        case=make_case(tmp_path),
        runner_dir=tmp_path,
        host_agent_dir=tmp_path,
        secrets=Secrets([], {}),
    )
    with pytest.raises(EvalConfigError, match="needs a model"):
        _ = await ctx.judge("x", "y")


@pytest.mark.asyncio
async def test_default_setup_runs_the_case_seed(tmp_path: Path):
    env = FakeEnvironment()
    ctx = agent_ctx(tmp_path, env)
    await runner_class()().setup(ctx)
    assert env.calls == []  # no seed script
    _ = ctx.case.file("seed.py").write_text("print(1)")
    await runner_class()().setup(ctx)
    assert any(c.startswith("python3 /tmp/lmnr-") for c in env.commands())


def test_lmnr_evals_never_imports_harbor():
    code = (
        "import sys, lmnr.evals, lmnr.evals.describe; "
        "assert not any(m == 'harbor' or m.startswith('harbor.') for m in sys.modules)"
    )
    _ = subprocess.run([sys.executable, "-c", code], check=True)

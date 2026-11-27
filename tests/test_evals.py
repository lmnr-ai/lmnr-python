"""Tests for `lmnr.evals` (EvalRunner, ctx, describe). No Harbor needed: a fake
environment records the `exec` / `upload` calls `ctx` makes."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY

import pytest

from lmnr.evals import (
    AgentContext,
    Artifact,
    Artifacts,
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
    _ = (path.parent / "Dockerfile").write_text("FROM python:3.13-slim\n")
    return path


def layer_steps(lmnr: str = f"lmnr=={importlib.metadata.version('lmnr')}") -> list[dict[str, str]]:
    return [{"origin": f"Laminar layer: Python 3.13, {lmnr}", "step": "laminar layer"}]


SUPPORT_EVAL = """
    from lmnr.evals import Artifact, EvalRunner, Image, secret

    print("module output must not reach the JSON")

    class SupportEval(EvalRunner):
        image = Image.from_dockerfile("Dockerfile", context="../..")
        model = "anthropic/claude-sonnet-5-5"
        secrets = ["ANTHROPIC_API_KEY", "JUDGE_API_KEY"]
        env = {"ANTHROPIC_API_KEY": secret("ANTHROPIC_API_KEY"), "SUPPORT_TONE": "formal"}
        new_task_toml = {"agent": {"timeout_sec": 300}}
        artifacts = ["/srv/app/out/", Artifact("/var/lib/app/state.db", destination="app.db")]

        async def run(self, ctx):
            ctx.set_result(reply="hi")

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
        ({"grade_with_test_sh": True}, "keep one"),
        ({"grade_with_test_sh": "yes"}, "must be True or False"),
        ({"env": {"KEY": secret("KEY")}}, "not in MyEval.secrets"),
        ({"secrets": ["KEY=value"]}, "env var names"),
        ({"env": {"KEY": 1}}, "must be a string or secret"),
        ({"services": ["postgres"]}, "services must be a list of services"),
        ({"mocks": {"crm.get": print}}, "the handler must be a function"),
        ({"mock_dependencies": "anthropic"}, "list of pip requirements"),
        ({"artifacts": "/srv/out"}, "must be a list"),
        ({"artifacts": [1]}, "container paths or Artifact"),
        ({"artifacts": ["/srv/../etc"]}, "must not contain '..'"),
        ({"artifacts": [Artifact("/srv/db", destination="/abs")]}, "relative path"),
        ({"artifacts": [Artifact("/srv/db", destination="../db")]}, "relative path"),
        ({"artifacts": [Artifact("/srv/db", destination="manifest.json")]}, "reserved"),
        (
            {"artifacts": [Artifact("/srv/out", exclude="*.tmp")]},  # pyright: ignore[reportArgumentType]
            "list of patterns",
        ),
    ],
)
def test_invalid_runner_raises_actionable_error(attrs: dict[str, Any], message: str):
    with pytest.raises(EvalConfigError, match=message):
        _ = runner_class(**attrs).validate()


def test_run_command_replaces_run_and_test_sh_replaces_verify():
    runner = runner_class(
        run=EvalRunner.run,
        run_command="python app.py",
        verify=EvalRunner.verify,
        grade_with_test_sh=True,
    )
    assert runner.validate() == []
    assert runner_class(run_command="python app.py").validate() == [
        "MyEval defines run(), so `run_command` is ignored."
    ]


def test_resolved_env_reads_secrets_and_skips_unset_ones():
    runner = runner_class(
        secrets=["A", "B"], env={"A_KEY": secret("A"), "B_KEY": secret("B"), "TONE": "formal"}
    )
    assert runner.resolved_env({"A": "a-value"}) == {"A_KEY": "a-value", "TONE": "formal"}


def test_describe_outputs_json_for_lmnr_cli(tmp_path: Path):
    path = write_eval(tmp_path, SUPPORT_EVAL)
    result = describe(path)
    # The Laminar layer makes even a bare Dockerfile a generated image.
    image = result["images"]["support-evals"]
    assert image["kind"] == "generated"
    assert image["context"] == str(tmp_path.resolve())
    assert "FROM python:3.13-slim AS lmnr-app" in image["dockerfile_text"]
    assert "COPY --from=lmnr-layer /opt/lmnr /opt/lmnr" in image["dockerfile_text"]
    assert image["steps"] == [
        {"origin": ANY, "step": "from_dockerfile Dockerfile"},
        *layer_steps(),
    ]
    assert result == {
        "version": 1,
        "runner": "SupportEval",
        "images": {"support-evals": image},
        "default_image": "support-evals",
        "model": "anthropic/claude-sonnet-5-5",
        "env": {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}", "SUPPORT_TONE": "formal"},
        "secrets": ["ANTHROPIC_API_KEY", "JUDGE_API_KEY"],
        "new_task_toml": {"agent": {"timeout_sec": 300}},
        "grade_with_test_sh": False,
        "warnings": [],
        "modules": ["support/agent/eval.py"],
        "artifacts": [
            "/srv/app/out/",
            {"source": "/var/lib/app/state.db", "destination": "app.db"},
        ],
        "services": [],
        "classes": [{"name": "SupportEval", "file": "support/agent/eval.py"}],
        "mocks": [],
        "mock_dependencies": [],
        "laminar_layer": {
            "python": "3.13",
            "lmnr": f"lmnr=={importlib.metadata.version('lmnr')}",
            "build_contexts": {},
        },
    }


def test_describe_named_images(tmp_path: Path):
    path = write_eval(
        tmp_path,
        """
        from lmnr.evals import EvalRunner, Image

        class E(EvalRunner):
            image = {"new": Image.from_registry("python:3.14"), "old": Image.from_dockerfile("old/Dockerfile")}
            run_command = "true"
            grade_with_test_sh = True
        """,
    )
    (path.parent / "old").mkdir()
    _ = (path.parent / "old" / "Dockerfile").write_text("FROM debian:trixie-slim\n")
    result = describe(path)
    new, old = result["images"]["new"], result["images"]["old"]
    assert (new["kind"], old["kind"]) == ("generated", "generated")
    assert new["steps"] == [
        {"origin": "support/agent/eval.py:5 (E)", "step": "from python:3.14"},
        *layer_steps(),
    ]
    assert old["context"] == str((path.parent / "old").resolve())
    assert old["steps"] == [
        {"origin": "support/agent/eval.py:5 (E)", "step": "from_dockerfile old/Dockerfile"},
        *layer_steps(),
    ]
    assert result["default_image"] == "new"
    assert result["grade_with_test_sh"] is True


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
            await ctx.container.run_file("start.sh")
            await ctx.container.copy("fixtures", "/srv/fixtures")

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
    _ = (shared / "Dockerfile").write_text("FROM python:3.13-slim\n")
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
    image = result["images"]["refunds-evals"]
    assert image["context"] == str(root.resolve())
    assert "# from _shared/Dockerfile (SupportRunner)" in image["dockerfile_text"]
    # Declared in the shared module, by the shared class.
    assert image["steps"] == [
        {"origin": "_shared/support.py:5 (SupportRunner)", "step": "from_dockerfile Dockerfile"},
        *layer_steps(),
    ]
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
                await ctx.container.run_file("seed.py")

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


def test_secrets_repr_shows_names_never_values():
    secrets = Secrets(["JUDGE", "MISSING"], {"JUDGE": "j-value", "OTHER": "o-value"})
    assert repr(secrets) == "Secrets(set=['JUDGE'], unset=['MISSING'])"
    assert "value" not in str(secrets)


# Commands


@pytest.mark.asyncio
async def test_sh_adds_runner_env_and_maps_users(tmp_path: Path):
    env = FakeEnvironment()
    ctx = agent_ctx(tmp_path, env, exec_env={"TONE": "formal"})
    result = await ctx.container.sh("echo hi", env={"EXTRA": "1"}, timeout=2.5)
    _ = await ctx.container.sh("whoami", user="root")
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
        _ = await ctx.container.sh("false")
    assert "partial output" in str(info.value)
    assert "boom" in str(info.value)
    result = await ctx.container.sh("false", check=False)
    assert result.return_code == 3


@pytest.mark.asyncio
async def test_run_file_uploads_runs_and_deletes(tmp_path: Path):
    env = FakeEnvironment()
    ctx = agent_ctx(tmp_path, env)
    _ = (ctx.runner_dir / "seed.py").write_text("print('seed')")
    _ = await ctx.container.run_file(ctx.runner_dir / "seed.py", "--case", "a b")

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
    await ctx.container.copy(ctx.runner_dir / "fixtures" / "rows.json", "/srv/data/rows.json")
    await ctx.container.copy(ctx.runner_dir / "fixtures", "/srv/fixtures")
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
        _ = await ctx.container.run_file(ctx.runner_dir / "start")
    assert env.commands()[-1].startswith("rm -rf /tmp/lmnr-")


@pytest.mark.asyncio
async def test_run_file_runs_shebang_scripts_directly(tmp_path: Path):
    env = FakeEnvironment()
    ctx = agent_ctx(tmp_path, env)
    _ = (ctx.runner_dir / "start").write_text("#!/usr/bin/env node\n")
    _ = await ctx.container.run_file(ctx.runner_dir / "start")
    remote = next(args[1] for kind, args in env.calls if kind == "upload_file")
    assert f"chmod a+x {remote}" in env.commands()
    assert remote in env.commands()


@pytest.mark.asyncio
async def test_service_starts_in_background_and_waits(tmp_path: Path):
    env = FakeEnvironment(lambda command: (0, "4242\n") if "nohup" in command else (0, ""))
    ctx = agent_ctx(tmp_path, env)
    process = await ctx.container.service("api", "python -m http.server 8000", ready=port(8000))
    assert (process.name, process.pid, process.log_path) == ("api", 4242, "/logs/agent/api.log")
    start, wait = [c for c in env.commands() if "nohup" in c or "relay.py wait" in c]
    assert "nohup sh -c 'python -m http.server 8000' > /logs/agent/api.log 2>&1" in start
    # The polling loop is the relay's; what to wait for comes on stdin, from a
    # temp file the shell deletes before the relay starts.
    request = next(path for path in env.files if path in wait)
    assert wait == (
        f"{{ rm -f {request} 2>/dev/null; /opt/lmnr/venv/bin/python -I /opt/lmnr/relay.py wait; }}"
        + f" < {request}"
    )
    assert json.loads(env.files[request]) == {
        "ready": {"kind": "port", "host": "127.0.0.1", "port": 8000},
        "timeout": 60,
        "pid": 4242,
    }
    # The agent user can't delete root's file in sticky /tmp: root does.
    assert env.commands()[-1] == f"rm -f {request}"


@pytest.mark.asyncio
async def test_service_that_exits_reports_its_log(tmp_path: Path):
    def respond(command: str) -> tuple[int, str]:
        if "nohup" in command:
            return 0, "7\n"
        if "relay.py wait" in command:
            return 2, ""  # the process exited
        if command.startswith("tail"):
            return 0, "Error: port in use"
        return 0, ""

    ctx = agent_ctx(tmp_path, FakeEnvironment(respond))
    with pytest.raises(RuntimeError, match="'api' exited") as info:
        _ = await ctx.container.service("api", "serve", ready=http("http://127.0.0.1:8000/health"))
    assert "Error: port in use" in str(info.value)


@pytest.mark.asyncio
async def test_wait_for_times_out(tmp_path: Path):
    ctx = agent_ctx(tmp_path, FakeEnvironment(lambda c: (1, "") if "relay.py" in c else (0, "")))
    with pytest.raises(TimeoutError, match="`pg_isready` was not ready after 5s"):
        await ctx.container.wait_for("pg_isready", timeout=5)


@pytest.mark.asyncio
async def test_files_round_trip(tmp_path: Path):
    env = FakeEnvironment()
    ctx = agent_ctx(tmp_path, env)
    await ctx.container.write_file("/case/data.json", '{"x": 1}')
    assert await ctx.container.read_json("/case/data.json") == {"x": 1}
    assert "mkdir -p /case" in env.commands()


# Results


def test_result_is_written_to_the_trial_agent_dir(tmp_path: Path):
    ctx = agent_ctx(tmp_path, FakeEnvironment())
    assert ctx.result() == {}
    _ = ctx.set_result(outcome="completed")
    assert ctx.set_result(reply="hi") == {"outcome": "completed", "reply": "hi"}
    assert ctx.result() == {"outcome": "completed", "reply": "hi"}
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
    # Recording is for the agent phase only.
    assert not hasattr(verify_ctx, "set_result")


# Artifacts


@pytest.mark.asyncio
async def test_artifacts_save_replaces_its_copy_under_logs_artifacts(tmp_path: Path):
    env = FakeEnvironment()
    ctx = agent_ctx(tmp_path, env)
    assert await ctx.artifacts.save("/srv/app/out/") == "/logs/artifacts/out"
    assert await ctx.artifacts.save("/var/lib/app.db", name="state/app.db") == (
        "/logs/artifacts/state/app.db"
    )
    first, second = env.execs()
    assert first["user"] == "root"
    assert first["command"] == (
        "rm -rf /logs/artifacts/out && mkdir -p /logs/artifacts && "
        + "cp -a /srv/app/out/ /logs/artifacts/out && chmod -R a+rX /logs/artifacts/out"
    )
    assert "mkdir -p /logs/artifacts/state && cp -a /var/lib/app.db" in second["command"]
    for dest in ("../x", "/abs", ""):
        with pytest.raises(EvalConfigError):
            _ = await ctx.artifacts.save("/srv/x", name=dest)


@pytest.mark.asyncio
async def test_artifacts_save_fails_when_the_source_is_missing(tmp_path: Path):
    ctx = agent_ctx(tmp_path, FakeEnvironment(lambda _: (1, "")))
    with pytest.raises(CommandError):
        _ = await ctx.artifacts.save("/srv/missing")


def test_artifacts_read_the_trial_artifacts_dir(tmp_path: Path):
    host = tmp_path / "trial" / "artifacts"
    (host / "srv" / "app" / "out").mkdir(parents=True)
    _ = (host / "srv" / "app" / "out" / "report.json").write_text('{"ok": true}')
    (host / "logs" / "artifacts").mkdir(parents=True)
    _ = (host / "logs" / "artifacts" / "notes.txt").write_text("saved in collect")
    (host / "db").mkdir()
    _ = (host / "db" / "app.db").write_text("sqlite")
    _ = (host / "manifest.json").write_text("[]")
    artifacts = Artifacts(host, ["/srv/app/out/", Artifact("/var/lib/app", destination="db")])

    # Container paths, as declared or saved...
    assert artifacts.read_json("/srv/app/out/report.json") == {"ok": True}
    assert artifacts.read_text("/logs/artifacts/notes.txt") == "saved in collect"
    # ...also when collected under a destination; or paths under artifacts/.
    assert artifacts.read_text("/var/lib/app/app.db") == "sqlite"
    assert artifacts.read_text("db/app.db") == "sqlite"
    assert not artifacts.exists("/srv/other")
    assert artifacts.list() == [
        "db/app.db",
        "logs/artifacts/notes.txt",
        "srv/app/out/report.json",
    ]
    with pytest.raises(ValueError):
        _ = artifacts.path("/srv/../etc/passwd")
    assert Artifacts(tmp_path / "missing").list() == []


def test_artifacts_resolve_through_the_most_specific_entry(tmp_path: Path):
    # Harbor collects nested sources separately; the parent's copy excludes db/.
    host = tmp_path / "artifacts"
    nested = Artifacts(
        host,
        [
            Artifact("/srv", destination="all", exclude=["db/*"]),
            Artifact("/srv/db", destination="db"),
            "/srv/db/raw",
        ],
    )
    assert nested.path("/srv/app.log") == host / "all" / "app.log"
    assert nested.path("/srv/db/state.json") == host / "db" / "state.json"
    # A nested entry without a destination is collected at its container path.
    assert nested.path("/srv/db/raw/0.bin") == host / "srv" / "db" / "raw" / "0.bin"
    # Declaration order doesn't matter.
    reordered = Artifacts(host, [Artifact("/srv/db", destination="db"), Artifact("/srv", "all")])
    assert reordered.path("/srv/db/state.json") == host / "db" / "state.json"


def test_verify_ctx_artifacts_default_to_the_trial_dir(tmp_path: Path):
    agent_dir = tmp_path / "trial" / "agent"
    ctx = VerifyContext(
        environment=FakeEnvironment(),
        case=make_case(tmp_path),
        runner_dir=tmp_path,
        host_agent_dir=agent_dir,
        secrets=Secrets([], {}),
    )
    assert ctx.artifacts.dir == tmp_path / "trial" / "artifacts"


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

"""`ctx`: what an `EvalRunner`'s phases get to drive the trial.

Commands and files go through the environment Harbor hands to the agent or the
verifier (`exec`, `upload_file`, `download_file`), so they work the same in local
Docker and in Laminar's hosted sandboxes. This module never imports Harbor: the
environment is duck-typed (`Environment`).
"""

from __future__ import annotations

import json
import logging
import math
import re
import shlex
import tempfile
import tomllib
import uuid
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from typing_extensions import override

from lmnr.evals.errors import OUTPUT_TAIL_CHARS, CommandError, EvalConfigError, ExecResult
from lmnr.evals.judge import judge
from lmnr.evals.paths import caller_dir
from lmnr.evals.readiness import Readiness
from lmnr.evals.readiness import cmd as cmd_ready
from lmnr.sdk.log import get_default_logger

logger = get_default_logger("lmnr.evals")

# Container paths, the same as Harbor's `EnvironmentPaths`.
AGENT_LOGS_DIR = "/logs/agent"
VERIFIER_LOGS_DIR = "/logs/verifier"
RESULT_FILE = "result.json"

User = Literal["agent", "root"] | str | int

_SERVICE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Files that `setup` runs by default when a case has one in `environment/`.
_SEED_SUFFIXES = (".py", ".sh", ".bash")


class Environment(Protocol):
    """The part of Harbor's `BaseEnvironment` that `ctx` uses."""

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> Any: ...

    async def upload_file(self, source_path: Path | str, target_path: str) -> Any: ...

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> Any: ...

    async def download_file(self, source_path: str, target_path: Path | str) -> Any: ...


@dataclass(frozen=True)
class BackgroundProcess:
    """A process started with `ctx.service(...)`."""

    name: str
    pid: int
    log_path: str


@dataclass(frozen=True)
class Case:
    """The task a trial runs, read from its directory on the host."""

    name: str
    dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    instruction: str = ""

    @classmethod
    def from_dir(cls, task_dir: Path | str) -> Case:
        task_dir = Path(task_dir).resolve()
        config: dict[str, Any] = {}
        task_toml = task_dir / "task.toml"
        if task_toml.is_file():
            config = tomllib.loads(task_toml.read_text())
        task = config.get("task")
        name = task.get("name") if isinstance(task, dict) else None
        metadata = config.get("metadata")
        instruction_path = task_dir / "instruction.md"
        return cls(
            name=name if isinstance(name, str) and name else task_dir.name,
            dir=task_dir,
            metadata=metadata if isinstance(metadata, dict) else {},
            instruction=instruction_path.read_text() if instruction_path.is_file() else "",
        )

    def file(self, path: str) -> Path:
        """A file under the case's `environment/` (data the app reads)."""
        return self.dir / "environment" / path

    def fixture(self, path: str) -> Path:
        """A file under the case's `tests/fixtures/` (data the agent must not see)."""
        return self.dir / "tests" / "fixtures" / path

    def fixture_json(self, path: str) -> Any:
        return json.loads(self.fixture(path).read_text())

    def expected(self) -> Any:
        """The case's reference answer, `tests/expected.json`."""
        return json.loads((self.dir / "tests" / "expected.json").read_text())

    def seed_file(self) -> Path | None:
        """The case's own seed script (`environment/seed.py`, `seed.sh`, or another
        `seed.*` with a shebang), if it has one."""
        env_dir = self.dir / "environment"
        if not env_dir.is_dir():
            return None
        for path in sorted(env_dir.glob("seed.*")):
            if path.is_file() and (path.suffix in _SEED_SUFFIXES or _has_shebang(path)):
                return path
        return None


class Secrets(Mapping[str, str]):
    """The runner's declared `secrets`, read from the Harbor process env."""

    def __init__(self, declared: list[str], values: Mapping[str, str]):
        self._declared: list[str] = list(declared)
        self._values: dict[str, str] = {
            name: values[name] for name in declared if values.get(name) is not None
        }

    @override
    def __getitem__(self, key: str) -> str:
        if key not in self._declared:
            raise KeyError(
                f"{key!r} is not in the runner's `secrets`. Add it there, and lmnr-cli "
                + "reads it from the shell, .env.local or .env."
            )
        if key not in self._values:
            raise KeyError(f"Secret {key!r} is not set in the environment `harbor run` runs in.")
        return self._values[key]

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    @override
    def __len__(self) -> int:
        return len(self._values)

    @override
    def __repr__(self) -> str:
        return f"Secrets({sorted(self._values)})"


class EvalLogger(logging.LoggerAdapter[logging.Logger]):
    """`ctx.logger`: logs through the SDK's logger, prefixed with the case name,
    and keeps the lines so the verifier can write them to its log."""

    def __init__(self, case_name: str):
        super().__init__(logger, {})
        self.case_name: str = case_name
        self.lines: list[str] = []

    @override
    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> tuple[Any, Any]:
        self.lines.append(str(msg))
        return f"[{self.case_name}] {msg}", kwargs


class Context:
    """Commands and files in the container, plus the case and the run settings.

    - `model`: the model to run (`--model`, harbor.yaml, or the runner's `model`).
    - `env`: the runner's `env`, resolved.
    - `case`: the task (`Case`).
    - `logger`: a logger whose lines also go to the verifier's log.
    """

    logs_dir: str = AGENT_LOGS_DIR

    def __init__(
        self,
        *,
        environment: Environment,
        case: Case,
        runner_dir: Path,
        model: str | None = None,
        env: dict[str, str] | None = None,
        exec_env: dict[str, str] | None = None,
    ):
        self.environment: Environment = environment
        self.case: Case = case
        self.model: str | None = model
        self.env: dict[str, str] = dict(env or {})
        self.logger: EvalLogger = EvalLogger(case.name)
        self.runner_dir: Path = runner_dir
        # Added to every command. Variables Harbor already sets on the agent's
        # commands (`agents[].env`) aren't repeated here.
        self._exec_env: dict[str, str] = dict(exec_env or {})

    # Commands

    async def sh(
        self,
        cmd: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        user: User = "agent",
        timeout: float | None = None,
        check: bool = True,
    ) -> ExecResult:
        """Run a shell command (`sh -c`) in the container.

        `user="agent"` runs as the task's agent (or verifier) user, `"root"` as root.
        With `check=True`, a non-zero exit raises `CommandError`.
        """
        return await self._exec(cmd, env=env, cwd=cwd, user=user, timeout=timeout, check=check)

    async def run_file(
        self,
        path: str | Path,
        *args: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        user: User = "agent",
        timeout: float | None = None,
        check: bool = True,
    ) -> ExecResult:
        """Upload a file and run it in the container, then delete it.

        A relative `path` resolves against the directory of the file whose code
        calls `run_file` (`eval.py`, or a shared module). The file goes to a fresh
        directory under `/tmp`, so harness scripts never stay where the agent
        under test could read them. A file with a shebang runs directly, a `.py`
        file with `python3`, anything else with `bash`.
        """
        local = self._host_path(path)
        if not local.is_file():
            raise FileNotFoundError(f"run_file: {local} does not exist")
        remote_dir = f"/tmp/lmnr-{uuid.uuid4().hex[:12]}"
        remote = f"{remote_dir}/{local.name}"
        _ = await self._root(f"mkdir -p {shlex.quote(remote_dir)}")
        try:
            await self.environment.upload_file(local, remote)
            _ = await self._root(f"chmod -R a+rX {shlex.quote(remote_dir)}")
            if _has_shebang(local):
                _ = await self._root(f"chmod a+x {shlex.quote(remote)}")
                argv = [remote]
            elif local.suffix == ".py":
                argv = ["python3", remote]
            else:
                argv = ["bash", remote]
            command = shlex.join([*argv, *args])
            return await self._exec(
                command,
                env=env,
                cwd=cwd,
                user=user,
                timeout=timeout,
                check=check,
                what=f"run_file({str(path)!r})",
            )
        finally:
            try:
                _ = await self._root(f"rm -rf {shlex.quote(remote_dir)}", check=False)
            except Exception as e:
                logger.warning(f"Could not remove {remote_dir}: {e}")

    async def service(
        self,
        name: str,
        cmd: str,
        ready: Readiness | str,
        env: dict[str, str] | None = None,
        timeout: float = 60,
        cwd: str | None = None,
        user: User = "agent",
    ) -> BackgroundProcess:
        """Start a background process and wait until it is ready.

        Output goes to `/logs/agent/<name>.log`. `ready` is `port(5432)`,
        `http(url)` or `cmd("...")` (a plain string is a command). If the process
        exits or isn't ready within `timeout` seconds, the error has the end of
        its log.
        """
        if not _SERVICE_NAME.match(name):
            raise EvalConfigError(
                f"Service name {name!r} must be letters, digits, '.', '_' or '-' (it names "
                + "the log file)."
            )
        log_path = f"{self.logs_dir}/{name}.log"
        start = (
            f"mkdir -p {shlex.quote(self.logs_dir)} && "
            + f"nohup sh -c {shlex.quote(cmd)} > {shlex.quote(log_path)} 2>&1 < /dev/null & "
            + "echo $!"
        )
        result = await self._exec(
            start, env=env, cwd=cwd, user=user, timeout=30, what=f"Starting service {name!r}"
        )
        try:
            pid = int(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise RuntimeError(
                f"Service {name!r} did not report its pid: {result.stdout!r}"
            ) from None
        process = BackgroundProcess(name=name, pid=pid, log_path=log_path)
        status = await self._wait(ready, timeout, pid=pid, user=user)
        if status != 0:
            reason = "exited" if status == 2 else f"was not ready after {timeout:g}s"
            tail = await self._exec(
                f"tail -n 50 {shlex.quote(log_path)}", user="root", check=False, timeout=30
            )
            log_tail = tail.stdout[-OUTPUT_TAIL_CHARS:].rstrip() or "(empty)"
            raise RuntimeError(
                f"Service {name!r} {reason} (waiting for {_readiness(ready)}). "
                + f"End of {log_path}:\n{log_tail}"
            )
        return process

    async def wait_for(self, ready: Readiness | str, timeout: float = 60) -> None:
        """Wait until `ready` passes, or raise `TimeoutError`."""
        if await self._wait(ready, timeout) != 0:
            raise TimeoutError(f"{_readiness(ready)} was not ready after {timeout:g}s")

    # Files in the container

    async def copy(self, host: str | Path, container: str) -> None:
        """Copy a file or directory from the host (the machine running Harbor) to
        `container`, a path in the container. A relative `host` path resolves
        against the directory of the file whose code calls `copy`."""
        source = self._host_path(host)
        if source.is_dir():
            _ = await self._root(f"mkdir -p {shlex.quote(container)}")
            await self.environment.upload_dir(source, container)
        else:
            _ = await self._root(f"mkdir -p {shlex.quote(_parent(container))}")
            await self.environment.upload_file(source, container)

    async def write_file(self, path: str, content: str | bytes) -> None:
        """Write `content` to `path` in the container."""
        with tempfile.TemporaryDirectory(prefix="lmnr-evals-") as tmp:
            local = Path(tmp) / "file"
            if isinstance(content, str):
                _ = local.write_text(content)
            else:
                _ = local.write_bytes(content)
            _ = await self._root(f"mkdir -p {shlex.quote(_parent(path))}")
            await self.environment.upload_file(local, path)

    async def read_text(self, path: str) -> str:
        """Read a file from the container."""
        with tempfile.TemporaryDirectory(prefix="lmnr-evals-") as tmp:
            local = Path(tmp) / "file"
            await self.environment.download_file(path, local)
            return local.read_text()

    async def read_json(self, path: str) -> Any:
        return json.loads(await self.read_text(path))

    # Internals

    def _host_path(self, path: str | Path) -> Path:
        """`path` on the host. Relative to the calling code's file, so call it
        first thing in a helper, before any `await`: a coroutine only knows its
        caller until it first suspends. Falls back to `eval.py`'s directory
        when the caller is unknown (a coroutine run as its own task)."""
        path = Path(path)
        if path.is_absolute():
            return path
        return (caller_dir() or self.runner_dir) / path

    async def _root(self, command: str, check: bool = True) -> ExecResult:
        return await self._exec(command, user="root", check=check, timeout=60, base_env=False)

    async def _exec(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        user: User = "agent",
        timeout: float | None = None,
        check: bool = True,
        what: str = "Command",
        base_env: bool = True,
    ) -> ExecResult:
        merged = {**self._exec_env, **(env or {})} if base_env else dict(env or {})
        raw = await self.environment.exec(
            command,
            cwd=cwd,
            env=merged or None,
            timeout_sec=math.ceil(timeout) if timeout is not None else None,
            user=None if user == "agent" else user,
        )
        result = ExecResult(
            stdout=getattr(raw, "stdout", None) or "",
            stderr=getattr(raw, "stderr", None) or "",
            return_code=int(getattr(raw, "return_code", 0) or 0),
        )
        if check and result.return_code != 0:
            raise CommandError(command, result, what=what)
        return result

    async def _wait(
        self,
        ready: Readiness | str,
        timeout: float,
        pid: int | None = None,
        user: User = "agent",
    ) -> int:
        """0 once ready, 1 on timeout, 2 if `pid` exited first. One `exec`: the
        polling loop runs in the container."""
        check = ready if isinstance(ready, Readiness) else cmd_ready(ready)
        alive = f"kill -0 {pid} 2>/dev/null || exit 2; " if pid is not None else ""
        script = (
            f"end=$((SECONDS + {math.ceil(timeout)})); "
            + f"until {{ {check.command}; }} >/dev/null 2>&1; do "
            + alive
            + '[ "$SECONDS" -ge "$end" ] && exit 1; sleep 0.5; done'
        )
        result = await self._exec(
            f"bash -c {shlex.quote(script)}",
            user=user,
            timeout=timeout + 30,
            check=False,
        )
        return result.return_code if result.return_code in (0, 2) else 1


class AgentContext(Context):
    """`ctx` in `setup`, `run` and `collect`, in the agent container."""

    def __init__(self, *, host_logs_dir: Path, **kwargs: Any):
        super().__init__(**kwargs)
        # The trial's `agent/` dir on the host, which Harbor syncs with
        # /logs/agent in the container.
        self.host_logs_dir: Path = host_logs_dir

    def result(self, **fields: Any) -> dict[str, Any]:
        """`ctx.result(reply=...)` records fields in `/logs/agent/result.json`
        (merged with earlier ones); `ctx.result()` reads them back.

        The file is written in the trial's agent logs dir, which Harbor syncs with
        the container's /logs/agent, so `verify` reads it in both verifier modes.
        """
        path = self.host_logs_dir / RESULT_FILE
        current = _read_json_object(path)
        if fields:
            current.update(fields)
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(json.dumps(current, indent=2, default=str))
        return current


class VerifyContext(Context):
    """`ctx` in `verify`: logic in the Harbor process, commands in the verifier
    container (the agent container in `shared` mode, a fresh one in `separate`)."""

    logs_dir: str = VERIFIER_LOGS_DIR

    def __init__(
        self,
        *,
        host_agent_dir: Path,
        secrets: Secrets,
        judge_model: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.host_agent_dir: Path = host_agent_dir
        self.secrets: Secrets = secrets
        self._judge_model: str | None = judge_model

    def result(self, **fields: Any) -> dict[str, Any]:
        """What the agent phases recorded with `ctx.result(...)` (`{}` if nothing)."""
        if fields:
            raise EvalConfigError(
                "ctx.result(...) can only record fields in setup, run or collect."
            )
        return _read_json_object(self.host_agent_dir / RESULT_FILE)

    async def judge(
        self,
        output: Any,
        criteria: str,
        model: str | None = None,
        api_key: str | None = None,
    ) -> int:
        """Ask an LLM whether `output` meets `criteria`: 1 for pass, 0 for fail.

        Uses LiteLLM model names (`"anthropic/claude-haiku-4-5"`); defaults to the
        runner's `model`. The verdict and its reason go to the verifier log.
        """
        resolved = model or self._judge_model
        if not resolved:
            raise EvalConfigError(
                "ctx.judge() needs a model: pass model=... or set `model` on the runner."
            )
        passed, reason = await judge(output, criteria, model=resolved, api_key=api_key)
        self.logger.info(f"judge ({resolved}): {'pass' if passed else 'fail'}: {reason}")
        return int(passed)


def normalize_rewards(value: Any) -> dict[str, float | int]:
    """`verify()`'s return value as Harbor rewards: a number becomes
    `{"reward": x}`, and booleans become 0 / 1."""
    if isinstance(value, Mapping):
        items = list(value.items())
    else:
        items = [("reward", value)]
    if not items:
        raise EvalConfigError("verify() returned no rewards.")
    rewards: dict[str, float | int] = {}
    for key, reward in items:
        if isinstance(reward, bool):
            reward = int(reward)
        if not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise EvalConfigError(
                f"verify() must return a number or a dict of numbers, got {key!r}: {reward!r}"
            )
        rewards[str(key)] = reward
    return rewards


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read {path}: {e}")
        return {}
    return data if isinstance(data, dict) else {"value": data}


def _has_shebang(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(2) == b"#!"
    except OSError:
        return False


def _parent(path: str) -> str:
    parent = path.rsplit("/", 1)[0]
    return parent or "/"


def _readiness(ready: Readiness | str) -> str:
    return str(ready) if isinstance(ready, Readiness) else f"`{ready}`"

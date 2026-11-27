"""`ctx`: what an `EvalRunner`'s phases get to drive the trial.

Commands and files go through the environment Harbor hands to the agent or the
verifier (`exec`, `upload_file`, `download_file`), so they work the same in local
Docker and in Laminar's hosted sandboxes. This module never imports Harbor: the
environment is duck-typed (`Environment`).
"""

from __future__ import annotations

import base64
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
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from typing_extensions import override

from lmnr.evals.artifacts import ARTIFACTS_DIR, Artifact, Artifacts, check_destination
from lmnr.evals.errors import (
    OUTPUT_TAIL_CHARS,
    CommandError,
    CommandTimeout,
    EvalConfigError,
    ExecResult,
    output_tails,
)
from lmnr.evals.layer import LAYER_DIR, relay_command
from lmnr.evals.mocks.calls import MockCalls
from lmnr.evals.mocks.mock import FixtureReader, fixture_key, missing_fixture
from lmnr.evals.paths import caller_dir
from lmnr.evals.readiness import Readiness
from lmnr.evals.readiness import cmd as cmd_ready
from lmnr.evals.relay import parse_result
from lmnr.sdk.log import get_default_logger

logger = get_default_logger("lmnr.evals")

# Container paths, the same as Harbor's `EnvironmentPaths`.
AGENT_LOGS_DIR = "/logs/agent"
VERIFIER_LOGS_DIR = "/logs/verifier"
RESULT_FILE = "result.json"

User = Literal["agent", "root"] | str | int
VerifierMode = Literal["shared", "separate"]

# Harbor's own timeout for a command run through `relay.py exec`: the
# command's plus this, so the relay's fires first in every environment.
EXEC_TIMEOUT_MARGIN_SEC = 30

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
    """A process started with `ctx.container.service(...)`."""

    name: str
    pid: int
    log_path: str


@dataclass(frozen=True)
class Case(FixtureReader):
    """The task a trial runs, read from its directory on the host.

    `fixture_bytes(path)` / `fixture_text` / `fixture_json` read its
    `tests/fixtures/`, as a mock handler's `ctx.case` does; `fixture_path`
    is the host path."""

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

    def fixture_path(self, path: str) -> Path:
        """A file under the case's `tests/fixtures/` (data the agent must not
        see), on the host. A mock handler's `ctx.case` has no paths, only the
        readers."""
        return self.dir / "tests" / "fixtures" / fixture_key(path)

    @override
    def fixture_bytes(self, path: str) -> bytes:
        file = self.fixture_path(path)
        if not file.is_file():
            raise missing_fixture(self.name, fixture_key(path))
        return file.read_bytes()

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
    """The runner's secrets (`secrets` and the `secret(...)`s in `env`), read
    from the Harbor process env."""

    def __init__(self, declared: list[str], values: Mapping[str, str]):
        self._declared: list[str] = list(declared)
        self._values: dict[str, str] = {
            name: values[name] for name in declared if values.get(name) is not None
        }

    @override
    def __getitem__(self, key: str) -> str:
        if key not in self._declared:
            raise KeyError(
                f"{key!r} is not one of the runner's secrets. Pass it to the agent with "
                + f"`env = {{{key!r}: secret({key!r})}}`, or list it in `secrets` to keep it "
                + "out of the container; lmnr-cli reads it from the shell, .env.local or .env."
            )
        if key not in self._values:
            raise KeyError(
                f"Secret {key!r} is not set in the environment `harbor run` runs in. "
                + "Set it in your shell, .env.local or .env (lmnr-cli reads them)."
            )
        return self._values[key]

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    @override
    def __len__(self) -> int:
        return len(self._values)

    @override
    def __repr__(self) -> str:
        # Names only, never values, so `ctx.logger.info(ctx.secrets)` or a
        # traceback showing `ctx` can't leak a secret into logs and traces.
        unset = [name for name in self._declared if name not in self._values]
        return f"Secrets(set={sorted(self._values.keys())}, unset={sorted(unset)})"


class EvalLogger(logging.LoggerAdapter[logging.Logger]):
    """`ctx.logger`: logs through the SDK's logger, prefixed with the case name,
    and keeps the lines so the verifier can write them to its log."""

    def __init__(self, case_name: str):
        super().__init__(logger, {})
        self.case_name: str = case_name
        self.lines: list[str] = []

    @override
    def log(self, level: int, msg: object, *args: object, **kwargs: Any) -> None:
        # Kept whatever the logger's level, with its arguments applied.
        self.lines.append(str(msg) % args if args else str(msg))
        super().log(level, msg, *args, **kwargs)

    @override
    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> tuple[Any, Any]:
        return f"[{self.case_name}] {msg}", kwargs


class Container:
    """`ctx.container`: commands and files in the trial's container. Every
    other `ctx` attribute is read on the host, where your `eval.py` runs.

    Which container: in `setup` / `run` / `collect`, the agent's. In `verify`,
    the grading container: the agent's, after the agent finished (the
    default), or a fresh one from the same image holding only the collected
    artifacts (`[verifier] environment_mode = "separate"` in `task.toml`).
    """

    def __init__(
        self,
        environment: Environment,
        *,
        runner_dir: Path,
        logs_dir: str,
        exec_env: dict[str, str] | None = None,
    ):
        self.environment: Environment = environment
        # Where `service` logs go: /logs/agent, or /logs/verifier in verify.
        self.logs_dir: str = logs_dir
        self._runner_dir: Path = runner_dir
        # Added to every command. Variables Harbor already sets on the agent's
        # commands (`agents[].env`) aren't repeated here.
        self._exec_env: dict[str, str] = dict(exec_env or {})

    def add_env(self, env: Mapping[str, str]) -> None:
        """Add variables to every later command (services' `env()`)."""
        self._exec_env.update(env)

    @property
    def exec_env(self) -> dict[str, str]:
        """What `add_env` added (a copy)."""
        return dict(self._exec_env)

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
        """Run a shell command (`bash -c`, or `sh -c` without bash) in the container.

        `user="agent"` runs as the task's agent (or verifier) user, `"root"` as root.
        `stdout` and `stderr` come back apart. With `check=True`, a non-zero exit
        raises `CommandError`. After `timeout` seconds the command and everything
        it started are killed and `CommandTimeout` (a `TimeoutError`) is raised,
        whatever `check` is; its `result` has the output so far.

        Commands run through the Laminar layer's relay, so they behave the same
        in local Docker and in Laminar's hosted sandboxes.
        """
        return await self._run(cmd, env=env, cwd=cwd, user=user, timeout=timeout, check=check)

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
        file with `python3`, anything else with `bash`. Output, exit code and
        `timeout` work as in `sh`.
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
            return await self._run(
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

        Output goes to `<logs dir>/<name>.log` (`/logs/agent`, or
        `/logs/verifier` in `verify`). `ready` is `port(5432)`, `http(url)`,
        `cmd("...")` (a plain string is a command) or
        `lmnr.evals.readiness.file(path)`; it runs with
        the same `env` as `cmd`. If the process
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
            # Only `nohup` goes to the background (`a && b &` would background a
            # subshell holding the exec's stdout open until the service exits).
            f"mkdir -p {shlex.quote(self.logs_dir)} || exit 1; "
            + f"nohup sh -c {shlex.quote(cmd)} > {shlex.quote(log_path)} 2>&1 < /dev/null & "
            + "echo $!"
        )
        # Not through the relay: it returns at once, and only the pid is read.
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
        status = await self._wait(ready, timeout, pid=pid, env=env, user=user)
        if status != 0:
            reason = "exited" if status == 2 else f"was not ready after {timeout:g}s"
            # stderr dropped: some environments merge it into stdout.
            tail = await self._exec(
                f"tail -n 50 {shlex.quote(log_path)} 2>/dev/null",
                user="root",
                check=False,
                timeout=30,
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

    async def copy(self, host: str | Path, path: str) -> None:
        """Copy a file or directory from the host (the machine running Harbor) to
        `path` in the container. A relative `host` path resolves against the
        directory of the file whose code calls `copy`."""
        source = self._host_path(host)
        if source.is_dir():
            _ = await self._root(f"mkdir -p {shlex.quote(path)}")
            await self.environment.upload_dir(source, path)
        else:
            _ = await self._root(f"mkdir -p {shlex.quote(_parent(path))}")
            await self.environment.upload_file(source, path)

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
        return (caller_dir() or self._runner_dir) / path

    async def _root(self, command: str, check: bool = True) -> ExecResult:
        return await self._exec(command, user="root", check=check, timeout=60, base_env=False)

    async def _run(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        user: User = "agent",
        timeout: float | None = None,
        check: bool = True,
        what: str = "Command",
    ) -> ExecResult:
        """A user command, through `relay.py exec`: stdout and stderr apart and
        the timeout enforced in the container, whatever the environment (Harbor's
        local Docker merges stderr into stdout and raises on a timeout; Laminar's
        sandboxes keep them apart and exit 124). One `exec`, plus the payload
        upload; a second `exec` removes the payload when the relay's user couldn't
        (`stdin_unlinked`)."""
        path = await _upload_payload(self.environment, {"cmd": command, "timeout": timeout})
        removed = False
        try:
            raw = await self._exec(
                _relay_shell("exec", path),
                env=env,
                cwd=cwd,
                user=user,
                timeout=None if timeout is None else timeout + EXEC_TIMEOUT_MARGIN_SEC,
                check=False,
            )
            _check_layer(raw)
            framed = parse_result(raw.stdout)
            if framed is None:
                raise RuntimeError(
                    f"{what}: the Laminar layer's relay printed no result (exit code "
                    + f"{raw.return_code}): {command}"
                    + output_tails(raw)
                )
            removed = bool(framed.get("stdin_unlinked"))
        finally:
            if not removed:
                await _remove_payload(self.environment, path)
        result = ExecResult(
            stdout=str(framed.get("stdout") or ""),
            stderr=str(framed.get("stderr") or ""),
            return_code=int(framed.get("return_code") or 0),
        )
        if framed.get("timed_out"):
            raise CommandTimeout(command, result, timeout or 0, what=what)
        if check and result.return_code != 0:
            raise CommandError(command, result, what=what)
        return result

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
        """`command` straight through the environment's `exec`: how stdout and
        stderr come back and what a timeout does depend on the environment.
        For the runner's own plumbing (the relay itself, mkdir / chmod / rm
        around uploads, `service`'s background start)."""
        merged = {**self._exec_env, **(env or {})} if base_env else dict(env or {})
        result = await _exec_raw(
            self.environment, command, env=merged, cwd=cwd, user=user, timeout=timeout
        )
        if check and result.return_code != 0:
            raise CommandError(command, result, what=what)
        return result

    async def _wait(
        self,
        ready: Readiness | str,
        timeout: float,
        pid: int | None = None,
        env: dict[str, str] | None = None,
        user: User = "agent",
    ) -> int:
        """0 once ready, 1 on timeout, 2 if `pid` exited first. One `exec`: the
        polling loop runs in the container (`relay.py wait`)."""
        check = ready if isinstance(ready, Readiness) else cmd_ready(ready)
        result = await relay(
            self,
            "wait",
            {"ready": check.spec(), "timeout": timeout, "pid": pid},
            env=env,
            user=user,
            timeout=timeout + 30,
            check=False,
        )
        return result.return_code if result.return_code in (0, 2) else 1


@dataclass(frozen=True)
class HttpResponse:
    """An HTTP response read inside the container (`relay_http`)."""

    status: int
    headers: dict[str, str]
    body: bytes

    def text(self) -> str:
        return self.body.decode(errors="replace")

    def json(self) -> Any:
        return json.loads(self.body)


async def relay(
    container: Container,
    command: str,
    payload: Any,
    *,
    env: dict[str, str] | None = None,
    user: User = "root",
    timeout: float = 60,
    check: bool = True,
    what: str | None = None,
) -> ExecResult:
    """Run `relay.py <command>` in the container with `payload` as its JSON
    stdin. The payload goes through a temp file the shell deletes before the
    relay starts, so nothing of it shows in `ps` or stays on disk. Runs with
    the container's env (`add_env`) plus `env`."""
    path = await _upload_payload(container.environment, payload)
    try:
        result = await _exec_raw(
            container.environment,
            _relay_shell(command, path),
            env={**container.exec_env, **(env or {})},
            user=user,
            timeout=timeout,
        )
    finally:
        if user != "root":
            # /tmp is sticky: only root removes the file root uploaded.
            await _remove_payload(container.environment, path)
    _check_layer(result)
    if check and result.return_code != 0:
        raise CommandError(f"relay {command}", result, what=what or f"relay {command}")
    return result


async def _exec_raw(
    environment: Environment,
    command: str,
    *,
    env: dict[str, str] | None,
    user: User,
    timeout: float | None,
    cwd: str | None = None,
) -> ExecResult:
    raw = await environment.exec(
        command,
        cwd=cwd,
        env=env or None,
        timeout_sec=math.ceil(timeout) if timeout is not None else None,
        user=None if user == "agent" else user,
    )
    return ExecResult(
        stdout=getattr(raw, "stdout", None) or "",
        stderr=getattr(raw, "stderr", None) or "",
        return_code=int(getattr(raw, "return_code", 0) or 0),
    )


async def _upload_payload(environment: Environment, payload: Any) -> str:
    """A relay's JSON input, as a temp file in the container's /tmp."""
    path = f"/tmp/.{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory(prefix="lmnr-evals-") as tmp:
        local = Path(tmp) / "payload"
        _ = local.write_text(json.dumps(payload))
        await environment.upload_file(local, path)
    return path


async def _remove_payload(environment: Environment, path: str) -> None:
    try:
        _ = await _exec_raw(
            environment, f"rm -f {shlex.quote(path)}", env=None, user="root", timeout=60
        )
    except Exception as e:
        logger.warning(f"Could not remove {path}: {e}")


def _relay_shell(command: str, path: str) -> str:
    """`relay.py <command>` reading `path` on stdin, which the shell deletes
    first (when it may: /tmp is sticky, and root uploaded it)."""
    quoted = shlex.quote(path)
    return f"{{ rm -f {quoted} 2>/dev/null; {relay_command()} {command}; }} < {quoted}"


def _check_layer(result: ExecResult) -> None:
    if result.return_code == 127:
        raise RuntimeError(
            f"{LAYER_DIR} is missing in this container: the runner's helpers run on the "
            + "Laminar layer lmnr-cli adds to every image of an eval.py runner. Rebuild the "
            + f"image with lmnr-cli.\n{(result.stderr or result.stdout)[-OUTPUT_TAIL_CHARS:]}"
        )


async def relay_http(
    container: Container,
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60,
) -> HttpResponse:
    """One HTTP request from inside the container (any status is a result)."""
    result = await relay(
        container,
        "http",
        {
            "method": method,
            "url": url,
            "headers": headers or {},
            "body_b64": base64.b64encode(body).decode() if body is not None else None,
            "timeout": timeout,
        },
        timeout=timeout + 30,
        what=f"{method} {url}",
    )
    response = json.loads(result.stdout)
    return HttpResponse(
        status=int(response["status"]),
        headers=dict(response.get("headers") or {}),
        body=base64.b64decode(response.get("body_b64") or ""),
    )


class Context:
    """`ctx`: what a phase gets. Commands and files in the container are under
    `ctx.container`; everything else here is read on the host.

    - `case`: the task (`Case`), from its directory on the host.
    - `model`: the model to run (`--model`, harbor.yaml, or the runner's `model`).
    - `env`: the runner's `env`, resolved.
    - `logger`: its lines also go to the verifier's log.
    """

    def __init__(
        self,
        *,
        environment: Environment,
        case: Case,
        runner_dir: Path,
        model: str | None = None,
        env: dict[str, str] | None = None,
        exec_env: dict[str, str] | None = None,
        logs_dir: str = AGENT_LOGS_DIR,
    ):
        self.container: Container = Container(
            environment, runner_dir=runner_dir, logs_dir=logs_dir, exec_env=exec_env
        )
        self.case: Case = case
        self.model: str | None = model
        self.env: dict[str, str] = dict(env or {})
        self.logger: EvalLogger = EvalLogger(case.name)
        self.runner_dir: Path = runner_dir
        # "separate" in a verify() whose grading container is fresh, where
        # services aren't running; None in the agent phase.
        self.verifier_mode: VerifierMode | None = None

    def add_env(self, env: Mapping[str, str], exec_env: Mapping[str, str] | None = None) -> None:
        """Add services' env: to `ctx.env`, and to every later command
        (`exec_env`, default the same; Harbor's own `agents[].env` wins anyway)."""
        self.env.update(env)
        self.container.add_env(env if exec_env is None else exec_env)


class ArtifactSaver:
    """`ctx.artifacts` in `setup` / `run` / `collect`: save files for `verify`."""

    def __init__(self, container: Container):
        self._container: Container = container

    async def save(self, path: str, name: str | None = None) -> str:
        """Copy a container file or folder to `/logs/artifacts/<name>` (default:
        its own name), replacing what is there. Harbor collects that folder
        after the agent phase, and `verify` reads it with `ctx.artifacts`
        (also in a separate grading container). Call it in `collect`, so the
        runner's copy replaces anything the agent under test put there.
        Returns the container path it wrote."""
        name = name if name is not None else PurePosixPath(path.rstrip("/")).name
        check_destination("ctx.artifacts.save's name", name)
        target = f"{ARTIFACTS_DIR}/{name.strip('/')}"
        quoted = shlex.quote(target)
        _ = await self._container.sh(
            f"rm -rf {quoted} && mkdir -p {shlex.quote(_parent(target))} && "
            + f"cp -a {shlex.quote(path)} {quoted} && chmod -R a+rX {quoted}",
            user="root",
            timeout=60,
        )
        return target


class AgentContext(Context):
    """`ctx` in `setup`, `run` and `collect`. `ctx.container` is the agent's
    container."""

    def __init__(self, *, host_logs_dir: Path, **kwargs: Any):
        super().__init__(**kwargs)
        # The trial's `agent/` dir on the host, which Harbor syncs with
        # /logs/agent in the container.
        self.host_logs_dir: Path = host_logs_dir
        self.artifacts: ArtifactSaver = ArtifactSaver(self.container)

    def set_result(self, **fields: Any) -> dict[str, Any]:
        """Record fields of the run's outcome (`outcome="completed",
        reply=...`), merged with earlier ones; returns them all. `verify`
        reads them with `ctx.result()`, and Laminar shows them as the trial's
        output when there is no trajectory.

        They go to the trial's agent logs dir on the host, which Harbor syncs
        with the container's /logs/agent, so `verify` reads them in both
        verifier modes.
        """
        path = self.host_logs_dir / RESULT_FILE
        current = _read_json_object(path)
        current.update(fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(json.dumps(current, indent=2, default=str))
        return current

    def result(self) -> dict[str, Any]:
        """What `ctx.set_result(...)` recorded so far (`{}` if nothing)."""
        return _read_json_object(self.host_logs_dir / RESULT_FILE)


class VerifyContext(Context):
    """`ctx` in `verify`, which runs on the host. `ctx.container` is the
    grading container (see `Container`); `result()`, `case`, `secrets`,
    `artifacts` and `mocks` are read on the host, so reference answers and
    keys never have to enter a container."""

    def __init__(
        self,
        *,
        host_agent_dir: Path,
        secrets: Secrets,
        host_artifacts_dir: Path | None = None,
        artifacts: list[str | Artifact] | None = None,
        verifier_mode: VerifierMode = "shared",
        mock_keys: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=VERIFIER_LOGS_DIR, **kwargs)
        self.verifier_mode = verifier_mode
        self.host_agent_dir: Path = host_agent_dir
        self.secrets: Secrets = secrets
        # The trial's artifacts/ dir (next to agent/), filled by Harbor after
        # the agent phase in both verifier modes.
        self.artifacts: Artifacts = Artifacts(
            host_artifacts_dir or host_agent_dir.parent / "artifacts", artifacts or []
        )
        # The runner's mock call log (`ctx.mocks.calls("payments.refund")`).
        self.mocks: MockCalls = MockCalls(self.artifacts, mock_keys or [])

    def result(self) -> dict[str, Any]:
        """What the agent phase recorded with `ctx.set_result(...)` (`{}` if
        nothing)."""
        return _read_json_object(self.host_agent_dir / RESULT_FILE)

    def state(self, service: str, table: str) -> Any:
        """A table a service saved in `collect` (`snapshot=[...]`): the rows of
        `/logs/artifacts/state/<service>/<table>.json`. Read on the host, so it
        works in both verifier modes."""
        path = f"{ARTIFACTS_DIR}/state/{service}/{table}.json"
        if not self.artifacts.exists(path):
            raise FileNotFoundError(
                f"No saved state for {service!r} / {table!r} ({path}): add {table!r} to the "
                + "service's snapshot=[...] so collect saves it."
            )
        return self.artifacts.read_json(path)


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

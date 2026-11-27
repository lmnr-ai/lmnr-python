"""`EvalRunner.mocks` on the host: validation, handler references, and the
mock server's lifecycle in the agent container.

Handlers run in the container (`lmnr.evals.mock_server`), which imports them
by reference, so each must be a top-level function of `eval.py`
(`__runner__:<name>`) or of a module under the evals root
(`_shared.support:<name>`). Before `setup` the runner uploads the runner's
modules, the case's fixtures and the config into a root-only temp dir, starts
the server, waits for its port and deletes the dir; in `collect` it fetches
the call log into `/logs/artifacts/mocks.jsonl` and stops the server.
"""

from __future__ import annotations

import inspect
import json
import re
import secrets
import shlex
import shutil
import sys
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from lmnr.evals.artifacts import ARTIFACTS_DIR
from lmnr.evals.context import relay, relay_http
from lmnr.evals.errors import OUTPUT_TAIL_CHARS, EvalConfigError
from lmnr.evals.layer import relay_command
from lmnr.evals.mock import HttpMock
from lmnr.evals.mock_calls import MOCK_LOG
from lmnr.evals.mock_server import CONFIG_VERSION, HTTP_PREFIX, PROTOCOL_PREFIX, RUNNER_MODULE
from lmnr.evals.paths import evals_root, loaded_modules
from lmnr.evals.readiness import file as file_ready

if TYPE_CHECKING:
    from lmnr.evals.context import AgentContext

MOCK_URL_ENV = "LMNR_MOCK_URL"
SERVER_LOG = "mocks.log"
HOST = "127.0.0.1"
START_TIMEOUT_SEC = 120

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SKIPPED_DIRS = {"site-packages", "dist-packages", "node_modules", "__pycache__"}


@dataclass(frozen=True)
class MockSpec:
    """One entry of `EvalRunner.mocks`, resolved."""

    key: str
    type: Literal["function", "http"]
    # `<module>:<function>`, `__runner__` being eval.py.
    ref: str
    base_url_env: str | None = None
    # HTTP mocks: `host[:port]` and the key's path ("/v1" or "").
    host: str | None = None
    path: str = ""

    def config(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "key": self.key,
            "type": self.type,
            "handler": {"kind": "python", "ref": self.ref},
        }
        if self.host is not None:
            entry["host"] = self.host
        return entry

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"key": self.key, "type": self.type, "handler": self.ref}
        if self.base_url_env is not None:
            out["base_url_env"] = self.base_url_env
        return out

    def base_url(self, port: int) -> str:
        return f"http://{HOST}:{port}{HTTP_PREFIX}{self.host}{self.path}"


def resolve_mocks(owner: str, mocks: Any, eval_path: Path | None) -> list[MockSpec]:
    """Validate `mocks` and resolve each handler to a reference the container
    can import. `eval_path` is the dataset's `eval.py`."""
    if not isinstance(mocks, Mapping):
        raise EvalConfigError(
            f"{owner}.mocks must be a dict of mock key -> handler function, got {mocks!r}."
        )
    if not mocks:
        return []
    if eval_path is None:
        raise EvalConfigError(f"{owner}.mocks: the runner must come from an eval.py file.")
    root = evals_root(eval_path)
    specs: list[MockSpec] = []
    hosts: dict[str, str] = {}
    envs: dict[str, str] = {}
    for key, value in mocks.items():
        where = f"{owner}.mocks[{key!r}]"
        if not isinstance(key, str) or not key or key != key.strip() or _CONTROL.search(key):
            raise EvalConfigError(
                f"{owner}.mocks: {key!r} is not a mock key; use an observed function's span "
                + 'name ("crm.get_customer") or an API origin ("https://api.zendesk.com").'
            )
        is_url = key.lower().startswith(("http://", "https://"))
        if isinstance(value, HttpMock):
            if not is_url:
                raise EvalConfigError(
                    f"{where}: an HTTP mock's key is the API's base URL "
                    + '("https://api.zendesk.com").'
                )
            host, path = _origin(where, key)
            if host in hosts:
                raise EvalConfigError(
                    f"{where}: {hosts[host]!r} already mocks {host}; keep one HTTP mock per host."
                )
            hosts[host] = key
            env = value.base_url_env
            if not isinstance(env, str) or not _ENV_NAME.match(env) or env == MOCK_URL_ENV:
                raise EvalConfigError(
                    f"{where}: base_url_env must be a variable name, got {env!r}."
                )
            if env in envs:
                raise EvalConfigError(f"{where}: {envs[env]!r} already sets {env}.")
            envs[env] = key
            ref = handler_ref(where, value.handler, eval_path, root)
            specs.append(MockSpec(key, "http", ref, base_url_env=env, host=host, path=path))
        else:
            if is_url:
                raise EvalConfigError(
                    f"{where}: wrap an HTTP mock's handler: "
                    + f'mock.http(handler, base_url_env="API_URL"), got {value!r}.'
                )
            specs.append(MockSpec(key, "function", handler_ref(where, value, eval_path, root)))
    return specs


def handler_ref(where: str, handler: Any, eval_path: Path, root: Path) -> str:
    """`__runner__:<name>` or `<module under the evals root>:<name>`."""
    hint = (
        "Handlers run inside the container, which imports them by name: define them with "
        + "`def` at the top level of eval.py or of a module under the evals root "
        + "(evals/_shared/mocks.py)."
    )
    if not inspect.isfunction(handler):
        raise EvalConfigError(f"{where}: the handler must be a function, got {handler!r}. {hint}")
    name = handler.__name__
    if name == "<lambda>":
        raise EvalConfigError(f"{where}: a lambda can't be imported in the container. {hint}")
    if handler.__qualname__ != name:
        raise EvalConfigError(
            f"{where}: {handler.__qualname__} is a method or a nested function. {hint}"
        )
    module = sys.modules.get(handler.__module__)
    file = getattr(module, "__file__", None)
    if module is None or not isinstance(file, str):
        raise EvalConfigError(
            f"{where}: {name} comes from {handler.__module__}, not a file. {hint}"
        )
    if getattr(module, name, None) is not handler:
        raise EvalConfigError(
            f"{where}: {handler.__module__}.{name} is not this function (redefined?). {hint}"
        )
    path = Path(file).resolve()
    if path == eval_path.resolve():
        return f"{RUNNER_MODULE}:{name}"
    if not path.is_relative_to(root) or any(
        part in _SKIPPED_DIRS or part.startswith(".") for part in path.relative_to(root).parts
    ):
        raise EvalConfigError(
            f"{where}: {handler.__module__}.{name} is defined in {path}, outside the evals "
            + f"root {root}, so it isn't uploaded to the container. {hint}"
        )
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        _ = parts.pop()
    expected = ".".join(parts)
    if handler.__module__ != expected:
        raise EvalConfigError(
            f"{where}: {name} comes from module {handler.__module__!r}, but its file "
            + f"{path.relative_to(root)} imports as {expected!r} from the evals root; "
            + f"import it as `from {expected} import {name}`."
        )
    return f"{expected}:{name}"


def _origin(where: str, key: str) -> tuple[str, str]:
    parts = urlsplit(key)
    if not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise EvalConfigError(
            f"{where}: an HTTP mock's key is a base URL without credentials, query or "
            + 'fragment ("https://api.zendesk.com" or "https://api.example.com/v2").'
        )
    host = parts.hostname.lower() + (f":{parts.port}" if parts.port else "")
    return host, parts.path.rstrip("/")


# Lifecycle (LaminarAgent)


@dataclass(frozen=True)
class RunningMocks:
    pid: int
    url: str
    # LMNR_MOCK_URL and every base_url_env, for every later command.
    env: dict[str, str]


async def start_mock_server(
    ctx: AgentContext, specs: list[MockSpec], *, eval_path: Path
) -> RunningMocks:
    """Upload, start, wait for the port, check health, delete the uploaded
    files; a server that started but failed a later step is killed. Runs after
    the services and before `setup`."""
    root = evals_root(eval_path)
    token = secrets.token_urlsafe(24)  # 32 URL-safe characters
    remote = f"/tmp/.{uuid.uuid4().hex[:12]}"
    quoted = shlex.quote(remote)
    fixtures = ctx.case.dir / "tests" / "fixtures"
    config = {
        "version": CONFIG_VERSION,
        "token": token,
        "src_dir": f"{remote}/src",
        "runner_file": eval_path.resolve().relative_to(root).as_posix(),
        "fixtures_dir": f"{remote}/fixtures" if fixtures.is_dir() else None,
        "case": {"name": ctx.case.name, "metadata": ctx.case.metadata},
        "env": ctx.env,
        "mocks": [spec.config() for spec in specs],
        "ready_file": f"{remote}/ready",
    }
    _ = await ctx.container.sh(f"mkdir -m 700 {quoted}", user="root", timeout=60)
    pid: int | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="lmnr-evals-") as tmp:
            local = Path(tmp)
            for rel in loaded_modules(eval_path):
                target = local / "src" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                _ = shutil.copy2(root / rel, target)
            if fixtures.is_dir():
                _ = shutil.copytree(fixtures, local / "fixtures", symlinks=False)
            _ = (local / "c.json").write_text(json.dumps(config, default=str))
            await ctx.container.copy(local, remote)
        # Only `nohup` goes to the background (see `Container.service`).
        started = await ctx.container.sh(
            f"chmod -R go= {quoted} || exit 1; "
            + f"nohup {relay_command()} serve {quoted}/c.json > /dev/null 2> {quoted}/e "
            + "< /dev/null & echo $!",
            user="root",
            timeout=60,
        )
        pid = int(started.stdout.strip().splitlines()[-1])
        waited = await relay(
            ctx.container,
            "wait",
            {
                "ready": file_ready(f"{remote}/ready").spec(),
                "timeout": START_TIMEOUT_SEC,
                "pid": pid,
            },
            timeout=START_TIMEOUT_SEC + 30,
            check=False,
        )
        if waited.return_code != 0:
            err = await ctx.container.sh(f"cat {quoted}/e", user="root", check=False, timeout=60)
            reason = "exited" if waited.return_code == 2 else "did not start in time"
            raise RuntimeError(
                f"The mock server {reason}:\n"
                + (err.stdout[-OUTPUT_TAIL_CHARS:].rstrip() or "(no output)")
            )
        port = int((await ctx.container.sh(f"cat {quoted}/ready", user="root", timeout=60)).stdout)
        url = f"http://{HOST}:{port}{PROTOCOL_PREFIX}{token}"
        health = await relay_http(ctx.container, "GET", f"{url}/health", timeout=30)
        if health.status != 200:
            raise RuntimeError(f"The mock server answered {health.status} to its health check.")
        env = {MOCK_URL_ENV: url}
        for spec in specs:
            if spec.base_url_env is not None:
                env[spec.base_url_env] = spec.base_url(port)
        return RunningMocks(pid=pid, url=url, env=env)
    except BaseException:
        # Not handed to collect_mocks, which stops it otherwise.
        if pid is not None:
            _ = await ctx.container.sh(f"kill {pid}", user="root", check=False, timeout=60)
        raise
    finally:
        # Before `setup`: the agent under test never sees these files.
        _ = await ctx.container.sh(f"rm -rf {quoted}", user="root", check=False, timeout=60)


async def collect_mocks(ctx: AgentContext, running: RunningMocks) -> None:
    """The call log to `/logs/artifacts/mocks.jsonl` (replacing whatever is
    there), the server's log to the trial's `agent/mocks.log`, then stop it."""
    try:
        calls = await relay_http(ctx.container, "GET", f"{running.url}/calls", timeout=60)
        log = await relay_http(ctx.container, "GET", f"{running.url}/log", timeout=60)
        if calls.status != 200:
            raise RuntimeError(f"The mock server answered {calls.status} for its call log.")
        path = f"{ARTIFACTS_DIR}/{MOCK_LOG}"
        _ = await ctx.container.sh(
            f"rm -rf {path} && mkdir -p {ARTIFACTS_DIR}", user="root", timeout=60
        )
        await ctx.container.write_file(path, calls.body)
        _ = await ctx.container.sh(f"chmod a+r {path}", user="root", timeout=60)
        ctx.host_logs_dir.mkdir(parents=True, exist_ok=True)
        _ = (ctx.host_logs_dir / SERVER_LOG).write_bytes(log.body)
    finally:
        _ = await ctx.container.sh(f"kill {running.pid}", user="root", check=False, timeout=60)

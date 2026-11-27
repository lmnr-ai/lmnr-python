"""`Service`: a dependency the runner installs into the image and starts in
every trial, before `setup`.

One object holds both halves, so the image and the startup can't drift apart:
`bundle_stages` / `install_steps` go into the generated Dockerfile (the
`ImageContributor` protocol of `lmnr.evals.image`); `start` → `ready` →
`migrate` → `seed` run in the agent container, and `env()` is added to every
later command. `snapshot` saves state for grading in `collect`.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import shlex
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import override

from lmnr.evals.artifacts import ARTIFACTS_DIR
from lmnr.evals.errors import CommandError, EvalConfigError
from lmnr.evals.image import PathMap
from lmnr.evals.paths import CallSite, caller_site
from lmnr.evals.readiness import Readiness
from lmnr.evals.readiness import cmd as cmd_ready
from lmnr.evals.services.bundle import BUNDLE_DIR, install_check, stage

if TYPE_CHECKING:
    from lmnr.evals.context import AgentContext, Context

# Where `collect` puts service state; `ctx.state(...)` reads it back.
STATE_DIR = f"{ARTIFACTS_DIR}/state"
HOST = "127.0.0.1"

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PLACEHOLDER = re.compile(r"{([a-z_]+)}")

EnvSpec = str | Mapping[str, str] | None


class Service:
    """A dependency declared in `EvalRunner.services`.

    Built-ins: `Postgres`, `ClickHouse`, `Valkey`; `Service.custom(...)` for
    anything else; mirrors (`lmnr.evals.mirrors`) for SaaS stand-ins.
    Subclasses override the hooks below; keep this interface small and stable.

    `env` maps variable names to templates over `template_values()`
    (`{"DATABASE_URL": "{url}"}`); a plain string names the variable that gets
    `{url}`.
    """

    # Shown in provenance comments and errors.
    display_name: ClassVar[str] = "Service"
    # Seconds `start` waits for `ready()`.
    start_timeout: ClassVar[float] = 120
    # The argument that makes `collect` save state for `ctx.state(...)`
    # (`"snapshot=[...]"`), for errors; None when the service saves none.
    snapshot_option: ClassVar[str | None] = None

    def __init__(
        self,
        name: str,
        version: str,
        *,
        env: EnvSpec = None,
        source: str | None = None,
    ):
        self.name: str = name
        self.version: str = version
        self.source: str | None = source
        self.env_spec: dict[str, str] = _env_templates(env)
        # Where it was declared: relative paths resolve there.
        self.site: CallSite | None = caller_site()
        self.base_dir: Path | None = self.site.file.parent if self.site else None

    # Build half

    def image_title(self, root: Path | None) -> str:
        where = f" ({self.site.label(root, owner=False)})" if self.site else ""
        return f"service: {self.display_name} {self.version}{where}"

    def image_step(self) -> str:
        return "service"

    def host_paths(self) -> list[Path]:
        """Host files the stages copy (for the build context)."""
        return []

    def bundle_stages(self, rel: PathMap) -> list[str]:
        """Dockerfile stages producing what the final image copies."""
        return []

    def install_steps(self, rel: PathMap) -> list[str]:
        """Final-image lines (`COPY --from`, users, dirs)."""
        return []

    # Run half

    async def start(self, ctx: AgentContext) -> None:
        """Start the process and wait for `ready()`."""
        _ = await ctx.container.spawn(
            self.name,
            self.start_command(),
            ready=self.ready(),
            timeout=self.start_timeout,
            user="root",
        )

    def start_command(self) -> str:
        raise NotImplementedError

    def ready(self) -> Readiness:
        raise NotImplementedError

    async def migrate(self, ctx: AgentContext) -> None:
        """Apply the schema, after `ready()`."""

    async def seed(self, ctx: AgentContext) -> None:
        """Load this case's data, after `migrate`."""

    def env(self) -> dict[str, str]:
        """Added to every command the runner runs after services started."""
        values = self.template_values()
        return {key: _render(template, values) for key, template in self.env_spec.items()}

    def template_values(self) -> dict[str, str]:
        """What `env` templates can use (`{url}`, `{port}`, ...)."""
        return {"host": HOST}

    async def snapshot(self, ctx: AgentContext, out_dir: str) -> None:
        """Save state under `out_dir` (`/logs/artifacts/state/<name>`) in
        `collect`, for grading in a separate verifier container."""

    # Configuration

    def ports(self) -> list[int]:
        """Ports it listens on (two services can't share one)."""
        return []

    def validate(self, owner: str) -> list[str]:
        """Raise `EvalConfigError` if it can't work; return warnings."""
        where = f"{owner}.services[{self.name!r}]"
        if not isinstance(self.name, str) or not _NAME.match(self.name):
            raise EvalConfigError(
                f"{owner}.services: name {self.name!r} must be lowercase letters, digits, "
                + "'.', '_' or '-' (it names the log file and the state folder)."
            )
        if not isinstance(self.version, str) or not self.version.strip():
            raise EvalConfigError(
                f'{where} needs a version, e.g. "16"; "latest" works but isn\'t reproducible.'
            )
        values = self.template_values()
        for key, template in self.env_spec.items():
            if not _ENV_NAME.match(key):
                raise EvalConfigError(f"{where}: env has an invalid variable name: {key!r}.")
            for placeholder in _PLACEHOLDER.findall(template):
                if placeholder not in values:
                    raise EvalConfigError(
                        f"{where}: env[{key!r}] uses {{{placeholder}}}; available: "
                        + ", ".join(f"{{{v}}}" for v in sorted(values))
                        + "."
                    )
        if self.version.strip() == "latest":
            return [
                f'{where} uses version "latest": trials run against whatever the image '
                + "has when it is built. Pin a version for reproducible runs."
            ]
        return []

    def describe(self, root: Path | None) -> dict[str, Any]:
        """For `lmnr-cli eval describe`."""
        return {
            "name": self.name,
            "kind": self.display_name,
            "version": self.version,
            "source": self.source_image(),
            "env": sorted(self.env_spec),
            "origin": self.site.label(root) if self.site else None,
        }

    def source_image(self) -> str | None:
        """The image the service is installed from."""
        return self.source

    def stage_name(self, suffix: str = "") -> str:
        return f"svc-{self.name}{suffix}"

    @staticmethod
    def custom(
        name: str,
        version: str,
        *,
        start: str,
        ready: Readiness | str,
        source: str | None = None,
        bundle: Sequence[str] = (),
        install: Sequence[str] = (),
        env: EnvSpec = None,
        port: int | None = None,
        user: str = "root",
    ) -> CustomService:
        """Any other dependency.

        Install it one of two ways: `bundle=[paths]` copies those paths from
        the `source` image (plus the shared libraries they need, which `start`
        and `ready` get on `LD_LIBRARY_PATH`), or `install=[Dockerfile lines]`
        adds lines to the final image (they re-run when the app's layers above
        them change). `start` runs as `user`; `env` templates can use `{port}`
        and `{url}` (`http://127.0.0.1:<port>`) when `port` is set.
        """
        return CustomService(
            name,
            version,
            start=start,
            ready=ready,
            source=source,
            bundle=bundle,
            install=install,
            env=env,
            port=port,
            user=user,
        )

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, version={self.version!r})"


class CustomService(Service):
    display_name: ClassVar[str] = "custom"

    def __init__(
        self,
        name: str,
        version: str,
        *,
        start: str,
        ready: Readiness | str,
        source: str | None,
        bundle: Sequence[str],
        install: Sequence[str],
        env: EnvSpec,
        port: int | None,
        user: str,
    ):
        super().__init__(name, version, env=env, source=source)
        self.start_cmd: str = start
        self.readiness: Readiness = ready if isinstance(ready, Readiness) else cmd_ready(ready)
        self.bundle_paths: list[str] = list(bundle)
        self.install_lines: list[str] = list(install)
        self.port: int | None = port
        self.user: str = user

    @override
    def image_title(self, root: Path | None) -> str:
        where = f" ({self.site.label(root, owner=False)})" if self.site else ""
        return f"service: {self.name} {self.version}{where}"

    @override
    def describe(self, root: Path | None) -> dict[str, Any]:
        return {**super().describe(root), "kind": "custom"}

    @property
    def lib_dir(self) -> str:
        return f"/usr/lib/{self.name}-libs"

    @override
    def bundle_stages(self, rel: PathMap) -> list[str]:
        if not self.bundle_paths or self.source is None:
            return []
        quoted = " ".join(shlex.quote(p) for p in self.bundle_paths)
        script = f"""for p in {quoted}; do
  mkdir -p "$out$(dirname "$p")"
  cp -a "$p" "$out$p"
done
collect_deps {self.lib_dir} $(find "$out" -type f -perm -u+x)
record_glibc {self.name}"""
        return stage(self.source, self.stage_name(), script)

    @override
    def install_steps(self, rel: PathMap) -> list[str]:
        lines: list[str] = []
        if self.bundle_paths and self.source is not None:
            lines.append(f"COPY --from={self.stage_name()} {BUNDLE_DIR}/ /")
            lines.append(install_check(self.name, self.name, self.source, []))
        return lines + self.install_lines

    @override
    def start_command(self) -> str:
        return self.start_cmd

    @override
    async def start(self, ctx: AgentContext) -> None:
        env = {"LD_LIBRARY_PATH": self.lib_dir} if self.bundle_paths else None
        _ = await ctx.container.spawn(
            self.name,
            self.start_cmd,
            ready=self.readiness,
            env=env,
            timeout=self.start_timeout,
            user=self.user,
        )

    @override
    def ready(self) -> Readiness:
        return self.readiness

    @override
    def template_values(self) -> dict[str, str]:
        values = super().template_values()
        if self.port is not None:
            values.update(port=str(self.port), url=f"http://{HOST}:{self.port}")
        return values

    @override
    def ports(self) -> list[int]:
        return [self.port] if self.port is not None else []

    @override
    def validate(self, owner: str) -> list[str]:
        warnings = super().validate(owner)
        where = f"{owner}.services[{self.name!r}]"
        if self.bundle_paths and self.source is None:
            raise EvalConfigError(f"{where}: bundle= copies from an image; set source=.")
        if not self.bundle_paths and not self.install_lines and self.source is not None:
            raise EvalConfigError(f"{where}: source= needs bundle=[paths] to copy from it.")
        if not all(isinstance(p, str) and p.startswith("/") for p in self.bundle_paths):
            raise EvalConfigError(f"{where}: bundle= takes absolute paths in the source image.")
        if not isinstance(self.start_cmd, str) or not self.start_cmd.strip():
            raise EvalConfigError(f"{where}: start= must be a shell command.")
        return warnings


# Lifecycle, driven by the Harbor adapters


async def start_services(services: Sequence[Service], ctx: AgentContext) -> dict[str, str]:
    """Start every service in parallel (each: start → ready → migrate → seed)
    and return their env, which the caller adds to every later command."""

    async def bring_up(service: Service) -> None:
        await service.start(ctx)
        await service.migrate(ctx)
        await service.seed(ctx)

    _ = await asyncio.gather(*(bring_up(service) for service in services))
    env: dict[str, str] = {}
    for service in services:
        env.update(service.env())
    return env


async def snapshot_services(services: Sequence[Service], ctx: AgentContext) -> None:
    for service in services:
        await service.snapshot(ctx, f"{STATE_DIR}/{service.name}")


# Helpers for subclasses


def require_live(service: Service, ctx: Context) -> None:
    """Live helpers need the running service: not in a separate verifier."""
    if getattr(ctx, "verifier_mode", None) == "separate":
        raise EvalConfigError(
            f"{service.name} isn't running in separate verifier mode: declare "
            + f"snapshot=[...] on it and read the state with ctx.state({service.name!r}, ...)."
        )


async def run_script(
    ctx: Context, script: str, *, what: str, timeout: float = 300, user: str = "root"
) -> str:
    """Run a shell script in the container through a temp file, so its size
    isn't limited by the command line; returns stdout."""
    path = f"/tmp/.{uuid.uuid4().hex[:12]}.sh"
    await ctx.container.write_file(path, script)
    result = await ctx.container.sh(
        f"bash {path}; rc=$?; rm -f {path}; exit $rc",
        user=user,
        timeout=timeout,
        check=False,
    )
    if result.return_code != 0:
        raise CommandError(what, result, what=what)
    return result.stdout


def heredoc(command: str, text: str) -> str:
    """`command <<'DELIM'` feeding `text` on stdin. The delimiter comes from
    the text's hash, so it can't appear in it and the same text always gives
    the same line (build-time scripts are part of the Dockerfile's cache key)."""
    delimiter = f"EOF_{hashlib.sha256(text.encode()).hexdigest()[:12].upper()}"
    return f"{command} <<'{delimiter}'\n{text.rstrip(chr(10))}\n{delimiter}"


def _env_templates(env: EnvSpec) -> dict[str, str]:
    if env is None:
        return {}
    if isinstance(env, str):
        return {env: "{url}"}
    if isinstance(env, Mapping):
        return {str(k): str(v) for k, v in env.items()}
    raise EvalConfigError(f"Service env must be a variable name or a dict, got {env!r}.")


def _render(template: str, values: dict[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), template)

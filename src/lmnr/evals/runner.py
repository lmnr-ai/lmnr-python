"""`EvalRunner`: a dataset's agent and verifier in one class, in `agent/eval.py`."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, TypeVar, overload

from lmnr.evals.artifacts import Artifact, validate_artifacts
from lmnr.evals.context import AgentContext, VerifyContext
from lmnr.evals.dependencies import check_mock_dependencies
from lmnr.evals.errors import EvalConfigError
from lmnr.evals.image import Image
from lmnr.evals.mirrors.base import Mirror
from lmnr.evals.mocks.host import MockSpec, resolve_mocks
from lmnr.evals.mocks.mock import ProjectDependencies
from lmnr.evals.paths import add_evals_root_to_path
from lmnr.evals.services.base import Service

Rewards = dict[str, float | int | bool] | float | int | bool

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

S = TypeVar("S", bound="Service")


@dataclass(frozen=True)
class Secret:
    """A secret in the runner's `env`, by name (see `secret`)."""

    name: str


def secret(name: str) -> Secret:
    """Pass a secret into the container env: `env = {"KEY": secret("KEY")}`.

    That declares `name` as a secret (no need to list it in `secrets` too), so
    `verify` can read it as well (`ctx.secrets[name]`). lmnr-cli resolves it
    from the shell, `.env.local` or `.env`; the value never appears in `eval.py`.
    """
    return Secret(name)


class EvalRunner:
    """A dataset's agent and verifier. Subclass it once in `agent/eval.py`.

    Your code runs on the host (your machine, or Laminar's orchestrator for
    `--remote`); it reaches the trial's container through `ctx.container`.

    Class attributes:

    - `image`: an `Image` (`Image.python("3.13").add_dir(...)`,
      `Image.from_dockerfile(...)`, `Image.from_registry(...)`), or a dict
      of named images (the first is the default; a task picks one with
      `docker_image` in its `task.toml`).
    - `model`: default model; `harbor.yaml` and `--model` override it.
    - `env`: env for every command in the container; `secret("NAME")` passes
      a secret, which lmnr-cli resolves from your shell or `.env` files.
    - `secrets`: secrets that must NOT reach the container (a judge's key):
      lmnr-cli resolves them too, and `verify` reads them, like the `env` ones,
      as `ctx.secrets[name]`.
    - `new_task_toml`: `task.toml` tables lmnr-cli writes into tasks that have
      no `task.toml` yet (`{"agent": {"timeout_sec": 300}}`).
    - `run_command`: a shell command `run` runs in the container, instead of
      defining `run`.
    - `services`: `Postgres(...)`, `ClickHouse(...)`, `Valkey(...)`,
      `Service.custom(...)` or mirrors (`lmnr.evals.mirrors`). Installed into
      the image, then started, migrated and seeded before `setup` in every
      trial; their `env()` (`DATABASE_URL`, ...) reaches every later command.
      Use their helpers with `self.service(Postgres)`.
    - `artifacts`: container paths (or `Artifact(source, destination=...)`)
      Harbor collects after the agent phase; `verify` reads them with
      `ctx.artifacts`, also when it grades in a separate container.
    - `mocks`: `{key: handler}`. A key is an observed function's span name
      (`"crm.get_customer"`, handler `(call, ctx)`), or an API's base URL
      with `mock.http(handler, base_url_env="API_URL")` (handler `(req,
      ctx)`). A mock server runs them in the container, from before `setup`
      to `collect`; `verify` reads the calls with `ctx.mocks.calls(key)`.
      Handlers are top-level functions of `eval.py` or of a module under the
      evals root.
    - `mock_dependencies`: what the mock server's Python installs besides
      `lmnr`. It imports `eval.py` in the container, so whatever `eval.py`,
      its shared modules and the handlers import at the top level must be
      there (`anthropic` for a judge). Default (None): the evals project's
      dependencies, which run `eval.py` on the host too: the evals root's
      `pyproject.toml` `[project] dependencies` and `eval.py`'s PEP 723
      `# /// script` block (a `[tool.uv.sources]` package or a local path is
      left out, with a warning). A list replaces them; put
      `mock.PROJECT_DEPENDENCIES` in it to extend them instead
      (`[mock.PROJECT_DEPENDENCIES, "httpx>=0.27"]`). Only runners with
      `mocks` install anything.
    - `grade_with_test_sh = True`: grade each case with its own
      `tests/test.sh`, run in the container by Harbor's default verifier,
      instead of `verify()` (for imported Harbor datasets that ship one).

    Phases, each with the trial's `ctx`:

    - `setup(ctx)`: before `run`. Default: the case's `environment/seed.*`, if any.
    - `run(ctx)`: the code under test. Required, unless `run_command` is set.
    - `collect(ctx)`: after `run`, also when it failed (not when `setup`
      failed). Default: nothing.
    - `verify(ctx)`: grading, on the host. Returns a number, a bool or a dict
      of them. Required, unless `grade_with_test_sh = True`.
    """

    image: ClassVar[Image | dict[str, Image] | None] = None
    model: ClassVar[str | None] = None
    secrets: ClassVar[list[str]] = []
    env: ClassVar[dict[str, str | Secret]] = {}
    new_task_toml: ClassVar[dict[str, Any]] = {}
    run_command: ClassVar[str | None] = None
    grade_with_test_sh: ClassVar[bool] = False
    artifacts: ClassVar[list[str | Artifact]] = []
    services: ClassVar[list[Service]] = []
    mocks: ClassVar[dict[str, Any]] = {}
    mock_dependencies: ClassVar[list[str | ProjectDependencies] | None] = None

    async def setup(self, ctx: AgentContext) -> None:
        seed = ctx.case.seed_file()
        if seed is not None:
            _ = await ctx.container.run_file(seed)

    async def run(self, ctx: AgentContext) -> None:
        if not self.run_command:
            raise EvalConfigError(f"{type(self).__name__} must define run() or set `run_command`.")
        _ = await ctx.container.sh(self.run_command)

    async def collect(self, ctx: AgentContext) -> None:
        pass

    async def verify(self, ctx: VerifyContext) -> Rewards:
        raise EvalConfigError(
            f"{type(self).__name__} must define verify(), or set grade_with_test_sh = True "
            + "to grade with each task's tests/test.sh."
        )

    # Services

    @overload
    def service(self, key: type[S]) -> S: ...

    @overload
    def service(self, key: str) -> Service: ...

    def service(self, key: type[Service] | str) -> Service:
        """A declared service, by class (`self.service(Postgres)`) or by name
        (`self.service("postgres")`), for its helpers."""
        if isinstance(key, str):
            found = [s for s in self.services if s.name == key]
        else:
            found = [s for s in self.services if isinstance(s, key)]
        what = key if isinstance(key, str) else key.__name__
        if not found:
            names = ", ".join(repr(s.name) for s in self.services) or "none"
            raise EvalConfigError(
                f"{type(self).__name__} declares no service {what!r} (services: {names})."
            )
        if len(found) > 1:
            names = ", ".join(repr(s.name) for s in found)
            raise EvalConfigError(
                f"{type(self).__name__} declares several {what} services ({names}); "
                + 'pick one by name: self.service("<name>").'
            )
        return found[0]

    def mirror(self, name: str) -> Mirror:
        """A declared mirror, by name: `self.mirror("slack").state(ctx)`."""
        service = self.service(name)
        if not isinstance(service, Mirror):
            raise EvalConfigError(f"Service {name!r} is not a mirror; use self.service({name!r}).")
        return service

    # Configuration, as read by lmnr-cli and the Harbor adapters

    @classmethod
    def images(cls) -> dict[str | None, Image]:
        """Named images; a single `image` has the name None (lmnr-cli names it)."""
        if isinstance(cls.image, Image):
            return {None: cls.image}
        return {name: image for name, image in (cls.image or {}).items()}

    @classmethod
    def secret_names(cls) -> list[str]:
        """Every secret: `secrets` plus each `secret(...)` in `env`, once each."""
        names = list(cls.secrets)
        for value in cls.env.values():
            if isinstance(value, Secret) and value.name not in names:
                names.append(value.name)
        return names

    @classmethod
    def resolved_env(cls, values: dict[str, str]) -> dict[str, str]:
        """`env` with secrets read from `values`. Unset secrets are left out."""
        resolved: dict[str, str] = {}
        for key, value in cls.env.items():
            if isinstance(value, Secret):
                if values.get(value.name) is not None:
                    resolved[key] = values[value.name]
            else:
                resolved[key] = value
        return resolved

    @classmethod
    def validate(cls) -> list[str]:
        """Raise `EvalConfigError` if the runner can't run; return warnings."""
        name = cls.__name__
        warnings: list[str] = []

        image = cls.image
        if image is None:
            raise EvalConfigError(
                f'{name} has no image. Set `image = Image.python("3.13").add_dir(...)`, '
                + '`Image.from_dockerfile("Dockerfile")` or '
                + '`Image.from_registry("python:3.13-slim")`.'
            )
        if isinstance(image, dict):
            if not image:
                raise EvalConfigError(f"{name}.image is an empty dict; name at least one image.")
            for key, value in image.items():
                if not isinstance(key, str) or not isinstance(value, Image):
                    raise EvalConfigError(
                        f"{name}.image must map image names to Image objects, got "
                        + f"{key!r}: {value!r}."
                    )
        elif not isinstance(image, Image):
            raise EvalConfigError(
                f"{name}.image must be an Image or a dict of them, got {image!r}."
            )

        defines_run = cls.run is not EvalRunner.run
        if not defines_run and not cls.run_command:
            raise EvalConfigError(
                f'{name} must define `async def run(self, ctx)` or set `run_command = "..."`.'
            )
        if defines_run and cls.run_command:
            warnings.append(f"{name} defines run(), so `run_command` is ignored.")

        if not isinstance(cls.grade_with_test_sh, bool):
            raise EvalConfigError(
                f"{name}.grade_with_test_sh must be True or False, got {cls.grade_with_test_sh!r}."
            )
        defines_verify = cls.verify is not EvalRunner.verify
        if cls.grade_with_test_sh and defines_verify:
            raise EvalConfigError(
                f"{name} defines verify() and sets grade_with_test_sh = True; keep one: "
                + "verify() grades on the host, grade_with_test_sh runs each task's "
                + "tests/test.sh in the container."
            )
        if not cls.grade_with_test_sh and not defines_verify:
            raise EvalConfigError(
                f"{name} must define `async def verify(self, ctx)`, or set "
                + "grade_with_test_sh = True to grade with each task's tests/test.sh."
            )

        secrets = cls.secrets
        if not isinstance(secrets, (list, tuple)) or not all(
            isinstance(s, str) and _ENV_NAME.match(s) for s in secrets
        ):
            raise EvalConfigError(
                f"{name}.secrets must be a list of env var names (no values), got {secrets!r}."
            )
        if not isinstance(cls.env, dict):
            raise EvalConfigError(f"{name}.env must be a dict, got {cls.env!r}.")
        for key, value in cls.env.items():
            if not isinstance(key, str) or not _ENV_NAME.match(key):
                raise EvalConfigError(f"{name}.env has an invalid variable name: {key!r}.")
            if isinstance(value, Secret):
                if not isinstance(value.name, str) or not _ENV_NAME.match(value.name):
                    raise EvalConfigError(
                        f"{name}.env[{key!r}]: secret() takes an env var name, "
                        + f"got {value.name!r}."
                    )
            elif not isinstance(value, str):
                raise EvalConfigError(
                    f"{name}.env[{key!r}] must be a string or secret(...), got {value!r}."
                )

        if cls.model is not None and not isinstance(cls.model, str):
            raise EvalConfigError(f"{name}.model must be a string, got {cls.model!r}.")
        if not isinstance(cls.new_task_toml, dict):
            raise EvalConfigError(f"{name}.new_task_toml must be a dict of task.toml tables.")
        try:
            _ = json.dumps(cls.new_task_toml)
        except (TypeError, ValueError) as e:
            raise EvalConfigError(f"{name}.new_task_toml must be plain data: {e}") from e

        validate_artifacts(name, cls.artifacts)
        warnings += _validate_services(cls)

        specs = cls.mock_specs()
        check_mock_dependencies(name, cls.mock_dependencies)
        taken = {key: f"{name}.env" for key in cls.env}
        for service in cls.services:
            taken.update({key: f"service {service.name!r}" for key in service.env_spec})
        for spec in specs:
            if spec.base_url_env in taken:
                raise EvalConfigError(
                    f"{name}.mocks[{spec.key!r}]: base_url_env {spec.base_url_env} is also set "
                    + f"by {taken[spec.base_url_env]}; keep one."
                )
        return warnings

    @classmethod
    def mock_specs(cls) -> list[MockSpec]:
        """`mocks`, validated, with each handler's reference for the container."""
        return resolve_mocks(cls.__name__, cls.mocks, cls.eval_file())

    @classmethod
    def eval_file(cls) -> Path | None:
        """The `eval.py` that defines this runner."""
        file = getattr(sys.modules.get(cls.__module__), "__file__", None)
        return Path(file).resolve() if isinstance(file, str) else None


def _validate_services(cls: type[EvalRunner]) -> list[str]:
    name = cls.__name__
    services = cls.services
    if not isinstance(services, (list, tuple)) or not all(isinstance(s, Service) for s in services):
        raise EvalConfigError(
            f"{name}.services must be a list of services (Postgres(...), Valkey(...), "
            + f"Service.custom(...), ...), got {services!r}."
        )
    warnings: list[str] = []
    names: set[str] = set()
    ports: dict[int, str] = {}
    env_owner: dict[str, str] = {key: f"{name}.env" for key in cls.env}
    for service in services:
        warnings += service.validate(name)
        if service.name in names:
            raise EvalConfigError(
                f"{name}.services has two services named {service.name!r}; pass name=... to one."
            )
        names.add(service.name)
        for number in service.ports():
            if number in ports:
                raise EvalConfigError(
                    f"{name}.services: {service.name!r} and {ports[number]!r} both use port "
                    + f"{number}; pass port=... to one."
                )
            ports[number] = service.name
        for key in service.env_spec:
            if key in env_owner:
                raise EvalConfigError(
                    f"{name}: {key} is set by both service {service.name!r} and "
                    + f"{env_owner[key]}; keep one (a service's env=... renames its variable)."
                )
            env_owner[key] = f"service {service.name!r}"
    return warnings


# Loading eval.py

_modules: dict[Path, ModuleType] = {}


def load_module(path: Path | str) -> ModuleType:
    """Import `eval.py` by path, once per process."""
    path = Path(path).resolve()
    if path in _modules:
        return _modules[path]
    if not path.is_file():
        raise EvalConfigError(f"{path} does not exist.")
    # `from _shared.support import SupportRunner` (shared runners).
    add_evals_root_to_path(path)
    module_name = f"lmnr_eval_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise EvalConfigError(f"Can't import {path}.")
    module = importlib.util.module_from_spec(spec)
    # Registered before running it, as `import` does, so dataclasses and
    # pickling in the user's module can find it.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    _modules[path] = module
    return module


def find_runner(module: ModuleType, name: str | None = None) -> type[EvalRunner]:
    """The `EvalRunner` subclass defined in `module` (exactly one unless `name`)."""
    runners = [
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and issubclass(value, EvalRunner)
        and value is not EvalRunner
        and value.__module__ == module.__name__
    ]
    where = getattr(module, "__file__", module.__name__)
    if name is not None:
        for runner in runners:
            if runner.__name__ == name:
                return runner
        raise EvalConfigError(f"{where} defines no EvalRunner subclass named {name!r}.")
    if not runners:
        raise EvalConfigError(
            f"{where} defines no EvalRunner subclass. Add `class MyEval(EvalRunner): ...`."
        )
    if len(runners) > 1:
        names = ", ".join(sorted(r.__name__ for r in runners))
        raise EvalConfigError(f"{where} defines several EvalRunner subclasses ({names}); keep one.")
    return runners[0]


def load_runner(ref: str, base_dir: Path | str | None = None) -> tuple[type[EvalRunner], Path]:
    """`"agent/eval.py:SupportEval"` (relative to `base_dir`, default the cwd) to
    the runner class and the path of its file."""
    path_part, _, name = ref.rpartition(":") if ":" in ref else (ref, "", "")
    path = Path(path_part)
    if not path.is_absolute():
        path = Path(base_dir or Path.cwd()) / path
    path = path.resolve()
    runner = find_runner(load_module(path), name or None)
    return runner, path


def default_image_name(eval_path: Path) -> str:
    """A single `image` is named after the dataset folder (`<folder>-evals`)."""
    return f"{eval_path.resolve().parent.parent.name}-evals"

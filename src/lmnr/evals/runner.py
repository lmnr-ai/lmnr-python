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
from typing import Any, ClassVar

from lmnr.evals.artifacts import Artifact, validate_artifacts
from lmnr.evals.context import AgentContext, VerifyContext
from lmnr.evals.errors import EvalConfigError
from lmnr.evals.image import Image
from lmnr.evals.paths import add_evals_root_to_path

Rewards = dict[str, float | int | bool] | float | int | bool

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Secret:
    """A reference to one of the runner's `secrets`, by name (see `secret`)."""

    name: str


def secret(name: str) -> Secret:
    """Pass a declared secret into the container env: `env = {"KEY": secret("KEY")}`.

    `name` must be in the runner's `secrets`. lmnr-cli resolves it from the shell,
    `.env.local` or `.env`; the value never appears in `eval.py`.
    """
    return Secret(name)


class EvalRunner:
    """A dataset's agent and verifier. Subclass it once in `agent/eval.py`.

    Your code runs on the host (your machine, or Laminar's orchestrator for
    `--remote`); it reaches the trial's container through `ctx.container`.

    Class attributes:

    - `image`: `Image.from_dockerfile(...)` / `Image.from_registry(...)`, or a dict
      of named images (the first is the default; a task picks one with
      `docker_image` in its `task.toml`).
    - `model`: default model; `harbor.yaml` and `--model` override it.
    - `secrets`: env var names lmnr-cli must resolve. `verify` reads them as
      `ctx.secrets[name]`; the container only gets the ones `env` passes on.
    - `env`: env for every command in the container; `secret("NAME")` passes
      a secret.
    - `new_task_toml`: `task.toml` tables lmnr-cli writes into tasks that have
      no `task.toml` yet (`{"agent": {"timeout_sec": 300}}`).
    - `run_command`: a shell command `run` runs in the container, instead of
      defining `run`.
    - `artifacts`: container paths (or `Artifact(source, destination=...)`)
      Harbor collects after the agent phase; `verify` reads them with
      `ctx.artifacts`, also when it grades in a separate container.
    - `grade_with_test_sh = True`: grade each case with its own
      `tests/test.sh`, run in the container by Harbor's default verifier,
      instead of `verify()` (for imported Harbor datasets that ship one).

    Phases, each with the trial's `ctx`:

    - `setup(ctx)`: before `run`. Default: the case's `environment/seed.*`, if any.
    - `run(ctx)`: the code under test. Required, unless `run_command` is set.
    - `collect(ctx)`: after `run`, also when it failed. Default: nothing.
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
    # Not supported yet; they will start databases and mock servers before `setup`.
    services: ClassVar[list[Any]] = []
    mocks: ClassVar[dict[str, Any]] = {}

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

    # Configuration, as read by lmnr-cli and the Harbor adapters

    @classmethod
    def images(cls) -> dict[str | None, Image]:
        """Named images; a single `image` has the name None (lmnr-cli names it)."""
        if isinstance(cls.image, Image):
            return {None: cls.image}
        return {name: image for name, image in (cls.image or {}).items()}

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
                f'{name} has no image. Set `image = Image.from_dockerfile("Dockerfile")` '
                + 'or `image = Image.from_registry("python:3.13-slim")`.'
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
                if value.name not in secrets:
                    raise EvalConfigError(
                        f"{name}.env[{key!r}] uses secret({value.name!r}), which is not in "
                        + f"{name}.secrets. Add {value.name!r} to `secrets`."
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

        if cls.services:
            raise EvalConfigError(f"{name}.services is not supported yet.")
        if cls.mocks:
            raise EvalConfigError(f"{name}.mocks is not supported yet.")
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

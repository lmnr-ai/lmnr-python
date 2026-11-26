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
from typing import Any, ClassVar, Literal

from lmnr.evals.context import AgentContext, VerifyContext
from lmnr.evals.errors import EvalConfigError
from lmnr.evals.image import Image

VerifyMode = Literal["runner", "task"]
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

    Class attributes:

    - `image`: `Image.from_dockerfile(...)` / `Image.from_registry(...)`, or a dict
      of named images (the first is the default; a task picks one with
      `docker_image` in its `task.toml`).
    - `model`: default model; `harbor.yaml` and `--model` override it.
    - `secrets`: env var names lmnr-cli must resolve. `verify` reads them as
      `ctx.secrets[name]`; the container only gets the ones `env` passes on.
    - `env`: env for every command in the container; `secret("NAME")` passes
      a secret.
    - `task_template`: `task.toml` keys lmnr-cli writes into new tasks.
    - `entrypoint`: a shell command to use instead of defining `run`.
    - `verify = "task"`: grade with each task's own `tests/test.sh` instead of
      `verify()`.

    Phases, each with the trial's `ctx`:

    - `setup(ctx)`: before `run`. Default: the case's `environment/seed.*`, if any.
    - `run(ctx)`: the code under test. Required, unless `entrypoint` is set.
    - `collect(ctx)`: after `run`, also when it failed. Default: nothing.
    - `verify(ctx)`: grading, in the Harbor process. Returns a number, a bool or
      a dict of them.
    """

    image: ClassVar[Image | dict[str, Image] | None] = None
    model: ClassVar[str | None] = None
    secrets: ClassVar[list[str]] = []
    env: ClassVar[dict[str, str | Secret]] = {}
    task_template: ClassVar[dict[str, Any]] = {}
    entrypoint: ClassVar[str | None] = None
    # Not supported yet; they will start databases and mock servers before `setup`.
    services: ClassVar[list[Any]] = []
    mocks: ClassVar[dict[str, Any]] = {}

    async def setup(self, ctx: AgentContext) -> None:
        seed = ctx.case.seed_file()
        if seed is not None:
            _ = await ctx.run_file(seed)

    async def run(self, ctx: AgentContext) -> None:
        if not self.entrypoint:
            raise EvalConfigError(f"{type(self).__name__} must define run() or set `entrypoint`.")
        _ = await ctx.sh(self.entrypoint)

    async def collect(self, ctx: AgentContext) -> None:
        pass

    async def verify(self, ctx: VerifyContext) -> Rewards:
        raise EvalConfigError(
            f'{type(self).__name__} must define verify(), or set verify = "task" to grade '
            + "with each task's tests/test.sh."
        )

    # Configuration, as read by lmnr-cli and the Harbor adapters

    @classmethod
    def verify_mode(cls) -> VerifyMode:
        return "task" if getattr(cls, "verify", None) == "task" else "runner"

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
        if not defines_run and not cls.entrypoint:
            raise EvalConfigError(
                f'{name} must define `async def run(self, ctx)` or set `entrypoint = "..."`.'
            )
        if defines_run and cls.entrypoint:
            warnings.append(f"{name} defines run(), so `entrypoint` is ignored.")

        verify = getattr(cls, "verify", None)
        if isinstance(verify, str) and verify != "task":
            raise EvalConfigError(f'{name}.verify must be a method or "task", got {verify!r}.')
        if verify is EvalRunner.verify:
            raise EvalConfigError(
                f'{name} must define `async def verify(self, ctx)`, or set verify = "task" '
                + "to grade with each task's tests/test.sh."
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
        if not isinstance(cls.task_template, dict):
            raise EvalConfigError(f"{name}.task_template must be a dict of task.toml tables.")
        try:
            _ = json.dumps(cls.task_template)
        except (TypeError, ValueError) as e:
            raise EvalConfigError(f"{name}.task_template must be plain data: {e}") from e

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

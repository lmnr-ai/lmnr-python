"""The image a dataset's trials start from. lmnr-cli builds it.

`Image` is a builder, like Modal's: a base, then steps, each returning a new
`Image`, so `base.apt_install(...)` is a variant that shares `base`'s layers.
`compile_image` turns one (plus the runner's services) into Dockerfile text and
a build context; describe hands both to lmnr-cli, which runs its guarded build.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from typing_extensions import override

from lmnr.evals.errors import EvalConfigError
from lmnr.evals.paths import CallSite, caller_site

# Copied in before the first uv step when the base doesn't have uv.
UV_IMAGE = "ghcr.io/astral-sh/uv:0.13.10"
# Installed with npm for `pnpm_install` when package.json has no
# `"packageManager": "pnpm@<version>"`.
PNPM_VERSION = "12.23.0"
# Matches the `[environment] workdir` new tasks get; `.workdir()` replaces it.
DEFAULT_WORKDIR = "/case"
# The stage name given to a user Dockerfile's last stage when it has none.
APP_STAGE = "lmnr-app"

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PNPM_VERSION = re.compile(r"^[0-9][0-9A-Za-z.+-]*$")
# Lines Docker reads before anything else; they must stay at the top.
_PARSER_DIRECTIVE = re.compile(r"^#\s*(syntax|escape|check)\s*=", re.IGNORECASE)
_FROM = re.compile(
    r"^(?P<head>[ \t]*FROM[ \t]+(?:--\S+[ \t]+)*(?P<ref>\S+))"
    + r"(?:[ \t]+AS[ \t]+(?P<alias>\S+))?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_USER = re.compile(r"^[ \t]*USER[ \t]+(?P<user>\S+)[ \t]*$", re.IGNORECASE | re.MULTILINE)

PathMap = Callable[[Path], str]


@dataclass(frozen=True)
class FromDockerfile:
    dockerfile: str
    context: str | None = None
    # The directory of the file that declared it: relative paths resolve there.
    base_dir: str | None = None
    site: CallSite | None = None


@dataclass(frozen=True)
class FromRegistry:
    ref: str
    site: CallSite | None = None
    # The image has `uv` on PATH (astral's images), so uv steps don't add it.
    has_uv: bool = False


Base = FromDockerfile | FromRegistry


@dataclass(frozen=True)
class Step:
    """One builder step; `method` is the `Image` method that added it."""

    method: str
    args: tuple[str, ...] = ()
    # Absolute host paths the step reads (resolved when it was declared).
    paths: tuple[str, ...] = ()
    dst: str | None = None
    env: tuple[tuple[str, str], ...] = ()
    frozen: bool = True
    site: CallSite | None = None

    def summary(self, root: Path | None = None) -> str:
        """`apt_install poppler-utils`, for `lmnr-cli eval describe`."""
        parts = [*self.args, *(_label(Path(p), root) for p in self.paths)]
        if self.dst:
            parts.append(f"-> {self.dst}")
        parts.extend(f"{key}={value}" for key, value in self.env)
        return " ".join([self.method, *parts])


class ImageContributor(Protocol):
    """Something that adds stages and final-image lines to every image of a
    runner: services and the Laminar layer (`lmnr.evals.layer`). Lines that
    read host files get them through `rel` (context-relative paths), which is
    known only once every contributor's `host_paths` is."""

    def image_title(self, root: Path | None) -> str: ...

    def image_step(self) -> str:
        """What `lmnr-cli eval describe` lists it as (`service`)."""
        ...

    def host_paths(self) -> list[Path]: ...

    def bundle_stages(self, rel: PathMap) -> list[str]: ...

    def install_steps(self, rel: PathMap) -> list[str]: ...


class Image:
    """A container image, built by `lmnr-cli` before the run.

    Start from a base, then add steps; each call returns a new `Image`:

        image = (
            Image.python("3.13")
            .apt_install("poppler-utils")
            .uv_sync("../../../pyproject.toml")
            .add_dir("../../../app", "/srv/app/app")
        )
        image = Image.from_dockerfile("Dockerfile", context="../../..")
        image = Image.from_registry("python:3.13-slim")

    Paths are relative to the file that declares the step (the dataset's
    `eval.py`, or a shared module in `evals/_shared/`), so a shared runner's
    image builds the same from every dataset. Variants share layers:
    `{"default": base, "ocr": base.apt_install("tesseract-ocr")}`.
    """

    def __init__(self, base: Base, steps: Sequence[Step] = ()):
        self.base: Base = base
        self.steps: tuple[Step, ...] = tuple(steps)

    # Bases

    @classmethod
    def from_dockerfile(cls, path: str, context: str | None = None) -> Image:
        """Build from a Dockerfile. `context` defaults to the Dockerfile's directory.

        Steps added after it (and services) go on top of its last stage."""
        site = caller_site()
        return cls(
            FromDockerfile(
                dockerfile=path,
                context=context,
                base_dir=str(site.file.parent) if site is not None else None,
                site=site,
            )
        )

    @classmethod
    def from_registry(cls, ref: str) -> Image:
        """Start from an image in a registry, e.g. `"python:3.13-slim"`."""
        return cls(FromRegistry(ref=ref, site=caller_site()))

    @classmethod
    def debian_slim(cls, version: str = "trixie") -> Image:
        """`debian:<version>-slim`. No Python: add it with `apt_install`."""
        return cls(FromRegistry(ref=f"debian:{version}-slim", site=caller_site()))

    @classmethod
    def python(cls, version: str) -> Image:
        """`python:<version>-slim-trixie`."""
        return cls(FromRegistry(ref=f"python:{version}-slim-trixie", site=caller_site()))

    @classmethod
    def uv_python(cls, version: str) -> Image:
        """Astral's `ghcr.io/astral-sh/uv:python<version>-trixie-slim`: Python
        with `uv` installed."""
        return cls(
            FromRegistry(
                ref=f"ghcr.io/astral-sh/uv:python{version}-trixie-slim",
                site=caller_site(),
                has_uv=True,
            )
        )

    @classmethod
    def node(cls, version: str) -> Image:
        """`node:<version>-trixie-slim`."""
        return cls(FromRegistry(ref=f"node:{version}-trixie-slim", site=caller_site()))

    # Packages

    def apt_install(self, *packages: str) -> Image:
        """`apt-get install` Debian / Ubuntu packages."""
        return self._add(Step("apt_install", args=_strings("apt_install", packages)))

    def pip_install(self, *packages: str) -> Image:
        """`pip install` into the image's `python3`."""
        return self._add(Step("pip_install", args=_strings("pip_install", packages)))

    def uv_pip_install(self, *packages: str) -> Image:
        """`uv pip install --system` into the image's Python (adds uv if needed)."""
        return self._add(Step("uv_pip_install", args=_strings("uv_pip_install", packages)))

    def pip_install_from_requirements(self, path: str) -> Image:
        """`pip install -r <path>`. The file is mounted for the build, not copied."""
        site = caller_site()
        return self._add(
            Step("pip_install_from_requirements", paths=(_host(path, site),)), site=site
        )

    def uv_sync(self, pyproject_path: str, frozen: bool = True) -> Image:
        """Install a uv project's locked dependencies (not the project itself)
        into the image's Python: `uv sync --no-install-workspace`, with
        `pyproject.toml` and `uv.lock` mounted for the build. Add the code with
        `add_dir` after it, so code changes keep this layer cached."""
        site = caller_site()
        return self._add(
            Step("uv_sync", paths=(_host(pyproject_path, site),), frozen=frozen), site=site
        )

    def npm_ci(self, dir: str, dst: str) -> Image:
        """Copy `package.json` and the lockfile from `dir` to `dst`, then `npm ci`
        there. Add the code to `dst` with `add_dir` after it."""
        site = caller_site()
        return self._add(Step("npm_ci", paths=(_host(dir, site),), dst=_abs(dst)), site=site)

    def pnpm_install(self, dir: str, dst: str, version: str | None = None) -> Image:
        """Copy `package.json` and `pnpm-lock.yaml` (and `pnpm-workspace.yaml`,
        which holds settings such as `onlyBuiltDependencies`) from `dir` to
        `dst`, then `pnpm install --frozen-lockfile` there, with `dir`'s
        `.npmrc` mounted for the build (never left in the image). pnpm is
        installed with npm, so the base needs Node (`Image.node`), at
        `version`, else package.json's `packageManager`, else `PNPM_VERSION`.
        Add the code to `dst` with `add_dir` after it."""
        site = caller_site()
        args = () if version is None else _strings("pnpm_install", (version,))
        return self._add(
            Step("pnpm_install", args=args, paths=(_host(dir, site),), dst=_abs(dst)), site=site
        )

    # Files

    def add_file(self, src: str, dst: str) -> Image:
        site = caller_site()
        return self._add(Step("add_file", paths=(_host(src, site),), dst=_abs(dst)), site=site)

    def add_dir(self, src: str, dst: str) -> Image:
        """Copy a directory's contents to `dst`."""
        site = caller_site()
        return self._add(Step("add_dir", paths=(_host(src, site),), dst=_abs(dst)), site=site)

    # Dockerfile directives

    def run_commands(self, *commands: str) -> Image:
        """One `RUN` per command."""
        return self._add(Step("run_commands", args=_strings("run_commands", commands)))

    def env(self, **variables: str) -> Image:
        for key, value in variables.items():
            if not _ENV_NAME.match(key) or not isinstance(value, str):
                raise EvalConfigError(f"Image.env: {key}={value!r} must be a name and a string.")
        return self._add(Step("env", env=tuple(variables.items())))

    def workdir(self, path: str) -> Image:
        """The image's working directory (default for builder images: `/case`)."""
        return self._add(Step("workdir", dst=_abs(path)))

    # Internals

    def _add(self, step: Step, site: CallSite | None = None) -> Image:
        return Image(self.base, [*self.steps, replace(step, site=site or caller_site())])

    @override
    def __repr__(self) -> str:
        return f"Image({self.base!r}, steps={list(self.steps)!r})"


@dataclass
class CompiledImage:
    """An image as lmnr-cli gets it in describe's `images`: always a generated
    Dockerfile, since describe adds the Laminar layer to every image."""

    dockerfile_text: str
    context: Path
    # The ignore file of a `from_dockerfile` image (`<Dockerfile>.dockerignore`),
    # which a generated Dockerfile would otherwise lose.
    ignore_file: Path | None = None
    # Each base, service and step with where it was declared.
    steps: list[dict[str, str]] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": "generated",
            "dockerfile_text": self.dockerfile_text,
            "context": str(self.context),
        }
        if self.ignore_file is not None:
            out["ignore_file"] = str(self.ignore_file)
        out["steps"] = self.steps
        return out


def compile_image(
    image: Image,
    contributors: Sequence[ImageContributor] = (),
    *,
    root: Path | None = None,
    fallback_dir: Path,
) -> CompiledImage:
    """`image` plus what `contributors` add, as a generated Dockerfile for
    lmnr-cli. Order, for caching: service stages, the base, service installs, the steps in
    declaration order, then `WORKDIR`. A `from_dockerfile` base keeps its own
    stages first; services and steps go on top of its last stage.
    `root` (the evals root) shortens the provenance comments; `fallback_dir`
    resolves paths declared where the caller is unknown.
    """
    base = image.base
    origin = _origin(base.site, root)
    steps_info = [{"origin": origin, "step": _base_summary(base)}]
    steps_info += [{"origin": c.image_title(root), "step": c.image_step()} for c in contributors]
    steps_info += [{"origin": _origin(s.site, root), "step": s.summary(root)} for s in image.steps]

    if isinstance(base, FromDockerfile):
        base_dir = Path(base.base_dir) if base.base_dir else fallback_dir
        dockerfile = (base_dir / base.dockerfile).resolve()
        context = (base_dir / base.context).resolve() if base.context else dockerfile.parent
        if not dockerfile.is_file():
            raise EvalConfigError(f"{dockerfile} does not exist.")
        host_paths = _host_paths(image, contributors)
        for path in host_paths:
            if not path.is_relative_to(context):
                raise EvalConfigError(
                    f"{path} is outside the build context {context} of {dockerfile}; "
                    + "widen `context=` in Image.from_dockerfile(...) to a folder that holds it."
                )
        ignore = dockerfile.with_name(f"{dockerfile.name}.dockerignore")
        text = _compose(
            dockerfile.read_text(),
            header=f"# from {_label(dockerfile, root)}"
            + (f" ({base.site.owner})" if base.site and base.site.owner else ""),
            image=image,
            contributors=contributors,
            rel=_relative_to(context),
            root=root,
        )
        return CompiledImage(
            dockerfile_text=text,
            context=context,
            ignore_file=ignore if ignore.is_file() else None,
            steps=steps_info,
        )

    # The deepest folder holding every added path; without any, the folder of
    # the file that declared the base (a shared module's, for every dataset).
    host_paths = _host_paths(image, contributors)
    if host_paths:
        context = _common_dir([p if p.is_dir() else p.parent for p in host_paths])
    else:
        context = base.site.file.parent if base.site is not None else fallback_dir
    text = _generate(base, image, contributors, rel=_relative_to(context), root=root)
    return CompiledImage(dockerfile_text=text, context=context, steps=steps_info)


# Dockerfile text


def _generate(
    base: FromRegistry,
    image: Image,
    contributors: Sequence[ImageContributor],
    rel: PathMap,
    root: Path | None,
) -> str:
    lines: list[str] = ["# Generated by lmnr.evals. Each step names the code that declared it."]
    for contributor in contributors:
        lines += _stages(contributor, rel, root)
    lines += ["", f"# {_origin(base.site, root)}", f"FROM {base.ref}"]
    for contributor in contributors:
        lines += _installs(contributor, rel, root)
    lines += _steps(image.steps, rel, root, has_uv=base.has_uv)
    if not any(step.method == "workdir" for step in image.steps):
        lines += ["", f"WORKDIR {DEFAULT_WORKDIR}"]
    return "\n".join(lines) + "\n"


def _compose(
    text: str,
    *,
    header: str,
    image: Image,
    contributors: Sequence[ImageContributor],
    rel: PathMap,
    root: Path | None,
) -> str:
    """The user's Dockerfile, its last stage named, then services and steps on
    top of that stage. The one place where builder images and Dockerfiles meet,
    so the Laminar layer (a contributor) works on both."""
    source = text.splitlines()
    directives: list[str] = []
    while source and _PARSER_DIRECTIVE.match(source[0]):
        directives.append(source.pop(0))
    body = "\n".join(source)
    froms = list(_FROM.finditer(body))
    if not froms:
        raise EvalConfigError(f"No FROM line in the Dockerfile ({header.removeprefix('# from ')}).")
    last = froms[-1]
    user = _stage_user(body, froms, len(froms) - 1)
    alias = last.group("alias")
    if alias is None:
        alias = APP_STAGE
        body = body[: last.end("head")] + f" AS {alias}" + body[last.end("head") :]

    lines = [*directives, header, body.rstrip("\n"), ""]
    lines.append("# Added by lmnr.evals. Each step names the code that declared it.")
    for contributor in contributors:
        lines += _stages(contributor, rel, root)
    lines += ["", f"FROM {alias}"]
    if user is not None:
        lines.append("USER root")
    for contributor in contributors:
        lines += _installs(contributor, rel, root)
    lines += _steps(image.steps, rel, root, has_uv=False)
    if user is not None:
        lines += ["", f"USER {user}"]
    return "\n".join(lines) + "\n"


def _stage_user(body: str, froms: list[re.Match[str]], index: int) -> str | None:
    """The `USER` stage `index` ends with: its own last `USER` line, else the
    user of the earlier stage it builds on (`FROM base`). None for root or an
    external base's default user, which the Dockerfile doesn't show."""
    stage = froms[index]
    end = froms[index + 1].start() if index + 1 < len(froms) else len(body)
    users = list(_USER.finditer(body, stage.end(), end))
    if users:
        return users[-1].group("user")
    ref = stage.group("ref").lower()
    for earlier in range(index - 1, -1, -1):
        alias = froms[earlier].group("alias")
        if alias is not None and alias.lower() == ref:
            return _stage_user(body, froms, earlier)
    return None


def _stages(contributor: ImageContributor, rel: PathMap, root: Path | None) -> list[str]:
    stages = contributor.bundle_stages(rel)
    if not stages:
        return []
    return ["", f"# {contributor.image_title(root)}", *stages]


def _installs(contributor: ImageContributor, rel: PathMap, root: Path | None) -> list[str]:
    steps = contributor.install_steps(rel)
    if not steps:
        return []
    return ["", f"# {contributor.image_title(root)}", *steps]


def _steps(steps: Sequence[Step], rel: PathMap, root: Path | None, *, has_uv: bool) -> list[str]:
    lines: list[str] = []
    for step in steps:
        lines += ["", f"# {_origin(step.site, root)}"]
        if step.method in ("uv_pip_install", "uv_sync") and not has_uv:
            lines.append(f"COPY --from={UV_IMAGE} /uv /uvx /usr/local/bin/")
            has_uv = True
        lines += _render(step, rel)
    return lines


def _render(step: Step, rel: PathMap) -> list[str]:
    args = " ".join(shlex.quote(a) for a in step.args)
    paths = [Path(p) for p in step.paths]
    match step.method:
        case "apt_install":
            return [
                "RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                + f"--no-install-recommends {args} && rm -rf /var/lib/apt/lists/*"
            ]
        case "pip_install":
            return [f"RUN PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --no-cache-dir {args}"]
        case "uv_pip_install":
            return [f"RUN uv pip install --system --break-system-packages --no-cache {args}"]
        case "pip_install_from_requirements":
            requirements = _existing_file(paths[0], step)
            return [
                f"RUN {_mount(rel(requirements), '/tmp/requirements.txt')} "
                + "PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --no-cache-dir "
                + "-r /tmp/requirements.txt"
            ]
        case "uv_sync":
            pyproject = _existing_file(paths[0], step)
            lock = pyproject.with_name("uv.lock")
            mounts = [_mount(rel(pyproject), "/tmp/uv-sync/pyproject.toml", rw=True)]
            if lock.is_file():
                mounts.append(_mount(rel(lock), "/tmp/uv-sync/uv.lock", rw=True))
            elif step.frozen:
                raise EvalConfigError(
                    f"uv_sync({str(pyproject)!r}) has no uv.lock next to it; run `uv lock`, "
                    + "or pass frozen=False."
                )
            flags = "--frozen " if step.frozen else ""
            return [
                f"RUN {' '.join(mounts)} cd /tmp/uv-sync && "
                + "UV_PROJECT_ENVIRONMENT=\"$(python3 -c 'import sys; print(sys.prefix)')\" "
                + f"UV_PYTHON_DOWNLOADS=never uv sync {flags}--no-install-workspace --no-cache"
            ]
        case "npm_ci":
            directory = paths[0]
            if not (directory / "package.json").is_file():
                raise EvalConfigError(f"npm_ci: {directory}/package.json does not exist.")
            lockfiles = [
                directory / name
                for name in ("package-lock.json", "npm-shrinkwrap.json")
                if (directory / name).is_file()
            ]
            if not lockfiles:
                raise EvalConfigError(
                    f"npm_ci: {directory} has no package-lock.json; run `npm install` there."
                )
            dst = _dir(step.dst)
            sources = [rel(directory / "package.json"), rel(lockfiles[0])]
            return [_copy(sources, dst), f"RUN cd {shlex.quote(dst)} && npm ci"]
        case "pnpm_install":
            return _render_pnpm(step, paths[0], rel)
        case "add_file":
            return [_copy([rel(_existing_file(paths[0], step))], _str(step.dst))]
        case "add_dir":
            directory = paths[0]
            if not directory.is_dir():
                raise EvalConfigError(f"add_dir: {directory} is not a directory.")
            return [_copy([_dir(rel(directory))], _dir(step.dst))]
        case "run_commands":
            return [f"RUN {command}" for command in step.args]
        case "env":
            return ["ENV " + " ".join(f"{key}={json.dumps(value)}" for key, value in step.env)]
        case "workdir":
            return [f"WORKDIR {step.dst}"]
        case _:
            raise EvalConfigError(f"Unknown image step {step.method!r}.")


def _render_pnpm(step: Step, directory: Path, rel: PathMap) -> list[str]:
    manifest = directory / "package.json"
    if not manifest.is_file():
        raise EvalConfigError(f"pnpm_install: {manifest} does not exist.")
    lock = directory / "pnpm-lock.yaml"
    if not lock.is_file():
        raise EvalConfigError(
            f"pnpm_install: {directory} has no pnpm-lock.yaml; run `pnpm install`."
        )
    sources = [rel(manifest), rel(lock)]
    workspace = directory / "pnpm-workspace.yaml"
    if workspace.is_file():
        if re.search(r"^packages\s*:", workspace.read_text(), re.MULTILINE):
            raise EvalConfigError(
                f"pnpm_install: {workspace} lists workspace packages, which need every "
                + "package's package.json before the install; use Image.from_dockerfile "
                + "with `pnpm fetch` for a workspace."
            )
        sources.append(rel(workspace))
    version = step.args[0] if step.args else _pnpm_version(manifest)
    if not _PNPM_VERSION.match(version):
        raise EvalConfigError(f"pnpm_install: {version!r} is not a pnpm version.")
    dst = _dir(step.dst)
    npmrc = directory / ".npmrc"
    mount = f"{_mount(rel(npmrc), f'{dst}.npmrc')} " if npmrc.is_file() else ""
    return [
        # Its own layer: kept while the lockfile changes.
        "RUN npm install -g --no-fund --no-audit --no-update-notifier "
        + f"pnpm@{version} && rm -rf /root/.npm /tmp/node-compile-cache",
        _copy(sources, dst),
        # The store is on the same filesystem, so node_modules hard-links it and
        # deleting it in the same layer keeps the image to one copy.
        f"RUN {mount}cd {shlex.quote(dst)} && pnpm install --frozen-lockfile "
        + "--store-dir /tmp/pnpm-store && rm -rf /tmp/pnpm-store* /tmp/node-compile-cache",
    ]


def _pnpm_version(manifest: Path) -> str:
    """package.json's `"packageManager": "pnpm@10.12.1+sha512..."` version, or
    `PNPM_VERSION`."""
    try:
        package_manager = json.loads(manifest.read_text()).get("packageManager")
    except (ValueError, AttributeError):
        raise EvalConfigError(f"pnpm_install: {manifest} is not a JSON object.") from None
    if isinstance(package_manager, str) and package_manager.startswith("pnpm@"):
        return package_manager.removeprefix("pnpm@").split("+", 1)[0]
    return PNPM_VERSION


def _copy(sources: list[str], dst: str) -> str:
    return "COPY " + json.dumps([*sources, dst])


def _mount(source: str, target: str, rw: bool = False) -> str:
    """A BuildKit bind mount of a context file, CSV-quoted when needed."""
    fields = ["type=bind", f"source={source}", f"target={target}"]
    if rw:
        fields.append("rw")
    quoted = [f'"{f}"' if re.search(r'[\s,"]', f) else f for f in fields]
    return "--mount=" + ",".join(quoted)


# Paths


def _host(path: str, site: CallSite | None) -> str:
    """`path` on the host, relative to the declaring file's directory."""
    if not isinstance(path, str) or not path:
        raise EvalConfigError(f"Image step paths must be non-empty strings, got {path!r}.")
    base = site.file.parent if site is not None else Path.cwd()
    return str((base / path).resolve())


def _abs(path: str) -> str:
    if not isinstance(path, str) or not PurePosixPath(path).is_absolute():
        raise EvalConfigError(f"Container paths in Image steps must be absolute, got {path!r}.")
    return path


def _dir(path: str | None) -> str:
    path = _str(path)
    return path if path.endswith("/") else path + "/"


def _str(value: str | None) -> str:
    assert value is not None
    return value


def _existing_file(path: Path, step: Step) -> Path:
    if not path.is_file():
        raise EvalConfigError(f"{step.method}: {path} does not exist.")
    return path


def _relative_to(context: Path) -> PathMap:
    def rel(path: Path) -> str:
        path = path.resolve()
        if not path.is_relative_to(context):
            raise EvalConfigError(f"{path} is outside the build context {context}.")
        relative = path.relative_to(context).as_posix()
        return relative if relative != "." else "."

    return rel


def _host_paths(image: Image, contributors: Sequence[ImageContributor]) -> list[Path]:
    paths = [Path(p) for step in image.steps for p in step.paths]
    for contributor in contributors:
        paths += contributor.host_paths()
    return paths


def _common_dir(paths: Sequence[Path]) -> Path:
    resolved = [str(p.resolve()) for p in paths]
    return Path(os.path.commonpath(resolved))


def _strings(method: str, values: Sequence[str]) -> tuple[str, ...]:
    if not values or not all(isinstance(v, str) and v for v in values):
        raise EvalConfigError(f"Image.{method}(...) takes one or more non-empty strings.")
    return tuple(values)


# Provenance


def _label(path: Path, root: Path | None) -> str:
    """`path` relative to the evals root (with `..` when outside it)."""
    if root is None:
        return path.as_posix()
    return Path(os.path.relpath(path, root)).as_posix()


def _origin(site: CallSite | None, root: Path | None) -> str:
    return site.label(root) if site is not None else "(unknown)"


def _base_summary(base: Base) -> str:
    if isinstance(base, FromRegistry):
        return f"from {base.ref}"
    return f"from_dockerfile {base.dockerfile}"

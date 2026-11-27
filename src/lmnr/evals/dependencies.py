"""What `EvalRunner.mock_dependencies` installs in the Laminar layer.

The mock server imports `eval.py` in the container, so whatever `eval.py` and
its shared modules import at the top level must be in the layer's venv. By
default that is the evals project's dependencies, the same set lmnr-cli
installs next to Harbor: the evals root's `pyproject.toml` `[project]
dependencies`, then `eval.py`'s PEP 723 `# /// script` `dependencies`.

Resolved on the host by describe: in the container there is no pyproject, and
importing `eval.py` there must not read one.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, cast

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from lmnr.evals.errors import EvalConfigError
from lmnr.evals.layer import validate_mock_dependencies
from lmnr.evals.mocks.mock import PROJECT_DEPENDENCIES, ProjectDependencies
from lmnr.evals.paths import evals_root

PYPROJECT = "pyproject.toml"
# PEP 723's reference regex (also lmnr-cli's `parseScriptDependencies`),
# tolerating trailing whitespace on the delimiter lines.
PEP_723_BLOCK = re.compile(
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)[ \t]*$\s(?P<content>(^#(| .*)$\s)+)^# ///[ \t]*$"
)


def check_mock_dependencies(owner: str, declared: object) -> None:
    """The attribute's shape: None, or a list of pip requirements and
    `mock.PROJECT_DEPENDENCIES`."""
    if declared is None:
        return
    entries = cast(list[object], declared) if isinstance(declared, (list, tuple)) else None
    if entries is None or not all(
        isinstance(d, ProjectDependencies) or (isinstance(d, str) and d.strip() and "\n" not in d)
        for d in entries
    ):
        raise EvalConfigError(
            f"{owner}.mock_dependencies must be a list of pip requirements, which may include "
            + 'mock.PROJECT_DEPENDENCIES ([mock.PROJECT_DEPENDENCIES, "httpx>=0.27"]), or None '
            + f"for the evals project's dependencies; got {declared!r}."
        )


def resolve_mock_dependencies(
    owner: str, declared: object, eval_path: Path, has_mocks: bool
) -> tuple[tuple[str, ...], list[str]]:
    """The requirements the layer installs besides `lmnr`, and warnings.

    Nothing without mocks: the layer stays lean. `None` means
    `[mock.PROJECT_DEPENDENCIES]`; the marker expands in place to the project's
    dependencies that the layer can install (a `[tool.uv.sources]` package or
    a local path can't be, and is dropped with a warning). `lmnr` itself is
    dropped: the layer pins it.
    """
    check_mock_dependencies(owner, declared)
    entries = cast("list[str | ProjectDependencies] | None", declared)
    if not has_mocks:
        if entries:
            return (), [
                f"{owner}.mock_dependencies is ignored: the runner has no mocks, so nothing "
                + "imports eval.py in the container."
            ]
        return (), []
    if entries is None:
        entries = [PROJECT_DEPENDENCIES]
    explicit = {_name(e) for e in entries if isinstance(e, str)}
    project: list[str] = []
    warnings: list[str] = []
    if any(isinstance(e, ProjectDependencies) for e in entries):
        for spec, origin, sources in project_dependencies(eval_path):
            name = _name(spec)
            reason = _not_installable(spec, name, sources) if name != "lmnr" else None
            if reason is None:
                project.append(spec)
            elif name is None or name not in explicit:
                warnings.append(
                    f"{owner}.mock_dependencies: {spec!r} from {origin} is not installed in "
                    + f"the Laminar layer ({reason}). If the mock handlers, or what eval.py "
                    + "imports at the top level, need it, add an installable spec to "
                    + f'mock_dependencies: [mock.PROJECT_DEPENDENCIES, "{name or "<package>"}'
                    + '>=..."].'
                )
    resolved = [
        spec
        for entry in entries
        for spec in (project if isinstance(entry, ProjectDependencies) else [entry.strip()])
    ]
    # lmnr's own entries go: the layer pins it.
    unique = list(dict.fromkeys(r for r in resolved if _name(r) != "lmnr"))
    return validate_mock_dependencies(owner, unique), warnings


def project_dependencies(eval_path: Path) -> list[tuple[str, str, set[str]]]:
    """`(requirement, where it's declared, [tool.uv.sources] names there)` for
    the evals root's `pyproject.toml` (only with a `[project]` table, as
    lmnr-cli) and `eval.py`'s PEP 723 block, in that order."""
    found: list[tuple[str, str, set[str]]] = []
    pyproject = evals_root(eval_path) / PYPROJECT
    if pyproject.is_file():
        data = _toml(pyproject.read_text(), str(pyproject))
        project = data.get("project")
        if isinstance(project, dict):
            project = cast(dict[str, Any], project)
            deps = _strings(project.get("dependencies", []), f"{pyproject}: [project] dependencies")
            sources = _uv_sources(data)
            found += [(dep, PYPROJECT, sources) for dep in deps]
    script = _script_metadata(eval_path)
    if script is not None:
        deps = _strings(
            script.get("dependencies", []),
            f"{eval_path}: `dependencies` in its `# /// script` block",
        )
        sources = _uv_sources(script)
        found += [(dep, f"{eval_path.name}'s # /// script block", sources) for dep in deps]
    return found


def _script_metadata(eval_path: Path) -> dict[str, Any] | None:
    blocks = [
        m for m in PEP_723_BLOCK.finditer(eval_path.read_text()) if m.group("type") == "script"
    ]
    if not blocks:
        return None
    if len(blocks) > 1:
        raise EvalConfigError(f"{eval_path} has more than one `# /// script` block.")
    content = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in blocks[0].group("content").splitlines(keepends=True)
    )
    return _toml(content, f"{eval_path}'s `# /// script` block")


def _toml(text: str, what: str) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise EvalConfigError(f"{what} is not valid TOML: {e}") from e


def _strings(value: object, what: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(v, str) for v in cast(list[object], value)
    ):
        raise EvalConfigError(f"{what} must be a list of requirement strings.")
    return [v.strip() for v in cast(list[str], value)]


def _uv_sources(data: dict[str, Any]) -> set[str]:
    tool = data.get("tool")
    uv = cast(dict[str, Any], tool).get("uv") if isinstance(tool, dict) else None
    sources = cast(dict[str, Any], uv).get("sources") if isinstance(uv, dict) else None
    if not isinstance(sources, dict):
        return set()
    return {canonicalize_name(str(name)) for name in cast(dict[str, Any], sources)}


def _name(spec: str) -> str | None:
    try:
        return canonicalize_name(Requirement(spec).name)
    except InvalidRequirement:
        return None


def _not_installable(spec: str, name: str | None, sources: set[str]) -> str | None:
    if name is None:
        return "not a requirement: a local path?"
    if name in sources:
        return "it comes from [tool.uv.sources]"
    url = Requirement(spec).url
    if url is not None and url.lower().startswith("file:"):
        return "a local path"
    return None

"""Artifacts: files collected from the agent container for grading.

Harbor copies them into the trial's `artifacts/` dir on the host after the
agent phase, in both verifier modes, and in `separate` mode back into the
grading container at their original paths. It always collects
`/logs/artifacts/`, where `ctx.save_artifact` puts files. On the host an entry
lands at its `destination`, else at its container path without the leading
`/` (`/srv/app/out/` -> `artifacts/srv/app/out/`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from lmnr.evals.errors import EvalConfigError

# The container folder Harbor always collects (Harbor's `EnvironmentPaths`).
ARTIFACTS_DIR = "/logs/artifacts"
# Harbor's record of what it collected, at the root of the host dir.
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class Artifact:
    """A container file or folder Harbor collects after the agent phase, like
    Harbor's `ArtifactConfig`: `destination` is where it lands under the
    trial's `artifacts/` dir; `exclude` takes `tar --exclude` patterns."""

    source: str
    destination: str | None = None
    exclude: list[str] = field(default_factory=list)

    def to_harbor(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"source": self.source}
        if self.destination:
            entry["destination"] = self.destination
        if self.exclude:
            entry["exclude"] = list(self.exclude)
        return entry


def describe_artifacts(entries: Sequence[str | Artifact]) -> list[str | dict[str, Any]]:
    """Entries for the job config's `artifacts`."""
    return [entry if isinstance(entry, str) else entry.to_harbor() for entry in entries]


def validate_artifacts(owner: str, entries: Any) -> None:
    """Harbor's own rules, so a bad entry fails in describe, not in a trial."""
    if not isinstance(entries, (list, tuple)):
        raise EvalConfigError(
            f"{owner}.artifacts must be a list of container paths or Artifact(...), "
            + f"got {entries!r}."
        )
    for entry in entries:
        if isinstance(entry, str):
            entry = Artifact(entry)
        if not isinstance(entry, Artifact):
            raise EvalConfigError(
                f"{owner}.artifacts entries must be container paths or Artifact(...), "
                + f"got {entry!r}."
            )
        if not isinstance(entry.source, str) or not entry.source:
            raise EvalConfigError(f"{owner}.artifacts: an entry has no source path.")
        if ".." in PurePosixPath(entry.source).parts:
            raise EvalConfigError(f"{owner}.artifacts: {entry.source!r} must not contain '..'.")
        if entry.destination is not None:
            check_destination(f"{owner}.artifacts: destination", entry.destination)
            if entry.destination.rstrip("/") == MANIFEST_FILE:
                raise EvalConfigError(
                    f"{owner}.artifacts: destination {MANIFEST_FILE!r} is reserved by Harbor."
                )
        if not isinstance(entry.exclude, (list, tuple)) or not all(
            isinstance(p, str) for p in entry.exclude
        ):
            raise EvalConfigError(
                f"{owner}.artifacts: exclude of {entry.source!r} must be a list of patterns."
            )


def check_destination(what: str, path: Any) -> None:
    """A path relative to an artifacts folder, which can't leave it."""
    if not isinstance(path, str) or not path.strip("/"):
        raise EvalConfigError(f"{what} must name a file or folder, got {path!r}.")
    pure = PurePosixPath(path)
    if "\\" in path or pure.is_absolute() or ".." in pure.parts:
        raise EvalConfigError(f"{what} must be a relative path with '/' and no '..', got {path!r}.")


class Artifacts:
    """`ctx.artifacts` in `verify`: the trial's collected artifacts on the host.

    A path is either the container path an artifact came from
    (`"/srv/app/out/report.json"`, `"/logs/artifacts/state.db"`), or a path
    relative to the trial's `artifacts/` dir, as `list()` returns them (where
    an entry's `destination` lands).
    """

    def __init__(self, host_dir: Path, declared: Sequence[str | Artifact] = ()):
        self.dir: Path = host_dir
        self._declared: list[Artifact] = [
            Artifact(entry) if isinstance(entry, str) else entry for entry in declared
        ]

    def path(self, path: str) -> Path:
        """The host path of an artifact file or folder (it may not exist)."""
        pure = PurePosixPath(path)
        if ".." in pure.parts:
            raise ValueError(f"Artifact path must not contain '..': {path!r}")
        if not pure.is_absolute():
            return self.dir.joinpath(*pure.parts)
        # A declared entry with a destination: collected under that name.
        for entry in self._declared:
            source = PurePosixPath(entry.source)
            if entry.destination and (pure == source or source in pure.parents):
                destination = PurePosixPath(entry.destination)
                return self.dir.joinpath(*destination.parts, *pure.relative_to(source).parts)
        return self.dir.joinpath(*pure.parts[1:])

    def exists(self, path: str) -> bool:
        return self.path(path).exists()

    def read_text(self, path: str) -> str:
        return self.path(path).read_text()

    def read_json(self, path: str) -> Any:
        return json.loads(self.read_text(path))

    def list(self) -> list[str]:
        """Every collected file, relative to the trial's `artifacts/` dir."""
        if not self.dir.is_dir():
            return []
        return sorted(
            path.relative_to(self.dir).as_posix()
            for path in self.dir.rglob("*")
            if path.is_file() and path != self.dir / MANIFEST_FILE
        )

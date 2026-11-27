"""The image a dataset's trials start from. lmnr-cli builds it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lmnr.evals.paths import caller_dir


@dataclass(frozen=True)
class FromDockerfile:
    dockerfile: str
    context: str | None = None
    # The directory of the file that declared it: relative paths resolve there.
    base_dir: str | None = None


@dataclass(frozen=True)
class FromRegistry:
    ref: str


ImageStep = FromDockerfile | FromRegistry


class Image:
    """A container image, built by `lmnr-cli` before the run.

    Start from a Dockerfile or a registry image:

        image = Image.from_dockerfile("Dockerfile", context="../../..")
        image = Image.from_registry("python:3.13-slim")

    Paths are relative to the file that calls `from_dockerfile` (the dataset's
    `eval.py`, or a shared module in `evals/_shared/`), so a shared runner's
    image builds the same from every dataset.
    """

    def __init__(self, steps: list[ImageStep]):
        # A list so later steps (packages, files, ...) can extend a base image.
        self.steps: list[ImageStep] = steps

    @classmethod
    def from_dockerfile(cls, path: str, context: str | None = None) -> Image:
        """Build from a Dockerfile. `context` defaults to the Dockerfile's directory."""
        base_dir = caller_dir()
        return cls(
            [
                FromDockerfile(
                    dockerfile=path,
                    context=context,
                    base_dir=str(base_dir) if base_dir is not None else None,
                )
            ]
        )

    @classmethod
    def from_registry(cls, ref: str) -> Image:
        """Use an image from a registry as it is, e.g. `"python:3.13-slim"`."""
        return cls([FromRegistry(ref=ref)])

    def describe(self, base_dir: Path) -> dict[str, Any]:
        """The image for lmnr-cli, with paths resolved against the declaring
        file's directory (`base_dir` if it isn't known)."""
        base = self.steps[0]
        if isinstance(base, FromRegistry):
            return {"kind": "registry", "ref": base.ref}
        if base.base_dir is not None:
            base_dir = Path(base.base_dir)
        dockerfile = (base_dir / base.dockerfile).resolve()
        context = (base_dir / base.context).resolve() if base.context else dockerfile.parent
        return {"kind": "dockerfile", "dockerfile": str(dockerfile), "context": str(context)}

    def __repr__(self) -> str:
        return f"Image({self.steps!r})"

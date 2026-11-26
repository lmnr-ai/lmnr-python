"""The image a dataset's trials start from. lmnr-cli builds it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FromDockerfile:
    dockerfile: str
    context: str | None = None


@dataclass(frozen=True)
class FromRegistry:
    ref: str


ImageStep = FromDockerfile | FromRegistry


class Image:
    """A container image, built by `lmnr-cli` before the run.

    Start from a Dockerfile or a registry image:

        image = Image.from_dockerfile("Dockerfile", context="../../..")
        image = Image.from_registry("python:3.13-slim")

    Paths are relative to the directory of the `eval.py` that defines the image.
    """

    def __init__(self, steps: list[ImageStep]):
        # A list so later steps (packages, files, ...) can extend a base image.
        self.steps: list[ImageStep] = steps

    @classmethod
    def from_dockerfile(cls, path: str, context: str | None = None) -> Image:
        """Build from a Dockerfile. `context` defaults to the Dockerfile's directory."""
        return cls([FromDockerfile(dockerfile=path, context=context)])

    @classmethod
    def from_registry(cls, ref: str) -> Image:
        """Use an image from a registry as it is, e.g. `"python:3.13-slim"`."""
        return cls([FromRegistry(ref=ref)])

    def describe(self, base_dir: Path) -> dict[str, Any]:
        """The image for lmnr-cli, with paths resolved against `base_dir`."""
        base = self.steps[0]
        if isinstance(base, FromRegistry):
            return {"kind": "registry", "ref": base.ref}
        dockerfile = (base_dir / base.dockerfile).resolve()
        context = (base_dir / base.context).resolve() if base.context else dockerfile.parent
        return {"kind": "dockerfile", "dockerfile": str(dockerfile), "context": str(context)}

    def __repr__(self) -> str:
        return f"Image({self.steps!r})"

"""`python -m lmnr.evals.describe agent/eval.py`: an `eval.py` as JSON for lmnr-cli.

Prints one JSON object to stdout (warnings also go to stderr). Exits 2 on a
configuration error and 1 if the file can't be imported, with the error on
stderr.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from lmnr.evals.artifacts import describe_artifacts
from lmnr.evals.errors import EvalConfigError
from lmnr.evals.image import compile_image
from lmnr.evals.layer import LAYER_PYTHON, LaminarLayer, validate_mock_dependencies
from lmnr.evals.paths import evals_root, loaded_modules
from lmnr.evals.runner import EvalRunner, Secret, default_image_name, find_runner, load_module

# Bumped when the shape changes incompatibly; later fields are added as optional.
DESCRIBE_VERSION = 1
# `--layer-lmnr=wheel` or `--layer-lmnr=<requirement>`: what the Laminar layer
# installs instead of this lmnr's own version (lmnr-cli's LMNR_EVAL_LMNR_SPEC).
LAYER_FLAG = "--layer-lmnr="
WHEEL = "wheel"


def describe(path: Path | str, layer_lmnr: str | None = None) -> dict[str, Any]:
    """`layer_lmnr` is what the Laminar layer installs: a requirement
    (`lmnr @ git+https://...`), `"wheel"` for the wheel lmnr-cli builds from a
    local checkout (the `lmnr-dev` build context), or None for the version
    running this."""
    path = Path(path).resolve()
    # Output of the user's module must not end up in the JSON.
    with contextlib.redirect_stdout(sys.stderr):
        runner = find_runner(load_module(path))
        warnings = runner.validate()

    root = evals_root(path)
    if layer_lmnr is None:
        layer_lmnr = f"lmnr=={importlib.metadata.version('lmnr')}"
    layer = LaminarLayer(
        lmnr=None if layer_lmnr == WHEEL else layer_lmnr,
        mock_dependencies=validate_mock_dependencies(runner.__name__, runner.mock_dependencies),
    )
    # The layer goes into every image: the runner's helpers run on it.
    contributors = [*runner.services, layer]
    images = {
        name or default_image_name(path): compile_image(
            image, contributors, root=root, fallback_dir=path.parent
        ).describe()
        for name, image in runner.images().items()
    }
    return {
        "version": DESCRIBE_VERSION,
        "runner": runner.__name__,
        "images": images,
        "default_image": next(iter(images)),
        "model": runner.model,
        "env": {
            key: f"${{{value.name}}}" if isinstance(value, Secret) else value
            for key, value in runner.env.items()
        },
        "secrets": list(runner.secrets),
        "new_task_toml": runner.new_task_toml,
        "grade_with_test_sh": runner.grade_with_test_sh,
        "warnings": warnings,
        # The runner's code: module files under the evals root, relative to it.
        "modules": loaded_modules(path),
        # The job config's `artifacts` (Harbor entries).
        "artifacts": describe_artifacts(runner.artifacts),
        # For `lmnr-cli eval describe`: what the dataset runs on, and where
        # each part was declared.
        "services": [service.describe(root) for service in runner.services],
        "classes": _class_chain(runner, root),
        "mocks": [spec.describe() for spec in runner.mock_specs()],
        "mock_dependencies": list(layer.mock_dependencies),
        # In every image; `build_contexts` are named contexts lmnr-cli passes
        # to each build (`--build-context lmnr-dev=<dir with the wheel>`).
        "laminar_layer": {
            "python": LAYER_PYTHON,
            "lmnr": layer.lmnr,
            "build_contexts": layer.build_contexts(),
        },
    }


def _class_chain(runner: type[EvalRunner], root: Path) -> list[dict[str, Any]]:
    """The runner and its bases up to `EvalRunner`, with their files."""
    chain: list[dict[str, Any]] = []
    for cls in runner.__mro__:
        if cls is EvalRunner or not issubclass(cls, EvalRunner):
            continue
        file = getattr(sys.modules.get(cls.__module__), "__file__", None)
        shown = None
        if isinstance(file, str):
            resolved = Path(file).resolve()
            shown = (
                resolved.relative_to(root).as_posix()
                if resolved.is_relative_to(root)
                else resolved.as_posix()
            )
        chain.append({"name": cls.__name__, "file": shown})
    return chain


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    layer_lmnr: str | None = None
    for arg in list(args):
        if arg.startswith(LAYER_FLAG):
            layer_lmnr = arg.removeprefix(LAYER_FLAG)
            args.remove(arg)
    if len(args) != 1 or layer_lmnr == "":
        print(
            "usage: python -m lmnr.evals.describe <path/to/eval.py> "
            + f"[{LAYER_FLAG}wheel|<requirement>]",
            file=sys.stderr,
        )
        return 2
    try:
        description = describe(args[0], layer_lmnr)
    except EvalConfigError as e:
        print(f"{args[0]}: {e}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1
    for warning in description["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    print(json.dumps(description, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

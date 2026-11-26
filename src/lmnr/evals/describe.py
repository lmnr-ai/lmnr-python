"""`python -m lmnr.evals.describe agent/eval.py`: an `eval.py` as JSON for lmnr-cli.

Prints one JSON object to stdout (warnings also go to stderr). Exits 2 on a
configuration error and 1 if the file can't be imported, with the error on
stderr.
"""

from __future__ import annotations

import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from lmnr.evals.errors import EvalConfigError
from lmnr.evals.runner import Secret, default_image_name, find_runner, load_module

# Bumped when the shape changes incompatibly; later fields are added as optional.
DESCRIBE_VERSION = 1


def describe(path: Path | str) -> dict[str, Any]:
    path = Path(path).resolve()
    # Output of the user's module must not end up in the JSON.
    with contextlib.redirect_stdout(sys.stderr):
        runner = find_runner(load_module(path))
        warnings = runner.validate()

    base_dir = path.parent
    images = {
        name or default_image_name(path): image.describe(base_dir)
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
        "task_template": runner.task_template,
        "verify": runner.verify_mode(),
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m lmnr.evals.describe <path/to/eval.py>", file=sys.stderr)
        return 2
    try:
        description = describe(args[0])
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

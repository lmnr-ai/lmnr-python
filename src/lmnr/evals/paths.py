"""Where paths in eval code resolve.

A relative path resolves against the file of the code that declares it (an
`Image.from_dockerfile(...)` in a class body, a `ctx.container.run_file(...)`
call), not against the dataset's `eval.py`, so a runner shared from
`evals/_shared/` works from every dataset that subclasses it.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_ASYNCIO_DIR = Path(asyncio.__file__).resolve().parent

# Module files under the evals root that aren't the user's code.
_SKIPPED_DIRS = {"site-packages", "dist-packages", "node_modules", "__pycache__"}


@dataclass(frozen=True)
class CallSite:
    """Where user code declared something: the file, the line, and the class
    whose body ran the call (None at module level)."""

    file: Path
    line: int
    owner: str | None = None

    def label(self, root: Path | None = None, owner: bool = True) -> str:
        """`_shared/support.py:12 (SupportRunner)`, relative to `root` when the
        file is under it."""
        shown = Path(os.path.relpath(self.file, root)) if root is not None else self.file
        suffix = f" ({self.owner})" if owner and self.owner else ""
        return f"{shown.as_posix()}:{self.line}{suffix}"


def caller_site() -> CallSite | None:
    """The first frame up the stack outside `lmnr.evals` (see `caller_dir`)."""
    frame = inspect.currentframe()
    while frame is not None:
        path = Path(frame.f_code.co_filename)
        # Skips "<string>", "<frozen ...>" and other non-files.
        if path.is_file():
            resolved = path.resolve()
            if resolved.is_relative_to(_ASYNCIO_DIR):
                return None
            if not resolved.is_relative_to(_PACKAGE_DIR):
                # A class body's namespace has `__qualname__`; a module's doesn't.
                owner = frame.f_locals.get("__qualname__")
                return CallSite(
                    file=resolved,
                    line=frame.f_lineno,
                    owner=owner if isinstance(owner, str) else None,
                )
        frame = frame.f_back
    return None


def caller_dir() -> Path | None:
    """The directory of the first file up the stack outside `lmnr.evals`.

    None when there is none, e.g. a coroutine started as its own task
    (`asyncio.gather(ctx.container.run_file(...))`), whose caller is the event
    loop.
    """
    site = caller_site()
    return site.file.parent if site is not None else None


def evals_root(eval_path: Path | str) -> Path:
    """The evals folder of `<evals root>/<dataset>/agent/eval.py`."""
    return Path(eval_path).resolve().parent.parent.parent


def add_evals_root_to_path(eval_path: Path | str) -> None:
    """Make shared modules (`evals/_shared/`) importable from `eval.py`.

    Appended, so a folder in the evals root never shadows an installed package.
    """
    root = str(evals_root(eval_path))
    if root not in sys.path:
        sys.path.append(root)


def loaded_modules(eval_path: Path | str) -> list[str]:
    """Every loaded module file under the evals root, relative to it (POSIX),
    `eval.py` included: the code a runner runs besides `lmnr`."""
    root = evals_root(eval_path)
    found: set[str] = set()
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not isinstance(file, str):
            continue
        path = Path(file).resolve()
        if not path.is_relative_to(root):
            continue
        relative = path.relative_to(root)
        if any(part in _SKIPPED_DIRS or part.startswith(".") for part in relative.parts[:-1]):
            continue
        found.add(relative.as_posix())
    return sorted(found)

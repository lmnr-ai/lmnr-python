"""Per-case files a service loads: `case_fixture("slack.json")`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lmnr.evals.context import Case
from lmnr.evals.errors import EvalConfigError


@dataclass(frozen=True)
class CaseFixture:
    """A file under each case's `tests/fixtures/`, resolved per trial."""

    path: str
    required: bool = False


def case_fixture(path: str, required: bool = False) -> CaseFixture:
    """A file under each case's `tests/fixtures/` (data the agent must not see),
    e.g. a mirror's seed: `seed=case_fixture("slack.json")`. A case without it
    gets no seed, unless `required=True`."""
    if (
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        raise EvalConfigError(
            f"case_fixture takes a path relative to tests/fixtures/, got {path!r}."
        )
    return CaseFixture(path, required)


SeedSpec = str | CaseFixture | None


def resolve_seed(seed: SeedSpec, case: Case, base_dir: Path | None, owner: str) -> Path | None:
    """The host file to load for this trial, or None when there is none.

    A plain string is relative to the file that declared the service; a
    `case_fixture` is the case's `tests/fixtures/<path>`.
    """
    if seed is None:
        return None
    if isinstance(seed, CaseFixture):
        path = case.fixture_path(seed.path)
        if path.is_file():
            return path
        if seed.required:
            raise FileNotFoundError(
                f"{owner}: case {case.name!r} has no tests/fixtures/{seed.path} "
                + "(case_fixture(..., required=True))."
            )
        return None
    path = Path(seed)
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    if not path.is_file():
        raise FileNotFoundError(f"{owner}: seed file {path} does not exist.")
    return path

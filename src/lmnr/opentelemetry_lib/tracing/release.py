"""Detects the release (usually the git SHA) of the running code.

Laminar groups an agent's traces into versions by release, so a detected value
must change on every code deploy. A value that can stay the same across
deploys would merge two versions; such sources are left out on purpose.
"""

from __future__ import annotations

import os
from pathlib import Path

from lmnr.sdk.utils import from_env

MAX_RELEASE_LENGTH = 128

# The GitHub Actions commit (CI runs, e.g. evals), explicitly set releases, then
# platform commit SHAs.
_RELEASE_ENV_VARS = (
    "GITHUB_SHA",
    "SENTRY_RELEASE",
    "VERCEL_GIT_COMMIT_SHA",
    "RENDER_GIT_COMMIT",
    "RAILWAY_GIT_COMMIT_SHA",
    "HEROKU_BUILD_COMMIT",
    "SOURCE_VERSION",  # Heroku
    "CODEBUILD_RESOLVED_SOURCE_VERSION",
    "CIRCLE_SHA1",
    # Deploy ids that change on every deploy but aren't commit SHAs.
    "K_REVISION",  # Cloud Run
    "GAE_DEPLOYMENT_ID",
    "FLY_IMAGE_REF",
)


def detect_release(explicit: str | None = None) -> str | None:
    """The first available of: `explicit`, `LMNR_RELEASE`, a known platform
    variable, a numeric AWS Lambda version, the checked-out git commit."""
    candidates = (
        lambda: explicit,
        lambda: from_env("LMNR_RELEASE"),
        *((lambda name=name: os.environ.get(name)) for name in _RELEASE_ENV_VARS),
        _lambda_version,
        _git_head,
    )
    for candidate in candidates:
        if release := _clean(candidate()):
            return release
    return None


def _clean(value: str | None) -> str | None:
    if not value or not (value := value.strip()):
        return None
    return value[:MAX_RELEASE_LENGTH]


def _lambda_version() -> str | None:
    # `$LATEST` stays the same across deploys; only published versions are numbered.
    version = os.environ.get("AWS_LAMBDA_FUNCTION_VERSION")
    function = os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    if not version or not version.isdigit() or not function:
        return None
    return f"{function}:{version}"


def _git_head() -> str | None:
    """The commit `.git/HEAD` points at, read from disk without running git."""
    try:
        git_dir = _find_git_dir(Path.cwd())
        if git_dir is None:
            return None
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.removeprefix("ref:").strip()
        # A linked worktree keeps its refs in the main repository.
        common_file = git_dir / "commondir"
        common_dir = (
            (git_dir / common_file.read_text().strip()).resolve()
            if common_file.is_file()
            else git_dir
        )
        for refs_dir in dict.fromkeys((git_dir, common_dir)):
            ref_file = refs_dir / ref
            if ref_file.is_file():
                return ref_file.read_text().strip()
        packed_refs = common_dir / "packed-refs"
        if packed_refs.is_file():
            for line in packed_refs.read_text().splitlines():
                sha, _, name = line.partition(" ")
                if name == ref:
                    return sha
    except Exception:
        return None
    return None


def _find_git_dir(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        dot_git = directory / ".git"
        if dot_git.is_dir():
            return dot_git
        if dot_git.is_file():
            # Worktrees and submodules: `.git` is a file with `gitdir: <path>`.
            content = dot_git.read_text().strip()
            if content.startswith("gitdir:"):
                return (directory / content.removeprefix("gitdir:").strip()).resolve()
    return None

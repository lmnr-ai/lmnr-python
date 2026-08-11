"""Best-effort collection of git state for trace metadata.

Collected once per process at `Laminar.initialize()` and merged into the
global trace metadata at the LOWEST precedence, so both `LMNR_TRACE_METADATA`
and the explicit `metadata=` init argument can override any of the keys. The
keys land on every span as `lmnr.association.properties.metadata.git.*` and
flow into `traces.metadata` server-side with no backend changes.

Collection must NEVER crash or block initialization: every git subprocess is
wrapped in a broad try/except with a short timeout, and a gitless environment
(CI checkout without `.git`, git binary missing, sandboxed runtime) degrades
to well-known CI environment variables, then to nothing.

Set `LMNR_DISABLE_GIT_METADATA` to a truthy value ("true", "1", "yes", "on")
to disable collection entirely.

This is a cross-language parity surface with the TS SDK
`lmnr-ts/packages/lmnr/src/git-metadata.ts` — keep the two line-comparable.
"""

import os
import subprocess
from functools import lru_cache

from lmnr.sdk.debug.config import _is_truthy

GIT_COMMIT_METADATA_KEY = "git.commit"
GIT_BRANCH_METADATA_KEY = "git.branch"
GIT_DIRTY_METADATA_KEY = "git.dirty"

_GIT_TIMEOUT_SECONDS = 1.5

# (commit env var, branch env var) per CI/deploy platform, checked in order.
# Used only when git itself is unavailable — e.g. gitless CI checkouts.
_CI_ENV_VARS: list[tuple[str, str]] = [
    ("GITHUB_SHA", "GITHUB_REF_NAME"),
    ("VERCEL_GIT_COMMIT_SHA", "VERCEL_GIT_COMMIT_REF"),
    ("CI_COMMIT_SHA", "CI_COMMIT_REF_NAME"),
    ("CIRCLE_SHA1", "CIRCLE_BRANCH"),
    ("RENDER_GIT_COMMIT", "RENDER_GIT_BRANCH"),
    ("RAILWAY_GIT_COMMIT_SHA", "RAILWAY_GIT_BRANCH"),
]


# Process-level opt-out recorded by `Laminar.initialize(disable_git_metadata=
# True)`. Kept here (not on the Laminar class) so EVERY collection point —
# global trace metadata AND eval run metadata — honors the same flag without
# re-plumbing the init argument through each call site.
_disabled: bool = False


def set_git_metadata_disabled(disabled: bool) -> None:
    """Record the initialize()-time opt-out for later collection points."""
    global _disabled
    _disabled = disabled


def _git_metadata_disabled() -> bool:
    return _disabled or _is_truthy(os.environ.get("LMNR_DISABLE_GIT_METADATA"))


def _run_git(*args: str) -> str | None:
    """Run a git command; return its stripped stdout, None on ANY failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def collect_git_metadata() -> dict[str, str | bool]:
    """Collect `git.commit` / `git.branch` / `git.dirty`, best-effort.

    Returns {} when collection is disabled — via the `disable_git_metadata`
    argument to `Laminar.initialize()` or the LMNR_DISABLE_GIT_METADATA env
    var. The disabled check runs on every call (NOT inside the cache) so an
    opt-out recorded after a prior collection still applies.

    `git.branch` is omitted on a detached HEAD, and `git.dirty` counts only
    tracked-file changes (untracked build artifacts should not flip it).
    """
    if _git_metadata_disabled():
        return {}
    return _collect_git_metadata_cached()


def reset_git_metadata_cache() -> None:
    """Reset the process-level collection cache. Exposed for tests only."""
    _collect_git_metadata_cached.cache_clear()


@lru_cache(maxsize=1)
def _collect_git_metadata_cached() -> dict[str, str | bool]:
    """The actual collection, cached for the process lifetime (git state is
    fixed once the process is running; re-running subprocesses per
    initialize()/evaluate() call would only add latency). Tests that vary cwd
    or env must call `reset_git_metadata_cache()`.
    """
    metadata: dict[str, str | bool] = {}
    commit = _run_git("rev-parse", "HEAD")
    if commit:
        metadata[GIT_COMMIT_METADATA_KEY] = commit
        branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
        if branch and branch != "HEAD":
            metadata[GIT_BRANCH_METADATA_KEY] = branch
        status = _run_git("status", "--porcelain", "--untracked-files=no")
        if status is not None:
            metadata[GIT_DIRTY_METADATA_KEY] = bool(status)
        return metadata

    # Not a git repo, no git binary, or an unborn HEAD — fall back to CI env
    # vars (a gap in Braintrust's approach: gitless CI checkouts get nothing).
    for commit_var, branch_var in _CI_ENV_VARS:
        env_commit = os.environ.get(commit_var)
        if env_commit:
            metadata[GIT_COMMIT_METADATA_KEY] = env_commit
            env_branch = os.environ.get(branch_var)
            if env_branch:
                metadata[GIT_BRANCH_METADATA_KEY] = env_branch
            break
    return metadata

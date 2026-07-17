import os
import subprocess

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from lmnr.sdk.evaluations import _with_git_metadata
from lmnr.sdk.git_metadata import (
    GIT_BRANCH_METADATA_KEY,
    GIT_COMMIT_METADATA_KEY,
    GIT_DIRTY_METADATA_KEY,
    collect_git_metadata,
)
from lmnr.sdk.laminar import Laminar

METADATA_ATTR_PREFIX = "lmnr.association.properties.metadata."


@pytest.fixture(autouse=True)
def setup_and_teardown(monkeypatch):
    """Reset Laminar state and the process-level git cache before each test.

    Re-enables git collection (conftest disables it session-wide via
    LMNR_DISABLE_GIT_METADATA) and restores `__global_metadata` afterwards so
    the git keys collected here never leak onto spans of tests that run after
    this module and assert on exact metadata contents.
    """
    monkeypatch.delenv("LMNR_DISABLE_GIT_METADATA", raising=False)

    original_initialized = Laminar._Laminar__initialized
    original_base_http_url = Laminar._Laminar__base_http_url
    original_project_api_key = Laminar._Laminar__project_api_key
    original_global_metadata = Laminar._Laminar__global_metadata

    Laminar._Laminar__initialized = False
    Laminar._Laminar__base_http_url = None
    Laminar._Laminar__project_api_key = None
    collect_git_metadata.cache_clear()

    yield

    Laminar._Laminar__initialized = original_initialized
    Laminar._Laminar__base_http_url = original_base_http_url
    Laminar._Laminar__project_api_key = original_project_api_key
    Laminar._Laminar__global_metadata = original_global_metadata
    collect_git_metadata.cache_clear()


def _git(cwd, *args):
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """A fresh git repo with one commit on branch `main`, as the cwd."""
    _git(tmp_path, "init", "--initial-branch=main")
    (tmp_path / "tracked.txt").write_text("v1")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "initial")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def no_git_dir(tmp_path, monkeypatch):
    """A cwd outside any git repo (git walks up, so hide parents via env)."""
    monkeypatch.chdir(tmp_path)
    # GIT_CEILING_DIRECTORIES does not apply to cwd itself, but tmp_path is
    # never a repo; it stops discovery from walking up into /tmp or /.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    return tmp_path


def _head_sha(cwd) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_collects_commit_branch_and_clean_state(git_repo):
    metadata = collect_git_metadata()
    head = _head_sha(git_repo)
    assert metadata[GIT_COMMIT_METADATA_KEY] == head
    assert metadata[GIT_BRANCH_METADATA_KEY] == "main"
    assert metadata[GIT_DIRTY_METADATA_KEY] is False


def test_dirty_flag_tracks_modified_files(git_repo):
    (git_repo / "tracked.txt").write_text("v2")
    metadata = collect_git_metadata()
    assert metadata[GIT_DIRTY_METADATA_KEY] is True


def test_untracked_files_do_not_flip_dirty(git_repo):
    (git_repo / "untracked.txt").write_text("scratch")
    metadata = collect_git_metadata()
    assert metadata[GIT_DIRTY_METADATA_KEY] is False


def test_detached_head_omits_branch(git_repo):
    head = _head_sha(git_repo)
    _git(git_repo, "checkout", "--detach", head)
    metadata = collect_git_metadata()
    assert metadata[GIT_COMMIT_METADATA_KEY] == head
    assert GIT_BRANCH_METADATA_KEY not in metadata


def test_no_repo_and_no_ci_env_collects_nothing(no_git_dir, monkeypatch):
    for var in ("GITHUB_SHA", "GITHUB_REF_NAME", "VERCEL_GIT_COMMIT_SHA"):
        monkeypatch.delenv(var, raising=False)
    assert collect_git_metadata() == {}


def test_ci_env_fallback_when_not_a_repo(no_git_dir, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_REF_NAME", "feature-x")
    metadata = collect_git_metadata()
    assert metadata[GIT_COMMIT_METADATA_KEY] == "abc123"
    assert metadata[GIT_BRANCH_METADATA_KEY] == "feature-x"
    assert GIT_DIRTY_METADATA_KEY not in metadata


def test_git_wins_over_ci_env(git_repo, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "not-the-real-sha")
    metadata = collect_git_metadata()
    assert metadata[GIT_COMMIT_METADATA_KEY] != "not-the-real-sha"


def test_env_opt_out_disables_collection(git_repo, monkeypatch):
    monkeypatch.setenv("LMNR_DISABLE_GIT_METADATA", "true")
    assert collect_git_metadata() == {}


def test_initialize_stamps_git_metadata_on_spans(
    git_repo, span_exporter: InMemorySpanExporter
):
    span_exporter.clear()
    Laminar.initialize(project_api_key="test_key")
    span = Laminar.start_span("test")
    span.end()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = spans[0].attributes
    assert attributes[METADATA_ATTR_PREFIX + GIT_COMMIT_METADATA_KEY]
    assert attributes[METADATA_ATTR_PREFIX + GIT_BRANCH_METADATA_KEY] == "main"
    assert attributes[METADATA_ATTR_PREFIX + GIT_DIRTY_METADATA_KEY] is False


def test_initialize_user_metadata_overrides_git(
    git_repo, span_exporter: InMemorySpanExporter
):
    span_exporter.clear()
    Laminar.initialize(
        project_api_key="test_key",
        metadata={GIT_COMMIT_METADATA_KEY: "user-override"},
    )
    span = Laminar.start_span("test")
    span.end()

    spans = span_exporter.get_finished_spans()
    assert (
        spans[0].attributes[METADATA_ATTR_PREFIX + GIT_COMMIT_METADATA_KEY]
        == "user-override"
    )


def test_initialize_env_trace_metadata_overrides_git(
    git_repo, monkeypatch, span_exporter: InMemorySpanExporter
):
    span_exporter.clear()
    monkeypatch.setenv(
        "LMNR_TRACE_METADATA", '{"git.commit": "env-override"}'
    )
    Laminar.initialize(project_api_key="test_key")
    span = Laminar.start_span("test")
    span.end()

    spans = span_exporter.get_finished_spans()
    assert (
        spans[0].attributes[METADATA_ATTR_PREFIX + GIT_COMMIT_METADATA_KEY]
        == "env-override"
    )


def test_initialize_disable_git_metadata_param(
    git_repo, span_exporter: InMemorySpanExporter
):
    span_exporter.clear()
    Laminar.initialize(project_api_key="test_key", disable_git_metadata=True)
    span = Laminar.start_span("test")
    span.end()

    spans = span_exporter.get_finished_spans()
    attributes = spans[0].attributes
    assert METADATA_ATTR_PREFIX + GIT_COMMIT_METADATA_KEY not in attributes
    assert METADATA_ATTR_PREFIX + GIT_BRANCH_METADATA_KEY not in attributes
    assert METADATA_ATTR_PREFIX + GIT_DIRTY_METADATA_KEY not in attributes


def test_with_git_metadata_stamps_eval_run_metadata(git_repo):
    metadata = _with_git_metadata({"user": "value"})
    assert metadata["user"] == "value"
    assert metadata[GIT_COMMIT_METADATA_KEY]
    assert metadata[GIT_BRANCH_METADATA_KEY] == "main"


def test_with_git_metadata_user_metadata_wins(git_repo):
    metadata = _with_git_metadata({GIT_COMMIT_METADATA_KEY: "user-override"})
    assert metadata[GIT_COMMIT_METADATA_KEY] == "user-override"


def test_with_git_metadata_no_git_returns_unchanged(no_git_dir, monkeypatch):
    for var in ("GITHUB_SHA", "GITHUB_REF_NAME", "VERCEL_GIT_COMMIT_SHA"):
        monkeypatch.delenv(var, raising=False)
    assert _with_git_metadata(None) is None
    assert _with_git_metadata({"a": 1}) == {"a": 1}

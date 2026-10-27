"""Tests for release detection and the `lmnr.span.release` stamp."""

from __future__ import annotations

import os
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from lmnr.opentelemetry_lib.tracing import release as release_module
from lmnr.opentelemetry_lib.tracing.attributes import SPAN_RELEASE
from lmnr.opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from lmnr.opentelemetry_lib.tracing.release import MAX_RELEASE_LENGTH, detect_release

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No release env vars, and a working directory outside any git repository."""
    for name in (
        "LMNR_RELEASE",
        "AWS_LAMBDA_FUNCTION_VERSION",
        "AWS_LAMBDA_FUNCTION_NAME",
        *release_module._RELEASE_ENV_VARS,
    ):
        monkeypatch.delenv(name, raising=False)
    # `from_env` also reads a `.env` file; the tests must not pick one up.
    monkeypatch.setattr(release_module, "from_env", os.environ.get)
    # tmp_path can sit inside a git checkout; ignore anything found above it.
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    find_git_dir = release_module._find_git_dir
    root = tmp_path.resolve()

    def find_git_dir_under_tmp(start: Path) -> Path | None:
        found = find_git_dir(start)
        return found if found is not None and found.resolve().is_relative_to(root) else None

    monkeypatch.setattr(release_module, "_find_git_dir", find_git_dir_under_tmp)


def _make_repo(path: Path, head: str) -> Path:
    git_dir = path / ".git"
    git_dir.mkdir(parents=True)
    _ = (git_dir / "HEAD").write_text(head + "\n")
    return git_dir


# --- precedence --------------------------------------------------------------


def test_nothing_detected() -> None:
    assert detect_release() is None


def test_explicit_wins_over_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMNR_RELEASE", "from-lmnr-env")
    monkeypatch.setenv("VERCEL_GIT_COMMIT_SHA", SHA)
    assert detect_release("v1.2.3") == "v1.2.3"


def test_lmnr_release_wins_over_platform_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMNR_RELEASE", "from-lmnr-env")
    monkeypatch.setenv("VERCEL_GIT_COMMIT_SHA", SHA)
    assert detect_release() == "from-lmnr-env"


def test_sentry_release_wins_over_platform_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTRY_RELEASE", "my-app@1.0.0")
    monkeypatch.setenv("RENDER_GIT_COMMIT", SHA)
    assert detect_release() == "my-app@1.0.0"


def test_github_sha_wins_over_sentry_release(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.setenv("SENTRY_RELEASE", "my-app@1.0.0")
    assert detect_release() == SHA


@pytest.mark.parametrize("name", release_module._RELEASE_ENV_VARS)
def test_each_platform_var(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, SHA)
    assert detect_release() == SHA


def test_platform_var_wins_over_git(monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _make_repo(Path.cwd(), OTHER_SHA)
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", SHA)
    assert detect_release() == SHA


def test_empty_values_fall_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTRY_RELEASE", "   ")
    monkeypatch.setenv("CIRCLE_SHA1", SHA)
    assert detect_release("") == SHA


def test_value_is_trimmed_and_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    assert detect_release(f"  {SHA}\n") == SHA
    assert detect_release("x" * 500) == "x" * MAX_RELEASE_LENGTH


# --- AWS Lambda --------------------------------------------------------------


def test_lambda_numbered_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "agent")
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_VERSION", "42")
    assert detect_release() == "agent:42"


def test_lambda_latest_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "agent")
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_VERSION", "$LATEST")
    assert detect_release() is None


def test_lambda_version_wins_over_git(monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _make_repo(Path.cwd(), SHA)
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "agent")
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_VERSION", "7")
    assert detect_release() == "agent:7"


# --- git ---------------------------------------------------------------------


def test_git_detached_head() -> None:
    _ = _make_repo(Path.cwd(), SHA)
    assert detect_release() == SHA


def test_git_branch_ref_file() -> None:
    git_dir = _make_repo(Path.cwd(), "ref: refs/heads/main")
    (git_dir / "refs" / "heads").mkdir(parents=True)
    _ = (git_dir / "refs" / "heads" / "main").write_text(SHA + "\n")
    assert detect_release() == SHA


def test_git_packed_refs() -> None:
    git_dir = _make_repo(Path.cwd(), "ref: refs/heads/main")
    _ = (git_dir / "packed-refs").write_text(
        "\n".join(
            (
                "# pack-refs with: peeled fully-peeled sorted",
                f"{OTHER_SHA} refs/heads/other",
                f"{SHA} refs/heads/main",
                f"^{OTHER_SHA}",
            )
        )
    )
    assert detect_release() == SHA


def test_git_found_from_a_subdirectory(monkeypatch: pytest.MonkeyPatch) -> None:
    _ = _make_repo(Path.cwd(), SHA)
    nested = Path.cwd() / "src" / "agent"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert detect_release() == SHA


def test_git_linked_worktree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Main repository: branch refs live here, packed.
    main_git = _make_repo(tmp_path / "main", "ref: refs/heads/main")
    _ = (main_git / "packed-refs").write_text(f"{SHA} refs/heads/feature\n")
    # Worktree metadata: its own HEAD, plus `commondir` pointing back at main.
    worktree_meta = main_git / "worktrees" / "feature"
    worktree_meta.mkdir(parents=True)
    _ = (worktree_meta / "HEAD").write_text("ref: refs/heads/feature\n")
    _ = (worktree_meta / "commondir").write_text("../..\n")
    # Worktree checkout: `.git` is a file.
    checkout = tmp_path / "feature"
    checkout.mkdir()
    _ = (checkout / ".git").write_text(f"gitdir: {worktree_meta}\n")
    monkeypatch.chdir(checkout)
    assert detect_release() == SHA


def test_git_unresolvable_ref() -> None:
    _ = _make_repo(Path.cwd(), "ref: refs/heads/main")
    assert detect_release() is None


# --- span stamping -----------------------------------------------------------


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def make_tracer(
    exporter: InMemorySpanExporter,
) -> Generator[Callable[[str | None], Tracer], None, None]:
    """A plain OTel tracer behind a `LaminarSpanProcessor`, outside the global provider."""
    providers: list[TracerProvider] = []

    def make(release: str | None) -> Tracer:
        provider = TracerProvider()
        provider.add_span_processor(
            LaminarSpanProcessor(exporter=exporter, disable_batch=True, release=release)
        )
        providers.append(provider)
        return provider.get_tracer("plain-otel")

    yield make
    for provider in providers:
        provider.shutdown()


def test_release_is_stamped_on_every_span(
    make_tracer: Callable[[str | None], Tracer], exporter: InMemorySpanExporter
) -> None:
    # No Laminar context is attached: the stamp must not depend on one.
    tracer = make_tracer("abc1234")
    with tracer.start_as_current_span("root"), tracer.start_as_current_span("child"):
        pass

    spans = exporter.get_finished_spans()
    assert {span.name for span in spans} == {"root", "child"}
    for span in spans:
        assert (span.attributes or {})[SPAN_RELEASE] == "abc1234"


def test_detected_release_is_stamped(
    monkeypatch: pytest.MonkeyPatch,
    make_tracer: Callable[[str | None], Tracer],
    exporter: InMemorySpanExporter,
) -> None:
    monkeypatch.setenv("RENDER_GIT_COMMIT", SHA)
    tracer = make_tracer(None)
    with tracer.start_as_current_span("root"):
        pass

    [span] = exporter.get_finished_spans()
    assert (span.attributes or {})[SPAN_RELEASE] == SHA


def test_no_release_no_attribute(
    make_tracer: Callable[[str | None], Tracer], exporter: InMemorySpanExporter
) -> None:
    tracer = make_tracer(None)
    with tracer.start_as_current_span("root"):
        pass

    [span] = exporter.get_finished_spans()
    assert SPAN_RELEASE not in (span.attributes or {})

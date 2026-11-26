"""`Image` builder and composition: golden Dockerfiles and the rules around them.

The golden files are in `tests/data/evals_images/`. After an intended change,
regenerate them with `LMNR_UPDATE_GOLDEN=1 uv run pytest tests/test_evals_images.py`
and review the diff. The images are declared in
`tests/data/evals_project/evals/_shared/images.py`, so provenance comments
don't move when this file changes.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

from lmnr.evals import EvalConfigError, EvalRunner, Image
from lmnr.evals.image import APP_STAGE, UV_IMAGE, compile_image

DATA = Path(__file__).parent / "data"
PROJECT = DATA / "evals_project"
ROOT = PROJECT / "evals"
GOLDEN = DATA / "evals_images"


@pytest.fixture(scope="module")
def images() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "evals_golden_images", ROOT / "_shared" / "images.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_golden(name: str, text: str) -> None:
    path = GOLDEN / f"{name}.Dockerfile"
    if os.environ.get("LMNR_UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(text)
    assert path.is_file(), f"{path} is missing; run with LMNR_UPDATE_GOLDEN=1"
    assert text == path.read_text()


def test_builder_steps_golden(images: ModuleType):
    compiled = compile_image(images.BUILDER, root=ROOT, fallback_dir=ROOT)
    assert compiled.kind == "generated"
    # The deepest folder holding the declaring file and every added path.
    assert compiled.context == PROJECT.resolve()
    assert compiled.dockerfile_text is not None
    check_golden("builder", compiled.dockerfile_text)
    assert compiled.steps[0] == {
        "origin": "_shared/images.py:10",
        "step": "from python:3.13-slim-trixie",
    }
    # Each step names its own line of the chain; paths are relative to the evals root.
    assert {"origin": "_shared/images.py:15", "step": "uv_sync ../pyproject.toml"} in compiled.steps


def test_uv_image_base_has_uv(images: ModuleType):
    compiled = compile_image(images.UV_BASE, root=ROOT, fallback_dir=ROOT)
    assert compiled.dockerfile_text is not None
    assert UV_IMAGE not in compiled.dockerfile_text
    check_golden("uv-base", compiled.dockerfile_text)


def test_services_golden(images: ModuleType):
    runner = images.ServicesRunner
    assert runner.validate() == []
    compiled = compile_image(runner.image, runner.services, root=ROOT, fallback_dir=ROOT)
    assert compiled.dockerfile_text is not None
    check_golden("services", compiled.dockerfile_text)


def test_dockerfile_composition_golden(images: ModuleType):
    compiled = compile_image(
        images.COMPOSED, images.COMPOSED_SERVICES, root=ROOT, fallback_dir=ROOT
    )
    assert compiled.kind == "generated"
    # The user's context, so their COPY lines still work.
    assert compiled.context == PROJECT.resolve()
    assert compiled.dockerfile_text is not None
    text = compiled.dockerfile_text
    check_golden("composed", text)
    # Parser directives stay first; the last stage gets a name; USER comes back.
    assert text.startswith("# syntax=docker/dockerfile:1\n# from support/agent/Dockerfile\n")
    assert f"FROM python:${{PYTHON}}-slim-bookworm AS {APP_STAGE}\n" in text
    assert text.rstrip().endswith("USER app")


def test_plain_bases_stay_as_they_are(tmp_path: Path):
    registry = compile_image(Image.from_registry("python:3.13"), fallback_dir=tmp_path)
    assert (registry.kind, registry.ref) == ("registry", "python:3.13")
    _ = (tmp_path / "Dockerfile").write_text("FROM python:3.13\n")
    dockerfile = compile_image(
        Image.from_dockerfile(str(tmp_path / "Dockerfile")), fallback_dir=tmp_path
    )
    assert dockerfile.kind == "dockerfile"
    assert dockerfile.describe()["dockerfile"] == str(tmp_path / "Dockerfile")


def test_steps_return_new_images_that_share_a_prefix():
    base = Image.python("3.13").apt_install("curl")
    ocr = base.apt_install("tesseract-ocr")
    assert len(base.steps) == 1
    assert ocr.steps[0] is base.steps[0]
    assert [s.args for s in ocr.steps] == [("curl",), ("tesseract-ocr",)]


def test_existing_stage_name_is_reused_and_root_user_is_left_alone(tmp_path: Path):
    _ = (tmp_path / "Dockerfile").write_text("FROM python:3.13 AS app\nRUN true\n")
    compiled = compile_image(
        Image.from_dockerfile(str(tmp_path / "Dockerfile")).run_commands("echo hi"),
        fallback_dir=tmp_path,
    )
    text = compiled.dockerfile_text or ""
    assert "FROM python:3.13 AS app\n" in text
    assert "\nFROM app\n" in text
    assert "USER" not in text
    # A builder step on a Dockerfile image doesn't add a WORKDIR.
    assert "WORKDIR" not in text


def test_dockerfile_ignore_file_is_passed_on(tmp_path: Path):
    _ = (tmp_path / "Dockerfile").write_text("FROM python:3.13\n")
    _ = (tmp_path / "Dockerfile.dockerignore").write_text("data/\n")
    compiled = compile_image(
        Image.from_dockerfile(str(tmp_path / "Dockerfile")).apt_install("jq"), fallback_dir=tmp_path
    )
    assert compiled.describe()["ignore_file"] == str(tmp_path / "Dockerfile.dockerignore")


def test_paths_outside_a_dockerfile_context_are_an_error(tmp_path: Path):
    (tmp_path / "ctx").mkdir()
    _ = (tmp_path / "ctx" / "Dockerfile").write_text("FROM python:3.13\n")
    (tmp_path / "app").mkdir()
    image = Image.from_dockerfile(str(tmp_path / "ctx" / "Dockerfile")).add_dir(
        str(tmp_path / "app"), "/srv/app"
    )
    with pytest.raises(EvalConfigError, match="outside the build context"):
        _ = compile_image(image, fallback_dir=tmp_path)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda d: Image.python("3.13").uv_sync(str(d / "pyproject.toml")), "no uv.lock"),
        (lambda d: Image.python("3.13").add_file(str(d / "missing.txt"), "/x"), "does not exist"),
        (
            lambda d: Image.python("3.13").add_dir(str(d / "pyproject.toml"), "/x"),
            "not a directory",
        ),
        (lambda d: Image.node("22").npm_ci(str(d), "/srv/web"), "package.json does not exist"),
    ],
)
def test_missing_files_fail_at_compile_time(tmp_path: Path, make, message: str):
    _ = (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    with pytest.raises(EvalConfigError, match=message):
        _ = compile_image(make(tmp_path), fallback_dir=tmp_path)


def test_step_arguments_are_checked_when_declared():
    with pytest.raises(EvalConfigError, match="absolute"):
        _ = Image.python("3.13").add_dir(".", "relative/path")
    with pytest.raises(EvalConfigError, match="non-empty strings"):
        _ = Image.python("3.13").apt_install()
    with pytest.raises(EvalConfigError, match="must be a name"):
        _ = Image.python("3.13").env(**{"BAD-NAME": "x"})


def test_shared_runner_image_resolves_from_its_own_file(tmp_path: Path):
    """A subclass in another folder extends the shared image; each step keeps
    its own origin and paths."""
    root = tmp_path / "evals"
    (root / "_shared").mkdir(parents=True)
    (tmp_path / "app").mkdir()
    _ = (root / "_shared" / "support.py").write_text(
        textwrap.dedent(
            """\
            from lmnr.evals import EvalRunner, Image

            class SupportRunner(EvalRunner):
                image = Image.python("3.13").add_dir("../../app", "/srv/app")
            """
        )
    )
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location(
            "_shared.support", root / "_shared" / "support.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runner: type[EvalRunner] = module.SupportRunner

        class Refunds(runner):
            image = runner.image.apt_install("jq")  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

        compiled = compile_image(Refunds.image, root=root, fallback_dir=root)  # pyright: ignore[reportArgumentType]
    finally:
        sys.path.remove(str(root))
    # The only added path is the app folder, so it is the context.
    assert compiled.context == (tmp_path / "app").resolve()
    text = compiled.dockerfile_text or ""
    assert '# _shared/support.py:4 (SupportRunner)\nCOPY ["./", "/srv/app/"]' in text
    assert "Refunds)\nRUN apt-get update" in text

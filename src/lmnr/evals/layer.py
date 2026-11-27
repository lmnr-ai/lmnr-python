"""The Laminar layer: Python and `lmnr` in `/opt/lmnr`, in every image of an
`eval.py` runner.

It is the in-container runtime for the mock server and for the runner's
helpers (`relay.py http` / `wait` / `rows`, `lmnr.evals.relay`), independent of
the app's language and Python. A stage `FROM debian:trixie-slim` installs a
standalone Python with uv, a venv with `lmnr` (plus the runner's
`mock_dependencies`) and the `relay.py` launcher; the final image copies
`/opt/lmnr`. Nothing in it depends on the app, so it stays cached across app
changes, and it holds no user code: handlers, fixtures and config are uploaded
per trial.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from lmnr.evals.errors import EvalConfigError
from lmnr.evals.image import UV_IMAGE, PathMap

LAYER_DIR = "/opt/lmnr"
LAYER_PYTHON = "3.13"
LAYER_STAGE = "lmnr-layer"
STAGE_BASE = "debian:trixie-slim"
# The newest glibc the layer's wheels may need (`uv pip install
# --python-platform`), whatever the stage's own: the layer goes into every
# image, so it must run on bookworm (2.36) and Ubuntu 24.04 apps too. The
# standalone Python itself needs only 2.17.
WHEEL_GLIBC = "2_36"
VENV_PYTHON = f"{LAYER_DIR}/venv/bin/python"
RELAY_FILE = f"{LAYER_DIR}/relay.py"
# The named build context lmnr-cli passes for a local `lmnr` checkout
# (`--build-context lmnr-dev=<dir with the wheel>`).
WHEEL_CONTEXT = "lmnr-dev"
# How the runner calls the relay in the container. `-I`: the app's
# PYTHONPATH and the like must not reach the layer's Python. Tests point it at
# `lmnr/evals/relay.py` on the host.
RELAY_COMMAND = f"{VENV_PYTHON} -I {RELAY_FILE}"

# Runs `lmnr/evals/relay.py` by path: importing the `lmnr` package takes
# seconds, the relay itself needs only the standard library.
LAUNCHER = (
    "import runpy, sysconfig\n"
    + 'runpy.run_path(sysconfig.get_path("purelib") + "/lmnr/evals/relay.py", '
    + 'run_name="__main__")\n'
)


# Removed from the standalone Python (relative to its root, and to its stdlib).
_UNUSED = (
    "lib/libpython3*",
    "lib/libtcl*",
    "lib/libtk*",
    "lib/tcl*",
    "lib/tk*",
    "lib/itcl*",
    "lib/thread*",
    "share",
    "include",
    "bin/idle*",
    "bin/pip*",
    "bin/pydoc*",
)
_UNUSED_STDLIB = (
    "idlelib",
    "tkinter",
    "turtledemo",
    "ensurepip",
    "test",
    "pydoc_data",
    "site-packages/pip*",
    "lib-dynload/_tkinter*",
)


def relay_command() -> str:
    return RELAY_COMMAND


@dataclass(frozen=True)
class LaminarLayer:
    """An `ImageContributor`: the stage and the final image's `COPY`.

    `lmnr` is the requirement to install (`lmnr==<the version describe runs
    as>`, or a git URL), or None to install the wheel lmnr-cli built from a
    local checkout, passed as the `lmnr-dev` build context.
    """

    lmnr: str | None
    mock_dependencies: tuple[str, ...] = ()

    def image_title(self, root: Path | None) -> str:
        what = self.lmnr if self.lmnr is not None else f"lmnr wheel ({WHEEL_CONTEXT})"
        extra = f" + {', '.join(self.mock_dependencies)}" if self.mock_dependencies else ""
        return f"Laminar layer: Python {LAYER_PYTHON}, {what}{extra}"

    def image_step(self) -> str:
        return "laminar layer"

    def host_paths(self) -> list[Path]:
        return []

    def build_contexts(self) -> dict[str, str]:
        """Named build contexts the stage reads, and what lmnr-cli puts there."""
        return {WHEEL_CONTEXT: "lmnr wheel"} if self.lmnr is None else {}

    def bundle_stages(self, rel: PathMap) -> list[str]:
        lines = [
            f"FROM {STAGE_BASE} AS {LAYER_STAGE}",
            f"COPY --from={UV_IMAGE} /uv /usr/local/bin/",
            f"ENV UV_PYTHON_INSTALL_DIR={LAYER_DIR}/python UV_PYTHON_PREFERENCE=only-managed "
            + "UV_NO_CACHE=1 UV_COMPILE_BYTECODE=1",
            # Its own layer: kept when lmnr or the dependencies change.
            f"RUN uv python install {LAYER_PYTHON} && "
            + f"uv venv --python {LAYER_PYTHON} {LAYER_DIR}/venv",
        ]
        requirements = [r for r in (self.lmnr, *self.mock_dependencies) if r is not None]
        if any("git+" in r for r in requirements):
            # A git requirement (lmnr's or a mock dependency's): the stage has no git.
            lines.append(
                "RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                + "--no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*"
            )
        if self.lmnr is None:
            lines.append(f"COPY --from={WHEEL_CONTEXT} / /tmp/lmnr-dist/")
            packages = ["/tmp/lmnr-dist/*.whl"]
        else:
            packages = []
        packages += [shlex.quote(r) for r in requirements]
        lines.append(
            f"RUN uv pip install --python {VENV_PYTHON} "
            + f'--python-platform "$(uname -m)-manylinux_{WHEEL_GLIBC}" {" ".join(packages)}'
        )
        lines += [
            f"COPY <<'EOF' {RELAY_FILE}",
            LAUNCHER.rstrip("\n"),
            "EOF",
            # What nothing here uses: the static binary doesn't need
            # libpython.so, and nothing builds against the headers once the
            # packages are in (184 MB -> 115 MB).
            f"RUN cd {LAYER_DIR}/python/cpython-{LAYER_PYTHON}.*/ && rm -rf "
            + " ".join(_UNUSED)
            + f" lib/python{LAYER_PYTHON}/"
            + f" lib/python{LAYER_PYTHON}/".join(_UNUSED_STDLIB),
            # Fails the build when this lmnr has no mock server.
            f"RUN {VENV_PYTHON} -c 'import lmnr.evals.mocks.server' && "
            + f"echo '[]' | {RELAY_COMMAND} rows",
        ]
        return lines

    def install_steps(self, rel: PathMap) -> list[str]:
        hint = (
            "lmnr.evals: this image can't run the Laminar layer's Python (glibc 2.17+), which "
            + "the runner needs in every trial. Use a Debian or Ubuntu base."
        )
        return [
            f"COPY --from={LAYER_STAGE} {LAYER_DIR} {LAYER_DIR}",
            f"RUN {VENV_PYTHON} -c '' || {{ echo {shlex.quote(hint)} >&2; exit 1; }}",
        ]


def validate_mock_dependencies(owner: str, dependencies: object) -> tuple[str, ...]:
    if not isinstance(dependencies, (list, tuple)) or not all(
        isinstance(d, str) and d.strip() and "\n" not in d for d in dependencies
    ):
        raise EvalConfigError(
            f"{owner}.mock_dependencies must be a list of pip requirements "
            + f'(["anthropic", "httpx>=0.27"]), got {dependencies!r}.'
        )
    return tuple(d.strip() for d in dependencies)

"""Shell for service bundle stages and their installs.

A service is installed from an image (its official one by default): a stage
`FROM <source>` copies the binaries plus every shared library `ldd` reports
(except glibc's own) into `/bundle/`, laid out as they land in the final image,
and the final image does `COPY --from=<stage> /bundle/ /`. So an app change
never re-downloads a service, and the service doesn't depend on what the app's
base has installed. Binaries run through small wrappers that put the bundled
libraries first on `LD_LIBRARY_PATH`.

The final image needs glibc at least as new as the source image's: the install
step checks it and fails the build with a hint otherwise.
"""

from __future__ import annotations

import re
import shlex

BUNDLE_DIR = "/bundle"
# Where a bundle records its source image's glibc version, for `install_check`.
GLIBC_MARKER = "/usr/local/share/.glibc-required-"

# glibc's own libraries come from the final image, never from the bundle.
_GLIBC = (
    "libc.so.*|libm.so.*|libpthread.so.*|libdl.so.*|librt.so.*|libresolv.so.*|"
    "libutil.so.*|libanl.so.*|libnsl.so.*|libnss_*|libmvec.so.*|libBrokenLocale.so.*|"
    "libthread_db.so.*|libcrypt.so.1|ld-linux*|ld64.so.*"
)

_PRELUDE = f"""set -eu
out={BUNDLE_DIR}"""

# Shell functions a stage's script can call; `stage` defines the ones it uses.
_FUNCTIONS = {
    "collect_deps": f"""# collect_deps DIR FILE...: copy the shared libraries FILEs need into DIR.
collect_deps() {{
  dir=$1; shift
  command -v ldd >/dev/null 2>&1 || {{ echo "ldd is not in the source image" >&2; exit 1; }}
  mkdir -p "$out$dir"
  for f in "$@"; do ldd "$f" 2>/dev/null | awk '/=> \\//{{print $3}}'; done | sort -u |
  while read -r lib; do
    case "${{lib##*/}}" in {_GLIBC}) continue ;; esac
    cp -L "$lib" "$out$dir/"
  done
}}""",
    "wrap": """# wrap REAL WRAPPER LIBDIR: a script at WRAPPER that runs REAL with LIBDIR's libraries.
wrap() {
  mkdir -p "$out${2%/*}"
  printf '#!/bin/sh\\nLD_LIBRARY_PATH=%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} exec %s "$@"\\n' \\
    "$3" "$1" > "$out$2"
  chmod 755 "$out$2"
}""",
    "record_glibc": """# record_glibc NAME: the source image's glibc version, checked by the install step.
# Not under /tmp: COPY would give the image's /tmp the bundle's mode (0755).
record_glibc() {
  mkdir -p "$out/usr/local/share"
  getconf GNU_LIBC_VERSION | awk '{print $2}' > "$out/usr/local/share/.glibc-required-$1"
}""",
}


def stage(source: str, name: str, script: str) -> list[str]:
    """A bundle stage: `FROM source AS name`, then `script`, after the shell
    functions it calls."""
    used = [body for function, body in _FUNCTIONS.items() if re.search(rf"\b{function}\b", script)]
    return [f"FROM {source} AS {name}", "RUN <<'EOF'", _PRELUDE, *used, script.strip("\n"), "EOF"]


def install_check(name: str, display: str, source: str, users: list[tuple[str, str]]) -> str:
    """The final image's `RUN` after `COPY --from`: glibc is new enough,
    `runuser` exists, and the service's OS users exist (`(user, home)`)."""
    hint = (
        f"{display} comes from {source}. Use a Debian or Ubuntu base with a newer glibc, "
        + "or pass source= an image built on an older distribution."
    )
    lines = [
        f"need=$(cat {GLIBC_MARKER}{name}) && rm -f {GLIBC_MARKER}{name}",
        "have=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}' || true)",
        'if [ -z "$have" ]; then echo '
        + shlex.quote(f"lmnr.evals: this image has no glibc (Alpine?). {hint}")
        + " >&2; exit 1; fi",
        'if [ "$(printf \'%s\\n%s\\n\' "$need" "$have" | sort -V | head -n1)" != "$need" ]; then '
        + 'echo "lmnr.evals: this image has glibc $have, but $need or newer is needed." '
        + shlex.quote(hint)
        + " >&2; exit 1; fi",
        "command -v runuser >/dev/null 2>&1 || { echo "
        + shlex.quote(f"lmnr.evals: {display} needs runuser (util-linux).")
        + " >&2; exit 1; }",
    ]
    for user, home in users:
        lines.append(
            f"(id -u {user} >/dev/null 2>&1 || useradd --system --user-group --no-create-home "
            + f"--home-dir {home} --shell /usr/sbin/nologin {user})"
        )
    return "RUN " + " && \\\n    ".join(lines)

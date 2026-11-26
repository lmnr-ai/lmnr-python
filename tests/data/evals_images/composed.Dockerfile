# syntax=docker/dockerfile:1
# from support/agent/Dockerfile
ARG PYTHON=3.13
FROM python:${PYTHON}-slim-bookworm AS deps
RUN pip install --no-cache-dir httpx

FROM python:${PYTHON}-slim-bookworm AS lmnr-app
COPY --from=deps /usr/local/lib /usr/local/lib
RUN useradd -m app
COPY app/ /srv/app/app/
USER app
WORKDIR /srv/app

# Added by lmnr.evals. Each step names the code that declared it.

# service: Valkey 8 (_shared/images.py:27)
FROM valkey/valkey:8-bookworm AS svc-valkey
RUN <<'EOF'
set -eu
out=/bundle
# collect_deps DIR FILE...: copy the shared libraries FILEs need into DIR.
collect_deps() {
  dir=$1; shift
  command -v ldd >/dev/null 2>&1 || { echo "ldd is not in the source image" >&2; exit 1; }
  mkdir -p "$out$dir"
  for f in "$@"; do ldd "$f" 2>/dev/null | awk '/=> \//{print $3}'; done | sort -u |
  while read -r lib; do
    case "${lib##*/}" in libc.so.*|libm.so.*|libpthread.so.*|libdl.so.*|librt.so.*|libresolv.so.*|libutil.so.*|libanl.so.*|libnsl.so.*|libnss_*|libmvec.so.*|libBrokenLocale.so.*|libthread_db.so.*|libcrypt.so.1|ld-linux*|ld64.so.*) continue ;; esac
    cp -L "$lib" "$out$dir/"
  done
}
# wrap REAL WRAPPER LIBDIR: a script at WRAPPER that runs REAL with LIBDIR's libraries.
wrap() {
  mkdir -p "$out${2%/*}"
  printf '#!/bin/sh\nLD_LIBRARY_PATH=%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} exec %s "$@"\n' \
    "$3" "$1" > "$out$2"
  chmod 755 "$out$2"
}
# record_glibc NAME: the source image's glibc version, checked by the install step.
# Not under /tmp: COPY would give the image's /tmp the bundle's mode (0755).
record_glibc() {
  mkdir -p "$out/usr/local/share"
  getconf GNU_LIBC_VERSION | awk '{print $2}' > "$out/usr/local/share/.glibc-required-$1"
}
mkdir -p "$out/usr/local/lib/valkey/bin"
for b in valkey-server valkey-cli; do cp -L "$(command -v $b)" "$out/usr/local/lib/valkey/bin/$b"; done
collect_deps /usr/local/lib/valkey/libs "$out/usr/local/lib/valkey"/bin/*
for b in valkey-server valkey-cli; do wrap /usr/local/lib/valkey/bin/$b /usr/local/bin/$b /usr/local/lib/valkey/libs; done
ln -s valkey-cli "$out/usr/local/bin/redis-cli"
ln -s valkey-server "$out/usr/local/bin/redis-server"
record_glibc valkey
EOF

FROM lmnr-app
USER root

# service: Valkey 8 (_shared/images.py:27)
COPY --from=svc-valkey /bundle/ /
RUN need=$(cat /usr/local/share/.glibc-required-valkey) && rm -f /usr/local/share/.glibc-required-valkey && \
    have=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}' || true) && \
    if [ -z "$have" ]; then echo 'lmnr.evals: this image has no glibc (Alpine?). Valkey comes from valkey/valkey:8-bookworm. Use a Debian or Ubuntu base with a newer glibc, or pass source= an image built on an older distribution.' >&2; exit 1; fi && \
    if [ "$(printf '%s\n%s\n' "$need" "$have" | sort -V | head -n1)" != "$need" ]; then echo "lmnr.evals: this image has glibc $have, but $need or newer is needed." 'Valkey comes from valkey/valkey:8-bookworm. Use a Debian or Ubuntu base with a newer glibc, or pass source= an image built on an older distribution.' >&2; exit 1; fi && \
    command -v runuser >/dev/null 2>&1 || { echo 'lmnr.evals: Valkey needs runuser (util-linux).' >&2; exit 1; } && \
    (id -u valkey >/dev/null 2>&1 || useradd --system --user-group --no-create-home --home-dir /var/lib/valkey --shell /usr/sbin/nologin valkey)

# _shared/images.py:26
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends jq && rm -rf /var/lib/apt/lists/*

USER app

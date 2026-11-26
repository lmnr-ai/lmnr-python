"""`Mirror`: a service that stands in for a SaaS API (Slack, Stripe, ...)."""

from __future__ import annotations

import json
import shlex
import uuid
import zlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlencode

from typing_extensions import override

from lmnr.evals.errors import EvalConfigError
from lmnr.evals.fixtures import SeedSpec, resolve_seed
from lmnr.evals.services.base import HOST, STATE_DIR, Service, run_script

if TYPE_CHECKING:
    from lmnr.evals.context import AgentContext, Context

STATE_FILE = "state.json"
# Default ports are picked from the name in this range, so a runner's mirrors
# rarely collide; `port=` settles it when they do.
DEFAULT_PORT_BASE = 18000
DEFAULT_PORT_SPAN = 2000

# `http METHOD URL [BODY_FILE]` prints the response body and fails on a
# non-2xx status: curl when the image has it, else bash's /dev/tcp (HTTP/1.0,
# so no chunked responses), which every image the runner supports has.
HTTP_FUNCTION = r"""http() {
  method=$1; url=$2; body=${3:-}
  if command -v curl >/dev/null 2>&1; then
    if [ -n "$body" ]; then
      curl -fsS -X "$method" -H 'Content-Type: application/json' --data-binary "@$body" "$url"
    else
      curl -fsS -X "$method" "$url"
    fi
    return
  fi
  rest=${url#http://}; hostport=${rest%%/*}; path=/${rest#*/}
  [ "$rest" = "$hostport" ] && path=/
  exec 3<>"/dev/tcp/${hostport%:*}/${hostport##*:}" || return 1
  if [ -n "$body" ]; then
    printf '%s %s HTTP/1.0\r\nHost: %s\r\nContent-Type: application/json\r\nContent-Length: %s\r\n\r\n' \
      "$method" "$path" "$hostport" "$(wc -c < "$body")" >&3
    cat "$body" >&3
  else
    printf '%s %s HTTP/1.0\r\nHost: %s\r\n\r\n' "$method" "$path" "$hostport" >&3
  fi
  IFS=' ' read -r _ code _ <&3
  while IFS= read -r line <&3; do line=${line%$'\r'}; [ -z "$line" ] && break; done
  cat <&3
  exec 3>&-
  case $code in 2*) return 0 ;; *) echo "HTTP $code from $method $url" >&2; return 1 ;; esac
}"""


class Mirror(Service):
    """A stand-in for a SaaS API, running in the agent container.

    On top of `Service`: a per-case `seed` loaded after start (uploaded to a
    temp file, loaded, deleted, all before the agent starts), a base `url` the
    app is pointed at through `env` templates (`{"SLACK_API_URL": "{url}"}`),
    and `state(ctx)` / `get(ctx, path)` for grading. Providers fill in
    `load_seed` and `read_state`.

    `snapshot=True` saves the state in `collect`, so `state(ctx)` also works in
    a separate verifier container.
    """

    display_name: ClassVar[str] = "Mirror"

    def __init__(
        self,
        name: str,
        version: str,
        *,
        seed: SeedSpec = None,
        env: Mapping[str, str] | None = None,
        port: int | None = None,
        snapshot: bool = False,
        source: str | None = None,
    ):
        super().__init__(name, version, env=env, source=source)
        self.seed_spec: SeedSpec = seed
        self.port: int = port if port is not None else default_port(name)
        self.snapshot_state: bool = snapshot

    @property
    def url(self) -> str:
        """The base URL in the container, `http://127.0.0.1:<port>`."""
        return f"http://{HOST}:{self.port}"

    @override
    def template_values(self) -> dict[str, str]:
        return {**super().template_values(), "port": str(self.port), "url": self.url}

    @override
    def ports(self) -> list[int]:
        return [self.port]

    # Provider hooks

    async def load_seed(self, ctx: AgentContext, path: str) -> None:
        """Load the seed at `path` (a temp file in the container) into the
        running mirror."""
        raise NotImplementedError

    async def read_state(self, ctx: Context) -> Any:
        """The mirror's state, read live."""
        raise NotImplementedError

    # Lifecycle

    @override
    async def seed(self, ctx: AgentContext) -> None:
        host = resolve_seed(self.seed_spec, ctx.case, self.base_dir, self.name)
        if host is None:
            return
        remote_dir = f"/tmp/.{uuid.uuid4().hex[:12]}"
        remote = f"{remote_dir}/{host.name}"
        try:
            await ctx.container.copy(host, remote)
            await self.load_seed(ctx, remote)
        finally:
            _ = await ctx.container.sh(f"rm -rf {remote_dir}", user="root", check=False)

    @override
    async def snapshot(self, ctx: AgentContext, out_dir: str) -> None:
        if not self.snapshot_state:
            return
        state = await self.read_state(ctx)
        quoted = shlex.quote(out_dir)
        _ = await ctx.container.sh(f"rm -rf {quoted} && mkdir -p {quoted}", user="root")
        await ctx.container.write_file(f"{out_dir}/{STATE_FILE}", json.dumps(state))
        _ = await ctx.container.sh(f"chmod -R a+rX {quoted}", user="root")

    # Grading

    async def state(self, ctx: Context) -> Any:
        """The mirror's state: live in the agent phase and a shared verifier;
        in a separate verifier, what `collect` saved (needs `snapshot=True`)."""
        if getattr(ctx, "verifier_mode", None) == "separate":
            return self.saved(ctx, STATE_FILE)
        return await self.read_state(ctx)

    def saved(self, ctx: Context, file: str) -> Any:
        """A file `snapshot` saved under `/logs/artifacts/state/<name>/`, read
        on the host through `ctx.artifacts` (verify only)."""
        if not self.snapshot_state:
            raise EvalConfigError(
                f"{self.name} isn't running in separate verifier mode: declare "
                + "snapshot=True on it to grade from its saved state."
            )
        artifacts = getattr(ctx, "artifacts", None)
        read_json = getattr(artifacts, "read_json", None)
        if read_json is None:
            raise EvalConfigError(f"{self.name}: saved state is read in verify().")
        return read_json(f"{STATE_DIR}/{self.name}/{file}")

    async def get(self, ctx: Context, path: str, **params: Any) -> Any:
        """GET `path` (with query `params`) from the running mirror; JSON is
        parsed, anything else comes back as text."""
        if getattr(ctx, "verifier_mode", None) == "separate":
            raise EvalConfigError(
                f"{self.name} isn't running in separate verifier mode; use "
                + f"self.mirror({self.name!r}).state(ctx) with snapshot=True."
            )
        return await self.request(ctx, "GET", path, params=params)

    async def request(
        self,
        ctx: Context,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        body: Any = None,
    ) -> Any:
        """An HTTP request to the mirror from inside the container."""
        query = _query(params or {})
        url = f"{self.url}{path if path.startswith('/') else '/' + path}{query}"
        lines = [HTTP_FUNCTION]
        if body is None:
            lines.append(f"http {shlex.quote(method)} {shlex.quote(url)}")
        else:
            body_file = f"/tmp/.{uuid.uuid4().hex[:12]}.json"
            await ctx.container.write_file(body_file, json.dumps(body))
            lines.append(
                f"http {shlex.quote(method)} {shlex.quote(url)} {body_file}; "
                + f"rc=$?; rm -f {body_file}; exit $rc"
            )
        out = await run_script(ctx, "\n".join(lines), what=f"{method} {url} ({self.name})")
        try:
            return json.loads(out)
        except ValueError:
            return out


def default_port(name: str) -> int:
    return DEFAULT_PORT_BASE + zlib.crc32(name.encode()) % DEFAULT_PORT_SPAN


def _query(params: Mapping[str, Any]) -> str:
    if not params:
        return ""
    return "?" + urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)

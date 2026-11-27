"""`Mirror`: a service that stands in for a SaaS API (Slack, Stripe, ...)."""

from __future__ import annotations

import json
import shlex
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlencode

from typing_extensions import override

from lmnr.evals.context import relay_http
from lmnr.evals.errors import OUTPUT_TAIL_CHARS, EvalConfigError
from lmnr.evals.fixtures import SeedSpec, resolve_seed
from lmnr.evals.services.base import HOST, STATE_DIR, Service

if TYPE_CHECKING:
    from lmnr.evals.context import AgentContext, Context

STATE_FILE = "state.json"
# Default ports are picked from the name in this range, so a runner's mirrors
# rarely collide; `port=` settles it when they do.
DEFAULT_PORT_BASE = 18000
DEFAULT_PORT_SPAN = 2000


class Mirror(Service):
    """A stand-in for a SaaS API, running in the agent container.

    On top of `Service`: a per-case `seed` loaded after start (before the
    agent starts; the provider sends it, nothing stays on disk), a base `url` the
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

    async def load_seed(self, ctx: AgentContext, path: Path) -> None:
        """Load the seed file at `path` (on the host) into the running mirror,
        e.g. `await self.request(ctx, "POST", ..., body=path.read_bytes())`."""
        raise NotImplementedError

    async def read_state(self, ctx: Context) -> Any:
        """The mirror's state, read live."""
        raise NotImplementedError

    # Lifecycle

    @override
    async def seed(self, ctx: AgentContext) -> None:
        host = resolve_seed(self.seed_spec, ctx.case, self.base_dir, self.name)
        if host is not None:
            await self.load_seed(ctx, host)

    @override
    async def snapshot(self, ctx: AgentContext, out_dir: str) -> None:
        if not self.snapshot_state:
            return
        files = {STATE_FILE: await self.read_state(ctx), **await self.snapshot_extras(ctx)}
        quoted = shlex.quote(out_dir)
        _ = await ctx.container.sh(f"rm -rf {quoted} && mkdir -p {quoted}", user="root")
        for name, content in files.items():
            await ctx.container.write_file(f"{out_dir}/{name}", json.dumps(content))
        # After every file: the upload's mode depends on the environment.
        _ = await ctx.container.sh(f"chmod -R a+rX {quoted}", user="root")

    async def snapshot_extras(self, ctx: AgentContext) -> dict[str, Any]:
        """More JSON files for the snapshot, by name, next to `state.json`
        (read back with `saved(ctx, name)`)."""
        del ctx
        return {}

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
        timeout: float = 60,
    ) -> Any:
        """An HTTP request to the mirror from inside the container. `body` is
        sent as JSON (bytes as they are); a JSON response is parsed, anything
        else comes back as text. A status of 400 or more raises."""
        query = _query(params or {})
        url = f"{self.url}{path if path.startswith('/') else '/' + path}{query}"
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
        response = await relay_http(
            ctx.container,
            method,
            url,
            body=data,
            headers={"Content-Type": "application/json"} if data is not None else None,
            timeout=timeout,
        )
        if response.status >= 400:
            raise RuntimeError(
                f"{method} {url} ({self.name}) answered {response.status}: "
                + response.text()[-OUTPUT_TAIL_CHARS:]
            )
        try:
            return response.json()
        except ValueError:
            return response.text()


def default_port(name: str) -> int:
    return DEFAULT_PORT_BASE + zlib.crc32(name.encode()) % DEFAULT_PORT_SPAN


def _query(params: Mapping[str, Any]) -> str:
    if not params:
        return ""
    return "?" + urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)

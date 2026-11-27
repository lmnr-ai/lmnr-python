"""`Valkey`: a Valkey (Redis-compatible) server in the agent container."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, ClassVar

from typing_extensions import override

from lmnr.evals.image import PathMap
from lmnr.evals.readiness import Readiness, cmd
from lmnr.evals.services.base import HOST, EnvSpec, Service, require_live
from lmnr.evals.services.bundle import BUNDLE_DIR, install_check, stage

if TYPE_CHECKING:
    from lmnr.evals.context import Context

DATA_DIR = "/var/lib/valkey"
LIB_DIR = "/usr/local/lib/valkey"

_BUNDLE = f"""mkdir -p "$out{LIB_DIR}/bin"
for b in valkey-server valkey-cli; do cp -L "$(command -v $b)" "$out{LIB_DIR}/bin/$b"; done
collect_deps {LIB_DIR}/libs "$out{LIB_DIR}"/bin/*
for b in valkey-server valkey-cli; do wrap {LIB_DIR}/bin/$b /usr/local/bin/$b {LIB_DIR}/libs; done
ln -s valkey-cli "$out/usr/local/bin/redis-cli"
ln -s valkey-server "$out/usr/local/bin/redis-server"
record_glibc {{name}}"""


class Valkey(Service):
    """A Valkey server, installed from `valkey/valkey:<version>-bookworm`.

        Valkey("8")

    In memory only (`--save '' --appendonly no`), on 127.0.0.1, as the
    `valkey` OS user; `redis-cli` and `redis-server` point at Valkey's. `env`:
    `"REDIS_URL"` (the default) gets `redis://127.0.0.1:<port>`; templates can
    use `{url}`, `{host}`, `{port}`.
    """

    display_name: ClassVar[str] = "Valkey"
    start_timeout: ClassVar[float] = 60

    def __init__(
        self,
        version: str,
        *,
        port: int = 6379,
        env: EnvSpec = "REDIS_URL",
        source: str | None = None,
        name: str = "valkey",
    ):
        super().__init__(name, version, env=env, source=source)
        self.port: int = port

    @override
    def source_image(self) -> str:
        if self.source:
            return self.source
        tag = "bookworm" if self.version == "latest" else f"{self.version}-bookworm"
        return f"valkey/valkey:{tag}"

    @override
    def bundle_stages(self, rel: PathMap) -> list[str]:
        return stage(self.source_image(), self.stage_name(), _BUNDLE.replace("{name}", self.name))

    @override
    def install_steps(self, rel: PathMap) -> list[str]:
        return [
            f"COPY --from={self.stage_name()} {BUNDLE_DIR}/ /",
            install_check(
                self.name, self.display_name, self.source_image(), [("valkey", DATA_DIR)]
            ),
        ]

    @override
    def start_command(self) -> str:
        return (
            f"mkdir -p {DATA_DIR} && chown valkey:valkey {DATA_DIR} && "
            + f"exec runuser -u valkey -- valkey-server --bind {HOST} --port {self.port} "
            + f"--save '' --appendonly no --dir {DATA_DIR}"
        )

    @override
    def ready(self) -> Readiness:
        return cmd(f'[ "$(valkey-cli -h {HOST} -p {self.port} ping 2>/dev/null)" = PONG ]')

    @override
    def template_values(self) -> dict[str, str]:
        return {
            **super().template_values(),
            "port": str(self.port),
            "url": f"redis://{HOST}:{self.port}",
        }

    @override
    def ports(self) -> list[int]:
        return [self.port]

    async def cmd(self, ctx: Context, *args: str) -> str:
        """Run a command (`await valkey.cmd(ctx, "GET", "key")`); returns its
        output (`valkey-cli --raw`)."""
        require_live(self, ctx)
        argv = " ".join(shlex.quote(a) for a in args)
        result = await ctx.container.sh(
            f"valkey-cli -h {HOST} -p {self.port} --raw {argv}", user="root", timeout=60
        )
        return result.stdout.rstrip("\n")

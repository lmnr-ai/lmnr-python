"""`Rystic`: Rystic's SaaS simulators ("twins") as mirrors.

Everything this adapter relies on about Rystic is in the constants and small
methods below, so it can be amended in one place; the user-facing signature
(`Rystic(service, version=..., seed=..., env=...)`) is the part we commit to.
Checked against rystic CLI 0.0.93 and lmnr's hosted-evals proxy (LAM-2592):

- the CLI is one static binary, installed by `INSTALLER_URL` at a pinned
  `RYSTIC_VERSION`; that needs no key, so the image build is key-free;
- a twin is downloaded per trial with `rystic pull <product> --version <v>`,
  which needs a key: `RYSTIC_TOKEN` (plus `RYSTIC_RELEASE_ENDPOINT` /
  `RYSTIC_LEASE_URL`) in the container's env. Laminar's remote sandboxes have
  them per trial; locally pass a key with `lmnr-cli eval run --env
  RYSTIC_TOKEN`. The running twin checks in with Rystic's lease server;
- `rystic run <product> --name <n> --port <p>` serves the vendor's API in the
  foreground (`-d` would detach);
- `GET /_rystic/routes` answers once it serves; `POST /_rystic/seed` takes a
  scenario (register map) and replaces the state; `GET /_rystic/inspect` is
  the state, `GET /_rystic/calls` every API call the app made, and
  `GET /_rystic/settle?quiet_ms=` returns once the app stopped reacting.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import override

from lmnr.evals.fixtures import SeedSpec
from lmnr.evals.image import PathMap
from lmnr.evals.mirrors.base import HTTP_FUNCTION, Mirror
from lmnr.evals.readiness import Readiness, cmd
from lmnr.evals.services.base import run_script
from lmnr.evals.services.bundle import BUNDLE_DIR

if TYPE_CHECKING:
    from lmnr.evals.context import AgentContext, Context

# The CLI, installed at build time.
DEFAULT_CLI_VERSION = "0.0.93"
INSTALLER_URL = "https://www.rystic.ai/install.sh"
# Any image with curl, tar and sh: the binary is static, so the stage's libc
# doesn't matter.
INSTALLER_IMAGE = "curlimages/curl:8.17.0"
BINARY = "/usr/local/bin/rystic"
PRODUCT_SUFFIX = "-twin"
PULL_COMMAND = "{binary} pull {product} --version {version}"
RUN_COMMAND = "{binary} run {product} --name {name} --port {port}"
READY_PATH = "/_rystic/routes"
SEED_PATH = "/_rystic/seed"
STATE_PATH = "/_rystic/inspect"
CALLS_PATH = "/_rystic/calls"
SETTLE_PATH = "/_rystic/settle"
CALLS_FILE = "calls.json"
# `rystic pull` downloads the twin.
PULL_TIMEOUT_SEC = 300


class Rystic(Mirror):
    """A Rystic simulator of `service` (`"slack"`, `"linear"`, `"github-pulls"`;
    the `-twin` product name works too) at the twin's `version`.

        Rystic("slack", version="1.4", seed=case_fixture("slack.json"),
               env={"SLACK_API_URL": "{url}/api/"})

    `seed` is a Rystic scenario (`{"<register>": {"<key>": {record}}}`, see
    `rystic docs slack-twin seeding-state`), loaded after start; `clock` (RFC
    3339) starts the twin's clock. In `verify`: `await
    self.mirror("slack").state(ctx)` (`/_rystic/inspect`), `.calls(ctx)`
    (`/_rystic/calls`) or `.get(ctx, "/api/conversations.history",
    channel="C1")`. With `snapshot=True`, `collect` saves the state and the
    calls, so both also work in a separate verifier.
    """

    display_name: ClassVar[str] = "Rystic"

    def __init__(
        self,
        service: str,
        *,
        version: str,
        seed: SeedSpec = None,
        env: Mapping[str, str] | None = None,
        port: int | None = None,
        snapshot: bool = False,
        clock: str | None = None,
        cli_version: str = DEFAULT_CLI_VERSION,
        name: str | None = None,
    ):
        product = service if service.endswith(PRODUCT_SUFFIX) else service + PRODUCT_SUFFIX
        super().__init__(
            name or product.removesuffix(PRODUCT_SUFFIX),
            version,
            seed=seed,
            env=env,
            port=port,
            snapshot=snapshot,
        )
        self.product: str = product
        self.clock: str | None = clock
        self.cli_version: str = cli_version

    @override
    def image_title(self, root: Path | None) -> str:
        where = f" ({self.site.label(root, owner=False)})" if self.site else ""
        return f"service: Rystic {self.product} {self.version}, CLI {self.cli_version}{where}"

    @override
    def source_image(self) -> str:
        return f"{INSTALLER_URL} (RYSTIC_VERSION={self.cli_version})"

    # Build: the CLI only. Twins need a key, so each trial pulls its own.

    @override
    def bundle_stages(self, rel: PathMap) -> list[str]:
        bin_dir = f"{BUNDLE_DIR}{Path(BINARY).parent}"
        return [
            f"FROM {INSTALLER_IMAGE} AS {self.stage_name()}",
            "USER root",
            # Saved first: `curl | sh` would pass on a failed download.
            f"RUN curl -fsSL {INSTALLER_URL} -o /tmp/rystic-install.sh && mkdir -p {bin_dir} && "
            + f"RYSTIC_VERSION={shlex.quote(self.cli_version)} RYSTIC_BIN={bin_dir} "
            + "RYSTIC_NO_LOGIN=1 CI=1 sh /tmp/rystic-install.sh && "
            + f"{BUNDLE_DIR}{BINARY} --help >/dev/null",
        ]

    @override
    def install_steps(self, rel: PathMap) -> list[str]:
        return [f"COPY --from={self.stage_name()} {BUNDLE_DIR}/ /"]

    # Run

    @override
    async def start(self, ctx: AgentContext) -> None:
        _ = await ctx.container.sh(
            PULL_COMMAND.format(
                binary=BINARY,
                product=shlex.quote(self.product),
                version=shlex.quote(self.version),
            ),
            user="root",
            timeout=PULL_TIMEOUT_SEC,
        )
        await super().start(ctx)

    @override
    def start_command(self) -> str:
        command = RUN_COMMAND.format(
            binary=BINARY,
            product=shlex.quote(self.product),
            name=shlex.quote(self.name),
            port=self.port,
        )
        if self.clock:
            command += f" --clock {shlex.quote(self.clock)}"
        return command

    @override
    def ready(self) -> Readiness:
        return cmd(f"{HTTP_FUNCTION}\nhttp GET {self.url}{READY_PATH} >/dev/null")

    @override
    async def load_seed(self, ctx: AgentContext, path: str) -> None:
        _ = await run_script(
            ctx,
            f"{HTTP_FUNCTION}\nhttp POST {self.url}{SEED_PATH} {shlex.quote(path)} >/dev/null",
            what=f"Seeding {self.name}",
        )

    @override
    async def read_state(self, ctx: Context) -> Any:
        return await self.request(ctx, "GET", STATE_PATH)

    @override
    async def snapshot(self, ctx: AgentContext, out_dir: str) -> None:
        await super().snapshot(ctx, out_dir)
        if self.snapshot_state:
            calls = await self.request(ctx, "GET", CALLS_PATH)
            await ctx.container.write_file(f"{out_dir}/{CALLS_FILE}", json.dumps(calls))

    # Grading

    async def calls(self, ctx: Context) -> Any:
        """Every API call the app made (`/_rystic/calls`); in a separate
        verifier, what `collect` saved (`snapshot=True`)."""
        if getattr(ctx, "verifier_mode", None) == "separate":
            return self.saved(ctx, CALLS_FILE)
        return await self.request(ctx, "GET", CALLS_PATH)

    async def settle(self, ctx: Context, quiet_ms: int = 2000) -> None:
        """Wait until the app stopped reacting for `quiet_ms` (e.g. a bot's
        turn after an event)."""
        _ = await self.request(ctx, "GET", SETTLE_PATH, params={"quiet_ms": quiet_ms})

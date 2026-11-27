"""`Rystic`: Rystic's SaaS simulators ("twins") as mirrors.

Everything this adapter relies on about Rystic is in the constants and small
methods below, so it can be amended in one place; the user-facing signature
(`Rystic(service, version=..., seed=..., env=...)`) is the part we commit to.
Checked against rystic CLI 0.0.93 and lmnr's hosted-evals proxy (LAM-2592):

- the CLI is one static binary, installed by `INSTALLER_URL` at a pinned
  `RYSTIC_VERSION`; that needs no key, so the image build is key-free. It is
  a Go binary that reads the system's CA certificates, which slim images
  (`node:*-slim`, `debian:*-slim`) don't have, so it ships with its own and
  `BINARY` is a wrapper that uses them when the image has none;
- a twin is downloaded per trial with `rystic pull <product> --version <v>`
  (into `$RYSTIC_HOME/twins/<product>/<v>`, marked current for `run`), which
  needs a key. Laminar's remote sandboxes give every trial its own:
  `RYSTIC_TOKEN`, `RYSTIC_RELEASE_ENDPOINT` and `RYSTIC_LEASE_URL` in the
  sandbox env, pointing at the app-server's Rystic proxy (lmnr-private
  `hosted_evals/rystic.rs`). Local runs use the user's own key, passed with
  `lmnr-cli eval run --env RYSTIC_TOKEN`, against rystic.ai. A trial without
  a token fails before the pull, saying so;
- `rystic run <product> --name <n> --port <p>` serves the vendor's API in the
  foreground (`-d` would detach), starting from an empty workspace (no
  `--scenario`); the twin checks in with the lease URL while it runs;
- control plane (`rystic help control-plane`; every route used here is on
  every plan): `GET /_rystic/routes` answers once it serves; `POST
  /_rystic/seed` merges a register map (or lands a scenario document) into
  the state and answers 204; `GET /_rystic/inspect` is the state,
  `GET /_rystic/calls` every vendor API call since reset (`{"calls": [...]}`),
  and `GET /_rystic/settle?quiet_ms=` long-polls until the app stopped
  reacting (`{"settled": bool, ...}`, unsettled after `timeout_ms`).
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import override

from lmnr.evals.fixtures import SeedSpec
from lmnr.evals.image import PathMap
from lmnr.evals.mirrors.base import Mirror
from lmnr.evals.readiness import Readiness, http
from lmnr.evals.services.bundle import BUNDLE_DIR

if TYPE_CHECKING:
    from lmnr.evals.context import AgentContext, Context

# The CLI, installed at build time.
DEFAULT_CLI_VERSION = "0.0.93"
INSTALLER_URL = "https://www.rystic.ai/install.sh"
# Any image with curl, tar and sh: the binary is static, so the stage's libc
# doesn't matter.
INSTALLER_IMAGE = "curlimages/curl:8.17.0"
BINARY = "/usr/local/bin/rystic"  # the wrapper, below
LIB_DIR = "/usr/local/lib/rystic"  # the real binary and its CA certificates
# Used only when the image has no CA bundle of its own (Debian / Ubuntu, RHEL).
WRAPPER = (
    "#!/bin/sh",
    'if [ -z "${SSL_CERT_FILE:-}" ] && [ ! -s /etc/ssl/certs/ca-certificates.crt ] '
    + "&& [ ! -s /etc/pki/tls/certs/ca-bundle.crt ]; then",
    f"  export SSL_CERT_FILE={LIB_DIR}/ca-certificates.crt",
    "fi",
    f'exec {LIB_DIR}/rystic "$@"',
)
PRODUCT_SUFFIX = "-twin"
PULL_COMMAND = "{binary} pull {product} --version {version}"
RUN_COMMAND = "{binary} run {product} --name {name} --port {port}"
READY_PATH = "/_rystic/routes"
SEED_PATH = "/_rystic/seed"
STATE_PATH = "/_rystic/inspect"
CALLS_PATH = "/_rystic/calls"
SETTLE_PATH = "/_rystic/settle"
# The snapshot file `calls()` goes to (`ctx.state(<mirror>, "calls")`).
CALLS = "calls"
# `settle` long-polls; Rystic answers unsettled after its own 30 s default.
SETTLE_TIMEOUT_SEC = 90
# `rystic pull` downloads the twin.
PULL_TIMEOUT_SEC = 300
# Whether the trial has Rystic access: Laminar's per-trial token, or the user's key.
ACCESS_CHECK = 'if [ -n "${RYSTIC_TOKEN:-}" ]; then echo yes; fi'
NO_ACCESS = (
    "Rystic mirrors need Rystic access in local (not `--remote`) runs. Pass your Rystic API "
    + "key with `--env RYSTIC_TOKEN` (read from your shell, .env.local or .env)."
)


class Rystic(Mirror):
    """A Rystic simulator of `service` (`"slack"`, `"linear"`, `"github-pulls"`;
    the `-twin` product name works too) at the twin's `version`.

        Rystic("slack", version="1.4", seed=case_fixture("slack.json"),
               env={"SLACK_API_URL": "{url}/api/"})

    Local (not `--remote`) runs need your Rystic API key
    (`--env RYSTIC_TOKEN`).

    `seed` is a register map (`{"<register>": {"<key>": {record}}}`, see
    `rystic docs slack-twin seeding-state`), merged into the twin's empty
    workspace after start; `clock` (RFC
    3339) starts the twin's clock. In `verify`: `await
    self.mirror("slack").state(ctx)` (`/_rystic/inspect`), `.calls(ctx)`
    (`/_rystic/calls`) or `.get(ctx, "/api/conversations.history",
    channel="C1")`. With `snapshot=True`, `collect` saves the state and the
    calls, so both also work in a separate verifier (also as
    `ctx.state("slack", "state")` / `ctx.state("slack", "calls")`).
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
        lib_dir = f"{BUNDLE_DIR}{LIB_DIR}"
        wrapper = f"{BUNDLE_DIR}{BINARY}"
        lines = " ".join(shlex.quote(line) for line in WRAPPER)
        return [
            f"FROM {INSTALLER_IMAGE} AS {self.stage_name()}",
            "USER root",
            # Saved first: `curl | sh` would pass on a failed download.
            f"RUN curl -fsSL {INSTALLER_URL} -o /tmp/rystic-install.sh && "
            + f"mkdir -p {lib_dir} {Path(wrapper).parent} && "
            + f"RYSTIC_VERSION={shlex.quote(self.cli_version)} RYSTIC_BIN={lib_dir} "
            + "RYSTIC_NO_LOGIN=1 CI=1 sh /tmp/rystic-install.sh && "
            + f"cp /etc/ssl/certs/ca-certificates.crt {lib_dir}/ && "
            + f"printf '%s\\n' {lines} > {wrapper} && chmod 755 {wrapper} && "
            + f"{lib_dir}/rystic --help >/dev/null",
        ]

    @override
    def install_steps(self, rel: PathMap) -> list[str]:
        return [f"COPY --from={self.stage_name()} {BUNDLE_DIR}/ /"]

    # Run

    @override
    async def start(self, ctx: AgentContext) -> None:
        access = (await ctx.container.sh(ACCESS_CHECK, user="root", check=False)).stdout.strip()
        if access != "yes":
            raise RuntimeError(f"{NO_ACCESS} This trial has no RYSTIC_TOKEN ({self.name}).")
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
        return http(f"{self.url}{READY_PATH}")

    @override
    async def load_seed(self, ctx: AgentContext, path: Path) -> None:
        _ = await self.request(ctx, "POST", SEED_PATH, body=path.read_bytes())

    @override
    async def read_state(self, ctx: Context) -> Any:
        return await self.request(ctx, "GET", STATE_PATH)

    @override
    async def snapshot_extras(self, ctx: AgentContext) -> dict[str, Any]:
        return {CALLS: await self._live_calls(ctx)}

    # Grading

    async def calls(self, ctx: Context) -> Any:
        """Every vendor API call the app made, oldest first (`/_rystic/calls`),
        each with its `summary` and `writes` (the records it created, changed
        or deleted); in a separate verifier, what `collect` saved
        (`snapshot=True`)."""
        if getattr(ctx, "verifier_mode", None) == "separate":
            return self._saved(ctx, CALLS)
        return await self._live_calls(ctx)

    async def _live_calls(self, ctx: Context) -> Any:
        log = await self.request(ctx, "GET", CALLS_PATH, params={"summary": 1, "writes": 1})
        return log.get("calls", log) if isinstance(log, dict) else log

    async def settle(self, ctx: Context, quiet_ms: int = 2000) -> Any:
        """Wait until the app stopped reacting for `quiet_ms` (e.g. a bot's
        turn after an event). Returns Rystic's answer: `settled` is false
        when it timed out, with what is still `pending`."""
        return await self.request(
            ctx, "GET", SETTLE_PATH, params={"quiet_ms": quiet_ms}, timeout=SETTLE_TIMEOUT_SEC
        )

"""`ClickHouse`: a ClickHouse server in the agent container."""

from __future__ import annotations

import json
import re
import shlex
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import override

from lmnr.evals.errors import EvalConfigError
from lmnr.evals.fixtures import SeedSpec, resolve_seed
from lmnr.evals.image import PathMap
from lmnr.evals.readiness import Readiness, cmd
from lmnr.evals.services.base import HOST, EnvSpec, Service, heredoc, require_live, run_script
from lmnr.evals.services.bundle import BUNDLE_DIR, GLIBC_MARKER, install_check, stage

if TYPE_CHECKING:
    from lmnr.evals.context import AgentContext, Context

DATA_DIR = "/var/lib/clickhouse"
CONFIG_DIR = "/etc/clickhouse-server"
# The binary is static apart from glibc, which it needs in an old version only.
GLIBC_REQUIRED = "2.17"
# Numbers come back as JSON numbers, not strings.
JSON_SETTINGS = "--output_format_json_quote_64bit_integers=0"

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_CONFIG = """<clickhouse>
  <logger><level>warning</level><console>1</console></logger>
  <listen_host>{host}</listen_host>
  <http_port>{http_port}</http_port>
  <tcp_port>{tcp_port}</tcp_port>
  <path>{data}/</path>
  <tmp_path>{data}/tmp/</tmp_path>
  <user_files_path>{data}/user_files/</user_files_path>
  <format_schema_path>{data}/format_schemas/</format_schema_path>
  <users_config>users.xml</users_config>
  <mark_cache_size>268435456</mark_cache_size>
  <mlock_executable>false</mlock_executable>
</clickhouse>"""

_USERS = """<clickhouse>
  <profiles><default/></profiles>
  <quotas><default/></quotas>
  <users>
    <default>
      <password></password>
      <networks><ip>127.0.0.1</ip><ip>::1</ip></networks>
      <profile>default</profile>
      <quota>default</quota>
      <access_management>1</access_management>
    </default>
  </users>
</clickhouse>"""


class ClickHouse(Service):
    """A ClickHouse server, installed from `clickhouse/clickhouse-server:<version>`.

        ClickHouse("25.8", database="analytics", migrations="../../../clickhouse")

    It runs as the `clickhouse` OS user on 127.0.0.1 (HTTP `http_port`, native
    `tcp_port`), user `default` without a password. Every trial gets a fresh
    server with `database` created, `migrations` (`*.sql` in natural order,
    through `clickhouse client --multiquery`) and this case's `seed` (`.sql`).

    - `env`: `"CLICKHOUSE_URL"` (the default) gets `http://127.0.0.1:<http_port>`;
      templates can use `{url}`, `{host}`, `{port}` (HTTP), `{tcp_port}`,
      `{database}`, `{user}`, `{password}`.
    - `snapshot=["events"]`: `collect` saves those tables as JSON for
      `ctx.state("clickhouse", "events")`.
    """

    display_name: ClassVar[str] = "ClickHouse"

    def __init__(
        self,
        version: str,
        *,
        database: str = "default",
        http_port: int = 8123,
        tcp_port: int = 9000,
        migrations: str | None = None,
        seed: SeedSpec = None,
        env: EnvSpec = "CLICKHOUSE_URL",
        snapshot: Sequence[str] = (),
        source: str | None = None,
        name: str = "clickhouse",
    ):
        super().__init__(name, version, env=env, source=source)
        self.database: str = database
        self.http_port: int = http_port
        self.tcp_port: int = tcp_port
        self.migrations: str | None = migrations
        self.seed_spec: SeedSpec = seed
        self.snapshot_tables: list[str] = list(snapshot)

    # Build

    @override
    def source_image(self) -> str:
        return self.source or f"clickhouse/clickhouse-server:{self.version}"

    @override
    def bundle_stages(self, rel: PathMap) -> list[str]:
        config = _CONFIG.format(
            host=HOST, http_port=self.http_port, tcp_port=self.tcp_port, data=DATA_DIR
        )
        script = f"""bin=$(command -v clickhouse || echo /usr/bin/clickhouse)
mkdir -p "$out/usr/bin" "$out{CONFIG_DIR}" "$out/usr/local/share"
cp -L "$bin" "$out/usr/bin/clickhouse"
for n in server client local; do ln -s clickhouse "$out/usr/bin/clickhouse-$n"; done
cat > "$out{CONFIG_DIR}/config.xml" <<'XML'
{config}
XML
cat > "$out{CONFIG_DIR}/users.xml" <<'XML'
{_USERS}
XML
echo {GLIBC_REQUIRED} > "$out{GLIBC_MARKER}{self.name}\""""
        return stage(self.source_image(), self.stage_name(), script)

    @override
    def install_steps(self, rel: PathMap) -> list[str]:
        return [
            f"COPY --from={self.stage_name()} {BUNDLE_DIR}/ /",
            install_check(
                self.name, self.display_name, self.source_image(), [("clickhouse", DATA_DIR)]
            ),
        ]

    # Run

    @override
    def start_command(self) -> str:
        return (
            f"mkdir -p {DATA_DIR} && chown clickhouse:clickhouse {DATA_DIR} && "
            + "exec runuser -u clickhouse -- clickhouse server "
            + f"--config-file={CONFIG_DIR}/config.xml"
        )

    @override
    def ready(self) -> Readiness:
        return cmd(f"{self._client()} -q 'SELECT 1'")

    @override
    async def start(self, ctx: AgentContext) -> None:
        await super().start(ctx)
        if self.database != "default":
            _ = await ctx.container.sh(
                f"{self._client()} -q {shlex.quote(f'CREATE DATABASE IF NOT EXISTS {self.database}')}",
                user="root",
                timeout=60,
            )

    @override
    async def migrate(self, ctx: AgentContext) -> None:
        if self.migrations is None:
            return
        migrations = self.migrations_dir()
        assert migrations is not None
        remote = f"/tmp/.{uuid.uuid4().hex[:12]}"
        try:
            await ctx.container.copy(migrations, remote)
            _ = await run_script(
                ctx,
                f"find {remote} -maxdepth 1 -type f -name '*.sql' | sort -V | "
                + "while IFS= read -r f; do "
                + f'{self._client(database=True)} --multiquery < "$f" '
                + '|| { echo "migration failed: ${f##*/}" >&2; exit 1; }; done',
                what=f"Migrations from {migrations} ({self.name})",
                timeout=600,
            )
        finally:
            _ = await ctx.container.sh(f"rm -rf {remote}", user="root", check=False)

    @override
    async def seed(self, ctx: AgentContext) -> None:
        path = resolve_seed(self.seed_spec, ctx.case, self.base_dir, self.name)
        if path is not None:
            _ = await self._client_stdin(ctx, "--multiquery", path.read_text(), what="seed")

    @override
    def template_values(self) -> dict[str, str]:
        return {
            **super().template_values(),
            "port": str(self.http_port),
            "tcp_port": str(self.tcp_port),
            "database": self.database,
            "user": "default",
            "password": "",
            "url": f"http://{HOST}:{self.http_port}",
        }

    @override
    async def snapshot(self, ctx: AgentContext, out_dir: str) -> None:
        if not self.snapshot_tables:
            return
        quoted = shlex.quote(out_dir)
        lines = ["set -e", f"rm -rf {quoted}", f"mkdir -p {quoted}"]
        for table in self.snapshot_tables:
            query = f"SELECT * FROM {table} FORMAT JSONEachRow"
            target = shlex.quote(f"{out_dir}/{table}.json")
            # JSONEachRow lines -> one JSON array.
            lines.append(
                f"{{ echo '['; {self._client(database=True)} {JSON_SETTINGS} "
                + f"-q {shlex.quote(query)} | sed '$!s/$/,/'; echo ']'; }} > {target}"
            )
        lines.append(f"chmod -R a+rX {quoted}")
        _ = await run_script(ctx, "\n".join(lines), what=f"Snapshot of {self.name}")

    # Helpers

    async def query(self, ctx: Context, sql: str) -> list[dict[str, Any]]:
        """Run a query and return its rows."""
        require_live(self, ctx)
        out = await self._client_stdin(
            ctx, f"--format JSONEachRow {JSON_SETTINGS}", sql, what="query"
        )
        return [json.loads(line) for line in out.splitlines() if line.strip()]

    async def execute(self, ctx: Context, sql: str) -> None:
        """Run statements."""
        require_live(self, ctx)
        _ = await self._client_stdin(ctx, "--multiquery", sql, what="execute")

    async def insert(self, ctx: Context, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Insert rows (dicts by column) as JSONEachRow."""
        require_live(self, ctx)
        if not rows:
            return
        payload = "\n".join(json.dumps(row, default=str) for row in rows)
        query = f"INSERT INTO {table} FORMAT JSONEachRow"
        _ = await self._client_stdin(
            ctx, f"-q {shlex.quote(query)}", payload, what=f"insert into {table}"
        )

    # Internals

    def migrations_dir(self) -> Path | None:
        if self.migrations is None:
            return None
        return ((self.base_dir or Path.cwd()) / self.migrations).resolve()

    def _client(self, database: bool = False) -> str:
        db = f" --database {shlex.quote(self.database)}" if database else ""
        return f"clickhouse client --host {HOST} --port {self.tcp_port}{db}"

    async def _client_stdin(self, ctx: Context, args: str, text: str, *, what: str) -> str:
        script = heredoc(f"{self._client(database=True)} {args}", text)
        return await run_script(ctx, script, what=f"{self.name}: {what}")

    # Configuration

    @override
    def ports(self) -> list[int]:
        return [self.http_port, self.tcp_port]

    @override
    def validate(self, owner: str) -> list[str]:
        warnings = super().validate(owner)
        where = f"{owner}.services[{self.name!r}]"
        if not _IDENT.match(self.database):
            raise EvalConfigError(
                f"{where}: database {self.database!r} must be a plain identifier."
            )
        migrations = self.migrations_dir()
        if migrations is not None and not migrations.is_dir():
            raise EvalConfigError(f"{where}: migrations folder {migrations} does not exist.")
        return warnings

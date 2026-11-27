"""`ClickHouse`: a ClickHouse server in the agent container."""

from __future__ import annotations

import json
import re
import shlex
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from typing_extensions import override

from lmnr.evals.context import relay
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
# The default migrator: https://www.npmjs.com/package/clickhouse-migrations, a
# Node CLI bundled with its own `node` from `MIGRATIONS_NODE_IMAGE`, so the
# app's image needs no Node. trixie's libstdc++ needs glibc 2.38 (Ubuntu 24.04
# and newer); bookworm apps use migrator="clickhouse-client".
MIGRATIONS_TOOL = "clickhouse-migrations"
MIGRATIONS_TOOL_VERSION = "1.6.0"
MIGRATIONS_NODE_IMAGE = "node:24-trixie-slim"
MIGRATIONS_TOOL_DIR = "/usr/lib/clickhouse-migrations"
MIGRATIONS_TOOL_BIN = f"{MIGRATIONS_TOOL_DIR}/bin/clickhouse-migrations"
# Its file names: the version number, `_`, anything (`1_init.sql`).
_TOOL_FILE = re.compile(r"^0*[1-9][0-9]*_.*\.sql$")
# Override files ClickHouse merges over the generated config / users.
_OVERRIDE_SUFFIXES = (".xml", ".yaml", ".yml")

Migrator = Literal["clickhouse-migrations", "clickhouse-client"]

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
    server with `database` created, `migrations` (see `migrator`) and this
    case's `seed` (`.sql`).

    - `env`: `"CLICKHOUSE_URL"` (the default) gets `http://127.0.0.1:<http_port>`;
      templates can use `{url}`, `{host}`, `{port}` (HTTP), `{tcp_port}`,
      `{database}`, `{user}`, `{password}`.
    - `migrator`: `"clickhouse-migrations"` (the default) applies
      `migrations` with the `clickhouse-migrations` CLI (files named
      `1_init.sql`, `2_events.sql`, ...; applied versions are recorded in a
      `_migrations` table), as an app that uses it does in production.
      `"clickhouse-client"` pipes every `*.sql` file, in natural order, through
      `clickhouse client --multiquery`, with no bookkeeping.
    - `config="clickhouse/config.xml"` / `users="clickhouse/users.xml"`: files
      (`.xml` or `.yaml`) ClickHouse merges over the generated `config.xml` /
      `users.xml` (`config.d/`, `users.d/`), so what they set wins. Keep the
      ports in step with `http_port` / `tcp_port`, and the `default` user
      without a password (the runner connects as it).
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
        migrator: Migrator = "clickhouse-migrations",
        config: str | None = None,
        users: str | None = None,
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
        self.migrator: Migrator = migrator
        self.config: str | None = config
        self.users: str | None = users
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
        stages = stage(self.source_image(), self.stage_name(), script)
        if self._uses_tool():
            stages += ["", *self._tool_stage()]
        return stages

    def _tool_stage(self) -> list[str]:
        """`clickhouse-migrations` with its own `node` and libraries, run
        through a wrapper outside PATH, so it never shadows the app's Node."""
        tool = MIGRATIONS_TOOL_DIR
        script = f"""npm install --prefix {tool} --omit=dev --no-fund --no-audit --no-update-notifier \\
  {MIGRATIONS_TOOL}@{MIGRATIONS_TOOL_VERSION}
mkdir -p "$out{tool}"
cp -a {tool}/node_modules "$out{tool}/"
cp -L "$(command -v node)" "$out{tool}/node"
collect_deps {tool}/libs "$(command -v node)"
wrap "{tool}/node {tool}/node_modules/{MIGRATIONS_TOOL}/lib/cli.js" {MIGRATIONS_TOOL_BIN} {tool}/libs
"$out{tool}/node" {tool}/node_modules/{MIGRATIONS_TOOL}/lib/cli.js --help >/dev/null
record_glibc_needed {self.name}-migrations {tool}/node {tool}/libs"""
        return stage(MIGRATIONS_NODE_IMAGE, self.stage_name("-migrations"), script)

    @override
    def install_steps(self, rel: PathMap) -> list[str]:
        steps = [
            f"COPY --from={self.stage_name()} {BUNDLE_DIR}/ /",
            install_check(
                self.name, self.display_name, self.source_image(), [("clickhouse", DATA_DIR)]
            ),
        ]
        if self._uses_tool():
            steps += [
                f"COPY --from={self.stage_name('-migrations')} {BUNDLE_DIR}/ /",
                install_check(
                    f"{self.name}-migrations",
                    MIGRATIONS_TOOL,
                    MIGRATIONS_NODE_IMAGE,
                    [],
                    alternative='pass migrator="clickhouse-client"',
                ),
            ]
        for path, folder in self._overrides():
            steps.append(f"COPY {json.dumps([rel(path), f'{CONFIG_DIR}/{folder}/{path.name}'])}")
        return steps

    @override
    def host_paths(self) -> list[Path]:
        return [path for path, _ in self._overrides()]

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
        if self._uses_tool():
            command = (
                f"{MIGRATIONS_TOOL_BIN} migrate --host=http://{HOST}:{self.http_port} "
                + f"--user=default --password= --db={self.database} --migrations-home={remote}"
            )
        else:
            command = (
                f"find {remote} -maxdepth 1 -type f -name '*.sql' | sort -V | "
                + "while IFS= read -r f; do "
                + f'{self._client(database=True)} --multiquery < "$f" '
                + '|| { echo "migration failed: ${f##*/}" >&2; exit 1; }; done'
            )
        try:
            await ctx.container.copy(migrations, remote)
            _ = await run_script(
                ctx,
                command,
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
        _ = await ctx.container.sh(f"rm -rf {quoted} && mkdir -p {quoted}", user="root", timeout=60)
        # JSONEachRow lines -> one JSON array per table; a failed query fails
        # the snapshot instead of saving `[]`.
        jobs = [
            {
                "command": f"{self._client(database=True)} {JSON_SETTINGS} "
                + f"-q {shlex.quote(f'SELECT * FROM {table} FORMAT JSONEachRow')}",
                "out": f"{out_dir}/{table}.json",
            }
            for table in self.snapshot_tables
        ]
        _ = await relay(ctx.container, "rows", jobs, timeout=600, what=f"Snapshot of {self.name}")
        _ = await ctx.container.sh(f"chmod -R a+rX {quoted}", user="root", timeout=60)

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

    def _uses_tool(self) -> bool:
        return self.migrations is not None and self.migrator == "clickhouse-migrations"

    def _overrides(self) -> list[tuple[Path, str]]:
        """`(host file, folder under CONFIG_DIR)` for `config` and `users`."""
        base = self.base_dir or Path.cwd()
        found: list[tuple[Path, str]] = []
        for value, folder in ((self.config, "config.d"), (self.users, "users.d")):
            if value is not None:
                found.append(((base / value).resolve(), folder))
        return found

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
        if self.migrator not in ("clickhouse-migrations", "clickhouse-client"):
            raise EvalConfigError(
                f'{where}: migrator must be "clickhouse-migrations" or "clickhouse-client", '
                + f"got {self.migrator!r}."
            )
        migrations = self.migrations_dir()
        if migrations is not None and not migrations.is_dir():
            raise EvalConfigError(f"{where}: migrations folder {migrations} does not exist.")
        if migrations is not None and self._uses_tool():
            misnamed = sorted(
                p.name for p in migrations.glob("*.sql") if not _TOOL_FILE.match(p.name)
            )
            if misnamed:
                raise EvalConfigError(
                    f"{where}: clickhouse-migrations needs file names that start with a version "
                    + f"number and '_' (1_init.sql); {migrations} has {', '.join(misnamed)}. "
                    + 'Rename them, or pass migrator="clickhouse-client" to apply every *.sql '
                    + "in natural order."
                )
        for path, folder in self._overrides():
            what = "config" if folder == "config.d" else "users"
            if not path.is_file():
                raise EvalConfigError(f"{where}: {what} file {path} does not exist.")
            if path.suffix not in _OVERRIDE_SUFFIXES:
                raise EvalConfigError(
                    f"{where}: {what} file {path.name} must be .xml or .yaml (ClickHouse "
                    + "only reads those from config.d / users.d)."
                )
        return warnings

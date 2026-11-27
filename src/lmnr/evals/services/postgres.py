"""`Postgres`: a PostgreSQL server in the agent container."""

from __future__ import annotations

import json
import re
import shlex
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import quote

from typing_extensions import override

from lmnr.evals.errors import EvalConfigError
from lmnr.evals.fixtures import SeedSpec, resolve_seed
from lmnr.evals.image import PathMap
from lmnr.evals.readiness import Readiness, cmd
from lmnr.evals.services.base import HOST, EnvSpec, Service, heredoc, require_live, run_script
from lmnr.evals.services.bundle import BUNDLE_DIR, install_check, stage

if TYPE_CHECKING:
    from lmnr.evals.context import AgentContext, Context

DATA_DIR = "/var/lib/postgresql/data"
SOCKET_DIR = "/var/run/postgresql"
# Ephemeral data: durability settings off, so trials don't wait on fsync.
SERVER_SETTINGS = (
    "listen_addresses={host}",
    "port={port}",
    "fsync=off",
    "synchronous_commit=off",
    "full_page_writes=off",
    "jit=off",
)

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")

# Copies the newest /usr/lib/postgresql/<major> (binaries, extension libraries,
# share files) plus their libraries, and wraps each binary in /usr/local/bin.
_BUNDLE = """pgbin=$(ls -d /usr/lib/postgresql/*/bin | sort -V | tail -n 1)
pgroot=${pgbin%/bin}
major=${pgroot##*/}
mkdir -p "$out$pgroot" "$out/usr/share/postgresql"
cp -a "$pgroot/bin" "$pgroot/lib" "$out$pgroot/"
# JIT pulls in LLVM (100+ MB) and is turned off.
rm -rf "$out$pgroot"/lib/llvmjit* "$out$pgroot/lib/bitcode"
# Some files there are symlinks into /usr/share/postgresql (the config samples).
cp -rL "/usr/share/postgresql/$major" "$out/usr/share/postgresql/"
rm -rf "$out/usr/share/postgresql/$major/man"
collect_deps "$pgroot/libs" "$out$pgroot"/bin/* "$out$pgroot"/lib/*.so
for b in "$pgroot"/bin/*; do wrap "$b" "/usr/local/bin/${b##*/}" "$pgroot/libs"; done
# Time zone data, for images without tzdata (debian:trixie-slim).
mkdir -p "$out/usr/share/zoneinfo-postgres"
cp -a /usr/share/zoneinfo/. "$out/usr/share/zoneinfo-postgres/"
record_glibc {name}"""


class Postgres(Service):
    """A PostgreSQL server, installed from the official image at `version`.

        Postgres("16", database="app", migrations="../../../migrations")

    It runs as the `postgres` OS user on 127.0.0.1 with trust auth (the
    password only goes into the URL). Before `setup`, every trial gets a fresh
    server with `database` created, `extensions` and `init_sql` applied, then
    `migrations` (`*.sql` in natural order, subfolders skipped) or the
    `migrate` command, then this case's `seed` (a `.sql` file).

    - `migrate_at="build"` (SQL migrations only) runs them while the image is
      built, so trials start from a ready data directory.
    - `env`: variables to templates over `{url}`, `{host}`, `{port}`,
      `{database}`, `{user}`, `{password}`; default `{"DATABASE_URL": "{url}"}`.
    - `snapshot=["refunds"]`: `collect` saves those tables as JSON for
      `ctx.state("postgres", "refunds")`, which works in a separate verifier.
    - `source="pgvector/pgvector:pg16-bookworm"`: install from another image
      with the official layout (e.g. for the `vector` extension).
    """

    display_name: ClassVar[str] = "Postgres"

    def __init__(
        self,
        version: str,
        *,
        database: str = "app",
        user: str = "postgres",
        password: str = "postgres",
        port: int = 5432,
        migrations: str | None = None,
        migrate: str | None = None,
        migrate_at: Literal["setup", "build"] = "setup",
        extensions: Sequence[str] = (),
        init_sql: Sequence[str] = (),
        seed: SeedSpec = None,
        env: EnvSpec = None,
        snapshot: Sequence[str] = (),
        source: str | None = None,
        name: str = "postgres",
    ):
        if env is None:
            env = {"DATABASE_URL": "{url}"}
        super().__init__(name, version, env=env, source=source)
        self.database: str = database
        self.user: str = user
        self.password: str = password
        self.port: int = port
        self.migrations: str | None = migrations
        self.migrate_command: str | None = migrate
        self.migrate_at: str = migrate_at
        self.extensions: list[str] = list(extensions)
        self.init_sql: list[str] = list(init_sql)
        self.seed_spec: SeedSpec = seed
        self.snapshot_tables: list[str] = list(snapshot)

    # Build

    @override
    def source_image(self) -> str:
        if self.source:
            return self.source
        tag = "bookworm" if self.version == "latest" else f"{self.version}-bookworm"
        return f"postgres:{tag}"

    def migrations_dir(self) -> Path | None:
        if self.migrations is None:
            return None
        return ((self.base_dir or Path.cwd()) / self.migrations).resolve()

    @override
    def host_paths(self) -> list[Path]:
        migrations = self.migrations_dir()
        return [migrations] if migrations is not None and self.migrate_at == "build" else []

    @override
    def bundle_stages(self, rel: PathMap) -> list[str]:
        stages = stage(self.source_image(), self.stage_name(), _BUNDLE.replace("{name}", self.name))
        migrations = self.migrations_dir()
        if self.migrate_at != "build" or migrations is None:
            return stages
        # The schema baked into a data dir, by the source image's own server.
        script = f"""set -eu
pgbin=$(ls -d /usr/lib/postgresql/*/bin | sort -V | tail -n 1)
export PGDATA=/bundle-data{DATA_DIR}
mkdir -p "$PGDATA" {SOCKET_DIR}
chown -R postgres:postgres /bundle-data {SOCKET_DIR}
runuser -u postgres -- "$pgbin/initdb" -D "$PGDATA" -U postgres -A trust --no-locale -E UTF8
runuser -u postgres -- "$pgbin/pg_ctl" -D "$PGDATA" -w -o "-c listen_addresses='' -c unix_socket_directories={SOCKET_DIR}" start
export PATH="$pgbin:$PATH"
{self._bootstrap_shell(f"-h {SOCKET_DIR} -p 5432")}
{self._migrations_shell("/tmp/migrations", f"-h {SOCKET_DIR} -p 5432")}
runuser -u postgres -- "$pgbin/pg_ctl" -D "$PGDATA" -w -m fast stop"""
        return [
            *stages,
            "",
            f"FROM {self.stage_name()} AS {self.stage_name('-data')}",
            f"COPY {json.dumps([rel(migrations) + '/', '/tmp/migrations/'])}",
            "RUN <<'EOF'",
            script,
            "EOF",
        ]

    @override
    def install_steps(self, rel: PathMap) -> list[str]:
        steps = [
            f"COPY --from={self.stage_name()} {BUNDLE_DIR}/ /",
            install_check(
                self.name,
                self.display_name,
                self.source_image(),
                [("postgres", "/var/lib/postgresql")],
            )
            + " && \\\n    "
            + "([ -e /usr/share/zoneinfo/UTC ] || { mkdir -p /usr/share/zoneinfo && "
            + "cp -a /usr/share/zoneinfo-postgres/. /usr/share/zoneinfo/; }) && "
            + "rm -rf /usr/share/zoneinfo-postgres",
        ]
        if self.migrate_at == "build" and self.migrations is not None:
            steps.append(
                f"COPY --from={self.stage_name('-data')} --chown=postgres:postgres "
                + f"/bundle-data{DATA_DIR} {DATA_DIR}"
            )
        return steps

    # Run

    @override
    def start_command(self) -> str:
        settings = " ".join(f"-c {s.format(host=HOST, port=self.port)}" for s in SERVER_SETTINGS)
        return (
            f"mkdir -p {SOCKET_DIR} /var/lib/postgresql && "
            + f"chown postgres:postgres {SOCKET_DIR} /var/lib/postgresql && "
            + f"{{ [ -s {DATA_DIR}/PG_VERSION ] || runuser -u postgres -- initdb -D {DATA_DIR} "
            + "-U postgres -A trust --no-locale -E UTF8 --no-sync; } && "
            # A data dir baked into the image (migrate_at="build") comes out of COPY as 0755.
            + f"chmod 700 {DATA_DIR} && "
            + f"exec runuser -u postgres -- postgres -D {DATA_DIR} {settings}"
        )

    @override
    def ready(self) -> Readiness:
        return cmd(f"pg_isready -q -h {HOST} -p {self.port} -U postgres")

    @override
    async def start(self, ctx: AgentContext) -> None:
        await super().start(ctx)
        if self.migrate_at != "build":
            _ = await run_script(
                ctx,
                self._bootstrap_shell(self._conn("postgres")),
                what=f"Creating database {self.database!r} ({self.name})",
            )

    @override
    async def migrate(self, ctx: AgentContext) -> None:
        if self.migrate_command is not None:
            _ = await ctx.container.sh(self.migrate_command, env=self.env(), timeout=600)
            return
        migrations = self.migrations_dir()
        if migrations is None or self.migrate_at == "build":
            return
        remote = f"/tmp/.{uuid.uuid4().hex[:12]}"
        try:
            await ctx.container.copy(migrations, remote)
            _ = await run_script(
                ctx,
                self._migrations_shell(remote, self._conn("postgres")),
                what=f"Migrations from {migrations} ({self.name})",
                timeout=600,
            )
        finally:
            _ = await ctx.container.sh(f"rm -rf {remote}", user="root", check=False)

    @override
    async def seed(self, ctx: AgentContext) -> None:
        path = resolve_seed(self.seed_spec, ctx.case, self.base_dir, self.name)
        if path is not None:
            _ = await self._psql(ctx, path.read_text(), what=f"seed {path.name}")

    @override
    def template_values(self) -> dict[str, str]:
        return {
            **super().template_values(),
            "port": str(self.port),
            "database": self.database,
            "user": self.user,
            "password": self.password,
            # `{user}` / `{password}` stay raw (PGPASSWORD); the URL needs them encoded.
            "url": f"postgresql://{quote(self.user, safe='')}:{quote(self.password, safe='')}"
            + f"@{HOST}:{self.port}/{quote(self.database, safe='')}",
        }

    @override
    async def snapshot(self, ctx: AgentContext, out_dir: str) -> None:
        if not self.snapshot_tables:
            return
        quoted = shlex.quote(out_dir)
        lines = [f"rm -rf {quoted}", f"mkdir -p {quoted}"]
        for table in self.snapshot_tables:
            query = f"select coalesce(json_agg(t), '[]'::json) from {table} t"
            lines.append(
                f"psql {self._conn(self.user)} -d {shlex.quote(self.database)} -v ON_ERROR_STOP=1 -X -At -c {shlex.quote(query)} "
                + f"> {shlex.quote(f'{out_dir}/{table}.json')}"
            )
        lines.append(f"chmod -R a+rX {quoted}")
        _ = await run_script(ctx, "set -e\n" + "\n".join(lines), what=f"Snapshot of {self.name}")

    # Helpers, in `setup`, `run`, `collect` and a shared-mode `verify`

    async def query(self, ctx: Context, sql: str) -> list[dict[str, Any]]:
        """Run a query and return its rows."""
        require_live(self, ctx)
        wrapped = f"select coalesce(json_agg(t), '[]'::json) from ({_strip(sql)}) t"
        out = await self._psql(ctx, wrapped, what="query")
        return json.loads(out or "[]")

    async def execute(self, ctx: Context, sql: str) -> None:
        """Run statements (`ON_ERROR_STOP`)."""
        require_live(self, ctx)
        _ = await self._psql(ctx, sql, what="execute")

    async def insert(self, ctx: Context, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Insert rows (dicts by column); columns no row has keep their defaults."""
        require_live(self, ctx)
        rows = list(rows)
        if not rows:
            return
        columns: list[str] = []
        for row in rows:
            columns += [c for c in row if c not in columns]
        names = ", ".join(_quote_ident(c) for c in columns)
        tag = f"$j{uuid.uuid4().hex[:8]}$"
        payload = json.dumps(rows, default=str)
        sql = (
            f"insert into {table} ({names}) select {names} from "
            + f"json_populate_recordset(null::{table}, {tag}{payload}{tag})"
        )
        _ = await self._psql(ctx, sql, what=f"insert into {table}")

    # Internals

    def _conn(self, user: str) -> str:
        return f"-h {HOST} -p {self.port} -U {shlex.quote(user)}"

    async def _psql(self, ctx: Context, sql: str, *, what: str) -> str:
        script = heredoc(
            f"psql {self._conn(self.user)} -d {shlex.quote(self.database)} "
            + "-v ON_ERROR_STOP=1 -X -q -At -f -",
            sql,
        )
        return (await run_script(ctx, script, what=f"{self.name}: {what}")).strip()

    def _bootstrap_shell(self, conn: str) -> str:
        """Role, database, extensions and `init_sql`; idempotent."""
        statements = ["\\set ON_ERROR_STOP on"]
        if self.user != "postgres":
            statements.append(
                "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = "
                + f"{_literal(self.user)}) THEN CREATE ROLE {_quote_ident(self.user)} LOGIN "
                + f"SUPERUSER PASSWORD {_literal(self.password)}; END IF; END $$;"
            )
        statements.append(
            f"SELECT format('CREATE DATABASE %I OWNER %I', {_literal(self.database)}, "
            + f"{_literal(self.user)}) WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = "
            + f"{_literal(self.database)})\\gexec"
        )
        statements.append(f"\\connect {_quote_ident(self.database)}")
        statements += [
            f"CREATE EXTENSION IF NOT EXISTS {_quote_ident(e)};" for e in self.extensions
        ]
        statements += [_terminated(sql) for sql in self.init_sql]
        return heredoc(f"psql {conn} -U postgres -d postgres -X -q", "\n".join(statements))

    def _migrations_shell(self, directory: str, conn: str) -> str:
        quoted = shlex.quote(directory)
        return (
            f"find {quoted} -maxdepth 1 -type f -name '*.sql' | sort -V | "
            + "while IFS= read -r f; do "
            + f'psql {conn} -U postgres -d {shlex.quote(self.database)} -v ON_ERROR_STOP=1 -X -q -f "$f" '
            + '|| { echo "migration failed: ${f##*/}" >&2; exit 1; }; done'
        )

    # Configuration

    @override
    def ports(self) -> list[int]:
        return [self.port]

    @override
    def validate(self, owner: str) -> list[str]:
        warnings = super().validate(owner)
        where = f"{owner}.services[{self.name!r}]"
        if self.migrate_at not in ("setup", "build"):
            raise EvalConfigError(f'{where}: migrate_at must be "setup" or "build".')
        if self.migrations is not None and self.migrate_command is not None:
            raise EvalConfigError(f"{where}: set migrations= or migrate=, not both.")
        if self.migrate_at == "build" and self.migrations is None:
            raise EvalConfigError(
                f'{where}: migrate_at="build" bakes a folder of .sql files; set migrations=.'
            )
        migrations = self.migrations_dir()
        if migrations is not None and not migrations.is_dir():
            raise EvalConfigError(f"{where}: migrations folder {migrations} does not exist.")
        for value, what in ((self.database, "database"), (self.user, "user")):
            if not isinstance(value, str) or not _IDENT.match(value):
                raise EvalConfigError(f"{where}: {what} {value!r} must be a plain identifier.")
        return warnings


def _strip(query: str) -> str:
    return query.strip().rstrip(";").strip()


def _terminated(sql: str) -> str:
    sql = sql.strip()
    return sql if sql.endswith(";") else sql + ";"


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'

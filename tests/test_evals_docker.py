"""Services in real containers: build generated images on the acceptance bases
and drive start / ready / migrate / helpers / snapshot through `ctx`.

Slow (pulls and builds images), so opt-in: `LMNR_EVALS_DOCKER=1 uv run pytest
tests/test_evals_docker.py`. Images build for linux/amd64, like remote runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import textwrap
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lmnr.evals import AgentContext, Case, ClickHouse, Image, Postgres, Valkey, VerifyContext
from lmnr.evals.context import Secrets
from lmnr.evals.fixtures import case_fixture
from lmnr.evals.image import compile_image
from lmnr.evals.mirrors import Rystic
from lmnr.evals.services.base import Service, snapshot_services, start_services

pytestmark = pytest.mark.skipif(
    os.environ.get("LMNR_EVALS_DOCKER") != "1" or shutil.which("docker") is None,
    reason="set LMNR_EVALS_DOCKER=1 (needs Docker)",
)

BASES = ["python:3.13-slim-bookworm", "debian:trixie-slim", "ubuntu:24.04"]


class DockerEnvironment:
    """Harbor's environment interface over `docker exec` / `docker cp`."""

    def __init__(self, container: str):
        self.container: str = container

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> Any:
        argv = ["docker", "exec"]
        if user is not None:
            argv += ["-u", str(user)]
        if cwd:
            argv += ["-w", cwd]
        for key, value in (env or {}).items():
            argv += ["-e", f"{key}={value}"]
        argv += [self.container, "sh", "-c", command]
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec or 600)
        return SimpleNamespace(
            stdout=out.decode(), stderr=err.decode(), return_code=proc.returncode
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        _ = await asyncio.to_thread(
            subprocess.run,
            ["docker", "cp", str(source_path), f"{self.container}:{target_path}"],
            check=True,
        )

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        _ = await asyncio.to_thread(
            subprocess.run,
            ["docker", "cp", f"{source_dir}/.", f"{self.container}:{target_dir}"],
            check=True,
        )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        _ = await asyncio.to_thread(
            subprocess.run,
            ["docker", "cp", f"{self.container}:{source_path}", str(target_path)],
            check=True,
        )


def build(image: Image, services: list[Service], tmp_path: Path) -> str:
    compiled = compile_image(image, services, fallback_dir=tmp_path)
    assert compiled.kind == "generated" and compiled.dockerfile_text and compiled.context
    tag = f"lmnr-evals-test:{uuid.uuid4().hex[:10]}"
    result = subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "-q",
            "-t",
            tag,
            "-f",
            "-",
            str(compiled.context),
        ],
        input=compiled.dockerfile_text.encode(),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()[-4000:]
    return tag


def start_container(tag: str) -> str:
    out = subprocess.run(
        ["docker", "run", "-d", "--platform", "linux/amd64", tag, "sleep", "infinity"],
        check=True,
        capture_output=True,
    )
    return out.stdout.decode().strip()


def project(tmp_path: Path) -> tuple[Path, Case]:
    """Migrations, a case with a seed, and a runner dir."""
    pg = tmp_path / "migrations"
    pg.mkdir()
    _ = (pg / "001_customers.sql").write_text(
        "create table customers (id serial primary key, email text not null, name text);\n"
    )
    # Natural order: 2 before 10.
    _ = (pg / "2_refunds.sql").write_text(
        "create table refunds (id serial primary key, customer_id int, amount int);\n"
    )
    _ = (pg / "10_seed.sql").write_text(
        "insert into refunds (customer_id, amount) values (1, 5);\n"
    )
    ch = tmp_path / "clickhouse"
    ch.mkdir()
    _ = (ch / "001.sql").write_text(
        "CREATE TABLE events (id UInt64, kind String) ENGINE = MergeTree ORDER BY id;\n"
        + "INSERT INTO events VALUES (1, 'open');\n"
    )
    case_dir = tmp_path / "tasks" / "refund"
    (case_dir / "tests" / "fixtures").mkdir(parents=True)
    _ = (case_dir / "tests" / "fixtures" / "customers.sql").write_text(
        "insert into customers (email, name) values ('ada@example.com', 'Ada');\n"
    )
    return tmp_path, Case.from_dir(case_dir)


def agent_ctx(env: DockerEnvironment, case: Case, tmp_path: Path) -> AgentContext:
    return AgentContext(
        environment=env,
        case=case,
        runner_dir=tmp_path,
        host_logs_dir=tmp_path / "trial" / "agent",
    )


@pytest.mark.parametrize("base", BASES)
def test_services_on_base(base: str, tmp_path: Path):
    root, case = project(tmp_path)
    postgres = Postgres(
        "16",
        migrations=str(root / "migrations"),
        seed=case_fixture("customers.sql"),
        snapshot=["refunds", "customers"],
    )
    valkey = Valkey("8")
    clickhouse = ClickHouse(
        "24.8", database="analytics", migrations=str(root / "clickhouse"), snapshot=["events"]
    )
    services: list[Service] = [postgres, valkey, clickhouse]
    tag = build(Image.from_registry(base), services, tmp_path)
    container = start_container(tag)
    try:
        env = DockerEnvironment(container)
        ctx = agent_ctx(env, case, tmp_path)

        async def scenario() -> None:
            service_env = await start_services(services, ctx)
            assert service_env == {
                "DATABASE_URL": "postgresql://postgres:postgres@127.0.0.1:5432/app",
                "REDIS_URL": "redis://127.0.0.1:6379",
                "CLICKHOUSE_URL": "http://127.0.0.1:8123",
            }
            ctx.add_env(service_env)
            # Bundles don't change the image's /tmp (later apt_install steps write there).
            mode = await ctx.container.sh("stat -c %a /tmp", user="root")
            assert mode.stdout.strip() == "1777"

            # Migrations in natural order, then the case's seed.
            assert await postgres.sql(ctx, "select amount from refunds") == [{"amount": 5}]
            assert await postgres.sql(ctx, "select name from customers;") == [{"name": "Ada"}]
            await postgres.insert(ctx, "refunds", [{"customer_id": 1, "amount": 7}])
            await postgres.execute(ctx, "update refunds set amount = amount * 2 where amount = 7")
            rows = await postgres.sql(ctx, "select id, amount from refunds order by id")
            assert rows == [{"id": 1, "amount": 5}, {"id": 2, "amount": 14}]
            # The app reads DATABASE_URL from every command's env.
            out = await ctx.container.sh('psql "$DATABASE_URL" -Atc "select count(*) from refunds"')
            assert out.stdout.strip() == "2"

            assert await valkey.cmd(ctx, "SET", "k", "v") == "OK"
            assert await valkey.cmd(ctx, "GET", "k") == "v"

            assert await clickhouse.query(ctx, "SELECT id, kind FROM events") == [
                {"id": 1, "kind": "open"}
            ]
            await clickhouse.insert(ctx, "events", [{"id": 2, "kind": "close"}])
            rows = await clickhouse.query(ctx, "SELECT count() AS n FROM events")
            assert rows == [{"n": 2}]

            # Service logs and neutral process owners.
            ps = await ctx.container.sh("ps -eo user=,comm= 2>/dev/null || true", user="root")
            if ps.stdout:
                assert "postgres" in ps.stdout and "valkey" in ps.stdout

            await snapshot_services(services, ctx)
            await env.download_file(
                "/logs/artifacts/state/postgres/refunds.json", tmp_path / "refunds.json"
            )
            assert json.loads((tmp_path / "refunds.json").read_text()) == rows_pg
            await env.download_file(
                "/logs/artifacts/state/clickhouse/events.json", tmp_path / "events.json"
            )
            # Two inserts are two parts: no row order without ORDER BY.
            events = json.loads((tmp_path / "events.json").read_text())
            assert sorted(events, key=lambda e: e["id"]) == [
                {"id": 1, "kind": "open"},
                {"id": 2, "kind": "close"},
            ]

        rows_pg = [
            {"id": 1, "customer_id": 1, "amount": 5},
            {"id": 2, "customer_id": 1, "amount": 14},
        ]
        asyncio.run(scenario())
    finally:
        _ = subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)
        _ = subprocess.run(["docker", "rmi", tag], check=False, capture_output=True)


def test_postgres_migrations_at_build_and_extensions(tmp_path: Path):
    root, case = project(tmp_path)
    postgres = Postgres(
        "16",
        database="shop",
        user="shop",
        password="secret",
        migrations=str(root / "migrations"),
        migrate_at="build",
        extensions=["pgcrypto"],
        init_sql=["create table settings (k text primary key, v text)"],
        env={"DATABASE_URL": "{url}", "PGUSER": "{user}"},
    )
    tag = build(Image.from_registry("python:3.13-slim-trixie"), [postgres], tmp_path)
    container = start_container(tag)
    try:
        env = DockerEnvironment(container)
        ctx = agent_ctx(env, case, tmp_path)

        async def scenario() -> None:
            service_env = await start_services([postgres], ctx)
            assert service_env == {
                "DATABASE_URL": "postgresql://shop:secret@127.0.0.1:5432/shop",
                "PGUSER": "shop",
            }
            # Baked in the image: the schema and the rows migrations inserted.
            assert await postgres.sql(ctx, "select count(*)::int as n from refunds") == [{"n": 1}]
            assert await postgres.sql(ctx, "select gen_random_uuid() is not null as ok") == [
                {"ok": True}
            ]
            assert await postgres.sql(ctx, "select count(*)::int as n from settings") == [{"n": 0}]

        asyncio.run(scenario())
    finally:
        _ = subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)
        _ = subprocess.run(["docker", "rmi", tag], check=False, capture_output=True)


def test_dockerfile_composition_with_non_root_user(tmp_path: Path):
    """Services on top of a user's Dockerfile whose last stage sets USER."""
    _ = (tmp_path / "Dockerfile").write_text(
        textwrap.dedent(
            """\
            FROM python:3.13-slim-bookworm
            RUN useradd -m app
            USER app
            WORKDIR /home/app
            """
        )
    )
    image = Image.from_dockerfile(str(tmp_path / "Dockerfile"))
    valkey = Valkey("8")
    tag = build(image, [valkey], tmp_path)
    container = start_container(tag)
    try:
        whoami = subprocess.run(
            ["docker", "exec", container, "whoami"], capture_output=True, check=True
        )
        assert whoami.stdout.decode().strip() == "app"
        env = DockerEnvironment(container)
        ctx = agent_ctx(env, Case(name="c", dir=tmp_path), tmp_path)

        async def scenario() -> None:
            _ = await start_services([valkey], ctx)
            assert await valkey.cmd(ctx, "PING") == "PONG"

        asyncio.run(scenario())
    finally:
        _ = subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)
        _ = subprocess.run(["docker", "rmi", tag], check=False, capture_output=True)


def test_alpine_base_fails_the_build_with_a_hint(tmp_path: Path):
    compiled = compile_image(
        Image.from_registry("python:3.13-alpine"), [Valkey("8")], fallback_dir=tmp_path
    )
    assert compiled.dockerfile_text is not None
    result = subprocess.run(
        ["docker", "build", "--platform", "linux/amd64", "-q", "-f", "-", str(tmp_path)],
        input=compiled.dockerfile_text.encode(),
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "this image has no glibc (Alpine?)" in result.stderr.decode()


def test_verify_reads_snapshots_in_separate_mode(tmp_path: Path):
    """`ctx.state` reads what `collect` saved, from the trial's artifacts."""
    artifacts = tmp_path / "trial" / "artifacts" / "logs" / "artifacts" / "state" / "postgres"
    artifacts.mkdir(parents=True)
    _ = (artifacts / "refunds.json").write_text('[{"id": 1}]')
    ctx = VerifyContext(
        environment=DockerEnvironment("unused"),
        case=Case(name="c", dir=tmp_path),
        runner_dir=tmp_path,
        host_agent_dir=tmp_path / "trial" / "agent",
        secrets=Secrets([], {}),
        verifier_mode="separate",
    )
    assert ctx.state("postgres", "refunds") == [{"id": 1}]


def test_rystic_cli_installs_without_a_key(tmp_path: Path):
    """The image gets the pinned CLI; twins are pulled per trial (with a key)."""
    tag = build(
        Image.from_registry("debian:trixie-slim"), [Rystic("slack", version="1.4")], tmp_path
    )
    try:
        out = subprocess.run(
            ["docker", "run", "--rm", tag, "rystic", "--help"], capture_output=True, check=True
        )
        assert "Rystic simulators" in out.stdout.decode()
    finally:
        _ = subprocess.run(["docker", "rmi", tag], check=False, capture_output=True)

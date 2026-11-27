"""Services in real containers: build generated images on the acceptance bases
and drive start / ready / migrate / helpers / snapshot through `ctx`.

Slow (pulls and builds images), so opt-in: `LMNR_EVALS_DOCKER=1 uv run pytest
tests/test_evals_docker.py`. Images build for linux/amd64, like remote runs.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import shlex
import shutil
import subprocess
import tempfile
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
from lmnr.evals.layer import WHEEL_CONTEXT, LaminarLayer
from lmnr.evals.mirrors import Rystic
from lmnr.evals.mocks.host import collect_mocks, start_mock_server
from lmnr.evals.runner import find_runner, load_module
from lmnr.evals.services.base import Service, snapshot_services, start_services

pytestmark = pytest.mark.skipif(
    os.environ.get("LMNR_EVALS_DOCKER") != "1" or shutil.which("docker") is None,
    reason="set LMNR_EVALS_DOCKER=1 (needs Docker)",
)

BASES = ["python:3.13-slim-bookworm", "debian:trixie-slim", "ubuntu:24.04"]
REPO = Path(__file__).resolve().parent.parent


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


@functools.cache
def lmnr_wheel() -> Path:
    """This checkout as a wheel, for the Laminar layer (what lmnr-cli does for
    `LMNR_EVAL_LMNR_SPEC=<path>`)."""
    out = Path(tempfile.mkdtemp(prefix="lmnr-wheel-"))
    _ = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out), str(REPO)],
        check=True,
        capture_output=True,
    )
    return out


def build(image: Image, services: list[Service], tmp_path: Path) -> str:
    """Builds like lmnr-cli: the services plus the Laminar layer, which the
    runner's helpers (readiness, HTTP, snapshots) run on."""
    compiled = compile_image(image, [*services, LaminarLayer(lmnr=None)], fallback_dir=tmp_path)
    assert compiled.kind == "generated" and compiled.dockerfile_text and compiled.context
    tag = f"lmnr-evals-test:{uuid.uuid4().hex[:10]}"
    result = subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "--build-context",
            f"{WHEEL_CONTEXT}={lmnr_wheel()}",
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
    # clickhouse-migrations (the default migrator) names: version, `_`, anything.
    _ = (ch / "1_events.sql").write_text(
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
        "24.8",
        database="analytics",
        migrations=str(root / "clickhouse"),
        snapshot=["events"],
        # The default migrator's Node comes from trixie and needs glibc 2.38.
        migrator="clickhouse-client" if "bookworm" in base else "clickhouse-migrations",
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
            if clickhouse.migrator == "clickhouse-migrations":
                # The tool recorded what it applied.
                applied = await clickhouse.query(ctx, "SELECT version FROM _migrations")
                assert applied == [{"version": 1}]
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


def test_clickhouse_client_migrator_and_config_overrides(tmp_path: Path):
    """migrator="clickhouse-client" applies any *.sql in natural order, and the
    config / users files are merged over the generated ones."""
    ch = tmp_path / "clickhouse"
    ch.mkdir()
    _ = (ch / "001_events.sql").write_text(
        "CREATE TABLE events (id UInt64) ENGINE = MergeTree ORDER BY id;\n"
    )
    _ = (ch / "2_more.sql").write_text("INSERT INTO events VALUES (2);\n")
    _ = (ch / "10_last.sql").write_text("INSERT INTO events SELECT max(id) * 5 FROM events;\n")
    _ = (tmp_path / "prod.xml").write_text(
        "<clickhouse><max_concurrent_queries>37</max_concurrent_queries></clickhouse>\n"
    )
    _ = (tmp_path / "users.yaml").write_text("profiles:\n  default:\n    max_threads: 3\n")
    clickhouse = ClickHouse(
        "25.8",
        migrations=str(ch),
        migrator="clickhouse-client",
        config=str(tmp_path / "prod.xml"),
        users=str(tmp_path / "users.yaml"),
    )
    tag = build(Image.python("3.13"), [clickhouse], tmp_path)
    container = start_container(tag)
    try:
        env = DockerEnvironment(container)
        ctx = agent_ctx(env, Case(name="c", dir=tmp_path), tmp_path)

        async def scenario() -> None:
            _ = await start_services([clickhouse], ctx)
            rows = await clickhouse.query(ctx, "SELECT id FROM events ORDER BY id")
            assert rows == [{"id": 2}, {"id": 10}]
            assert await clickhouse.query(ctx, "SHOW TABLES") == [{"name": "events"}]
            setting = await clickhouse.query(
                ctx,
                "SELECT value FROM system.server_settings WHERE name = 'max_concurrent_queries'",
            )
            assert setting == [{"value": "37"}]
            threads = await clickhouse.query(ctx, "SELECT getSetting('max_threads') AS n")
            assert threads == [{"n": 3}]

        asyncio.run(scenario())
    finally:
        _ = subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)
        _ = subprocess.run(["docker", "rmi", tag], check=False, capture_output=True)


def test_pnpm_install(tmp_path: Path):
    web = tmp_path / "web"
    web.mkdir()
    data = Path(__file__).parent / "data" / "evals_project" / "web-pnpm"
    for name in ("package.json", "pnpm-lock.yaml", ".npmrc"):
        _ = (web / name).write_bytes((data / name).read_bytes())
    _ = (web / "index.js").write_text("console.log(require('ms')('2 days'));\n")
    image = Image.node("24").pnpm_install(str(web), "/srv/web").add_dir(str(web), "/srv/web")
    tag = build(image, [], tmp_path)
    try:
        out = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                tag,
                "sh",
                "-c",
                "node /srv/web/index.js && pnpm --version && ls -A /tmp && ls -A /srv/web",
            ],
            capture_output=True,
            check=True,
        )
        lines = out.stdout.decode().split()
        assert lines[:2] == ["172800000", "10.20.0"]  # packageManager's version
        assert "pnpm-store" not in lines
        # .npmrc came in with add_dir; the install only mounted it.
        assert "node_modules" in lines
    finally:
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


def test_clickhouse_migrations_on_bookworm_fails_the_build_with_a_hint(tmp_path: Path):
    """The tool's bundled libstdc++ (trixie) needs glibc 2.38; bookworm has 2.36."""
    clickhouse = ClickHouse("24.8", migrations=str(tmp_path))
    compiled = compile_image(
        Image.from_registry("python:3.13-slim-bookworm"), [clickhouse], fallback_dir=tmp_path
    )
    assert compiled.dockerfile_text is not None
    result = subprocess.run(
        # Plain progress: the check's message is the step's output.
        ["docker", "build", "--platform", "linux/amd64", "--progress=plain", "-f", "-"]
        + [str(tmp_path)],
        input=compiled.dockerfile_text.encode(),
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    stderr = result.stderr.decode()
    assert "this image has glibc 2.36, but 2.38 or newer is needed." in stderr
    assert 'pass migrator="clickhouse-client"' in stderr


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


# The Laminar layer on an image with neither curl nor Python

FAKE_RYSTIC_JS = """\
#!/usr/bin/env node
// rystic pull <product> --version <v> | rystic run <product> --name <n> --port <p>
const http = require("http");
const args = process.argv.slice(2);
if (args[0] === "pull") { console.log(`pulled ${args[1]}`); process.exit(0); }
const port = Number(args[args.indexOf("--port") + 1]);
const state = { messages: [] };
const calls = [];
http.createServer((req, res) => {
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    const url = new URL(req.url, "http://x");
    // No Content-Length: Node answers chunked.
    const reply = (code, value) => { res.writeHead(code, { "Content-Type": "application/json" });
                                     res.write(JSON.stringify(value)); res.end(); };
    if (url.pathname === "/_rystic/routes") return reply(200, { routes: [] });
    if (url.pathname === "/_rystic/seed") { Object.assign(state, JSON.parse(body));
                                            res.writeHead(204); return res.end(); }
    if (url.pathname === "/_rystic/inspect") return reply(200, state);
    if (url.pathname === "/_rystic/calls") return reply(200, { calls, total: calls.length });
    if (url.pathname === "/_rystic/settle") return reply(200, { settled: true });
    calls.push({ path: url.pathname });
    reply(200, { ok: true, messages: state.messages });
  });
}).listen(port, "127.0.0.1");
"""

SHARED_HANDLERS = """
from lmnr.evals import mock


def get_customer(call, ctx):
    return ctx.fixture_json("crm.json").get(call.args[0] if isinstance(call.args, list)
                                            else call.args["email"])


def zendesk(req, ctx):
    if req.method == "POST":
        return mock.Response(201, json={"created": req.json()})
    return None
"""

MOCKS_EVAL = """
from lmnr.evals import EvalRunner, Image, mock
from _shared.handlers import get_customer, zendesk


def refund(call, ctx):
    if ctx.state.setdefault("refunded", False):
        raise mock.Error("DuplicateRefund", "charge already refunded")
    ctx.state["refunded"] = True
    return {"status": "ok"}


class SupportEval(EvalRunner):
    image = Image.node("24")
    mocks = {
        "crm.get_customer": get_customer,
        "payments.refund": refund,
        "https://api.zendesk.com": mock.http(zendesk, base_url_env="ZENDESK_URL"),
    }
    run_command = "true"
    grade_with_test_sh = True
"""

# What the app does with LMNR_MOCK_URL / ZENDESK_URL (phase 4's SDK does the
# function route itself).
FETCH_JS = """
const [key, args, method] = process.argv.slice(2);
const url = key ? `${process.env.LMNR_MOCK_URL}/fn/${encodeURIComponent(key)}`
                : `${process.env.ZENDESK_URL}/api/v2/tickets/1/comments`;
const body = key ? JSON.stringify({ args: JSON.parse(args), lang: "typescript" }) : args;
fetch(url, { method: method || "POST", body: method === "GET" ? undefined : body })
  .then(async (r) => console.log(JSON.stringify({ status: r.status, body: await r.json() })));
"""


class NoCli(Rystic):
    """Rystic with the fake simulator the image already has, not the real CLI."""

    def bundle_stages(self, rel: Any) -> list[str]:
        return []

    def install_steps(self, rel: Any) -> list[str]:
        return []


def test_layer_on_an_image_without_curl_or_python(tmp_path: Path):
    """Readiness, mirror HTTP (chunked), ClickHouse snapshots and the mock
    server all run on the layer's Python, on `node:24-trixie-slim`."""
    root, case = project(tmp_path)
    fake = tmp_path / "rystic.js"
    _ = fake.write_text(FAKE_RYSTIC_JS)
    _ = (case.dir / "tests" / "fixtures" / "crm.json").write_text(
        json.dumps({"ana@example.com": {"id": "cus_1"}})
    )
    _ = (case.dir / "tests" / "fixtures" / "slack.json").write_text(
        json.dumps({"messages": [{"text": "refund please"}]})
    )
    evals = tmp_path / "evals"
    (evals / "_shared").mkdir(parents=True)
    _ = (evals / "_shared" / "handlers.py").write_text(textwrap.dedent(SHARED_HANDLERS))
    eval_path = evals / "support" / "agent" / "eval.py"
    eval_path.parent.mkdir(parents=True)
    _ = eval_path.write_text(textwrap.dedent(MOCKS_EVAL))
    runner = find_runner(load_module(eval_path))
    assert runner.validate() == []

    postgres = Postgres("16", migrations=str(root / "migrations"), snapshot=["refunds"])
    clickhouse = ClickHouse("24.8", migrations=str(root / "clickhouse"), snapshot=["events"])
    slack = NoCli("slack", version="1.4", seed=case_fixture("slack.json"), snapshot=True)
    services: list[Service] = [postgres, clickhouse, slack]
    image = (
        Image.node("24")
        .add_file(str(fake), "/usr/local/bin/rystic")
        .run_commands("chmod 755 /usr/local/bin/rystic")
    )
    tag = build(image, services, tmp_path)
    container = start_container(tag)
    try:
        env = DockerEnvironment(container)
        ctx = AgentContext(
            environment=env,
            case=case,
            runner_dir=eval_path.parent,
            host_logs_dir=tmp_path / "trial" / "agent",
            exec_env={"RYSTIC_TOKEN": "rys_" + "a" * 43, "RYSTIC_RELEASE_ENDPOINT": "http://x"},
        )

        async def sh(command: str) -> str:
            return (await ctx.container.sh(command, user="root")).stdout

        async def scenario() -> None:
            assert "not found" in await sh("command -v curl python3 python || echo not found")
            size = await sh("du -sh /opt/lmnr | cut -f1")
            print(f"Laminar layer: {size.strip()}")

            ctx.add_env(await start_services(services, ctx))
            assert (await slack.state(ctx))["messages"] == [{"text": "refund please"}]
            assert (await slack.get(ctx, "/api/conversations.history"))["ok"] is True
            assert (await slack.settle(ctx))["settled"] is True

            running = await start_mock_server(ctx, runner.mock_specs(), eval_path=eval_path)
            ctx.add_env(running.env)
            # Nothing of the mocks on disk, a neutral process.
            leftovers = await sh(
                "find /tmp /logs /case \\( -name c.json -o -name 'mocks*' -o -name '*.py' "
                "-o -name crm.json \\) 2>/dev/null"
            )
            assert leftovers == ""
            assert (await sh(f"cat /proc/{running.pid}/comm")).strip() == "relay"
            processes = await sh(
                "for p in /proc/[0-9]*; do tr '\\0' ' ' < $p/cmdline 2>/dev/null; echo; done"
            )
            assert "relay.py serve /tmp/." in processes
            for word in ("mock", "eval", "test", "fixture"):
                assert word not in processes, processes

            _ = await env.upload_file(_write(tmp_path / "fetch.js", FETCH_JS), "/srv/fetch.js")

            async def app(*args: str) -> Any:
                out = await ctx.container.sh(shlex.join(["node", "/srv/fetch.js", *args]))
                return json.loads(out.stdout)

            assert await app("crm.get_customer", '["ana@example.com"]') == {
                "status": 200,
                "body": {"action": "respond", "value": {"id": "cus_1"}},
            }
            assert (await app("payments.refund", "{}"))["body"]["action"] == "respond"
            assert (await app("payments.refund", "{}"))["body"]["error"]["type"] == (
                "DuplicateRefund"
            )
            assert await app("", '{"body": "hi"}') == {
                "status": 201,
                "body": {"created": {"body": "hi"}},
            }
            assert (await app("", "{}", "GET"))["status"] == 501

            # The agent plants a call log; collect replaces it.
            _ = await sh(
                "mkdir -p /logs/artifacts && echo '{\"forged\": 1}' > /logs/artifacts/mocks.jsonl"
            )
            await snapshot_services(services, ctx)
            await collect_mocks(ctx, running)
            await env.download_file("/logs/artifacts/mocks.jsonl", tmp_path / "mocks.jsonl")
            await env.download_file(
                "/logs/artifacts/state/clickhouse/events.json", tmp_path / "events.json"
            )
            await env.download_file(
                "/logs/artifacts/state/slack/calls.json", tmp_path / "calls.json"
            )

        asyncio.run(scenario())
    finally:
        _ = subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)
        _ = subprocess.run(["docker", "rmi", tag], check=False, capture_output=True)

    calls = [json.loads(line) for line in (tmp_path / "mocks.jsonl").read_text().splitlines()]
    assert [(c["key"], c["outcome"]) for c in calls] == [
        ("crm.get_customer", "respond"),
        ("payments.refund", "respond"),
        ("payments.refund", "raise"),
        ("https://api.zendesk.com", "respond"),
        ("https://api.zendesk.com", "unmatched"),
    ]
    assert calls[0]["request"] == {"args": ["ana@example.com"], "lang": "typescript"}
    assert json.loads((tmp_path / "events.json").read_text()) == [{"id": 1, "kind": "open"}]
    assert json.loads((tmp_path / "calls.json").read_text()) == [
        {"path": "/api/conversations.history"}
    ]
    assert "listening on 127.0.0.1" in (tmp_path / "trial" / "agent" / "mocks.log").read_text()


def _write(path: Path, text: str) -> Path:
    _ = path.write_text(text)
    return path

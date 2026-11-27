"""Services and mirrors without Docker: validation, the lifecycle `LaminarAgent`
drives, separate-mode rules, and the Rystic adapter against a fake simulator
(a small HTTP server run by a fake `rystic` executable, commands run on the
host). Real containers: `tests/test_evals_docker.py`."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typing_extensions import override

from lmnr.evals import (
    AgentContext,
    Case,
    ClickHouse,
    EvalConfigError,
    EvalRunner,
    Image,
    Postgres,
    Service,
    Valkey,
    VerifyContext,
    case_fixture,
    cmd,
)
from lmnr.evals.context import Context, Secrets
from lmnr.evals.image import compile_image
from lmnr.evals.mirrors import Mirror, Rystic
from lmnr.evals.mirrors import rystic as rystic_module
from lmnr.evals.services.base import snapshot_services, start_services


class RecordingEnvironment:
    """Records commands; every command succeeds with empty output."""

    def __init__(self):
        self.commands: list[tuple[str, Any]] = []
        self.envs: list[dict[str, str] | None] = []

    async def exec(self, command: str, cwd=None, env=None, timeout_sec=None, user=None) -> Any:
        self.commands.append((command, user))
        self.envs.append(env)
        # `service` reads the pid from the last line.
        return SimpleNamespace(
            stdout="42\n" if "echo $!" in command else "", stderr="", return_code=0
        )

    async def upload_file(self, source_path, target_path) -> None:
        self.commands.append((f"upload {Path(source_path).name} -> {target_path}", None))

    async def upload_dir(self, source_dir, target_dir) -> None:
        self.commands.append((f"upload_dir {source_dir} -> {target_dir}", None))

    async def download_file(self, source_path, target_path) -> None:
        raise FileNotFoundError(source_path)


class LocalEnvironment:
    """Runs commands on the host with bash (users are ignored)."""

    async def exec(self, command: str, cwd=None, env=None, timeout_sec=None, user=None) -> Any:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return SimpleNamespace(
            stdout=out.decode(), stderr=err.decode(), return_code=proc.returncode
        )

    async def upload_file(self, source_path, target_path) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        _ = Path(target_path).write_bytes(Path(source_path).read_bytes())

    async def upload_dir(self, source_dir, target_dir) -> None:
        _ = await asyncio.to_thread(
            subprocess.run, ["cp", "-a", f"{source_dir}/.", target_dir], check=True
        )

    async def download_file(self, source_path, target_path) -> None:
        _ = Path(target_path).write_bytes(Path(source_path).read_bytes())


def runner(**attrs: Any) -> type[EvalRunner]:
    async def run(self, ctx):
        pass

    async def verify(self, ctx):
        return 1

    namespace: dict[str, Any] = {"image": Image.python("3.13"), "run": run, "verify": verify}
    namespace.update(attrs)
    return type("SupportEval", (EvalRunner,), namespace)


def make_case(tmp_path: Path, fixtures: dict[str, str] | None = None) -> Case:
    case_dir = tmp_path / "tasks" / "refund"
    (case_dir / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    for name, content in (fixtures or {}).items():
        _ = (case_dir / "tests" / "fixtures" / name).write_text(content)
    return Case.from_dir(case_dir)


# Validation


@pytest.mark.parametrize(
    ("services", "message"),
    [
        ([Postgres("16"), Postgres("16")], "two services named 'postgres'"),
        ([Postgres("16"), Postgres("16", name="pg2")], "both use port 5432"),
        (
            [Valkey("8"), Valkey("8", name="cache", port=6380, env="REDIS_URL")],
            "REDIS_URL is set by both",
        ),
        ([Postgres("")], "needs a version"),
        ([Postgres("16", env={"DB": "{nope}"})], r"uses \{nope\}"),
        ([Postgres("16", migrations="missing-dir")], "does not exist"),
        ([Postgres("16", migrate_at="build")], "set migrations="),
        ([Postgres("16", database="bad name")], "plain identifier"),
        ([Service.custom("x", "1", start="x", ready="true", bundle=["/x"])], "set source="),
        ([Rystic("Slack", version="1")], "lowercase"),
    ],
)
def test_invalid_services_raise(services: list[Service], message: str):
    with pytest.raises(EvalConfigError, match=message):
        _ = runner(services=services).validate()


def test_service_env_cannot_shadow_runner_env():
    with pytest.raises(EvalConfigError, match="DATABASE_URL is set by both"):
        _ = runner(env={"DATABASE_URL": "x"}, services=[Postgres("16")]).validate()


def test_latest_version_warns():
    warnings = runner(services=[Valkey("latest")]).validate()
    assert warnings == [
        "SupportEval.services['valkey'] uses version \"latest\": trials run against whatever the "
        + "image has when it is built. Pin a version for reproducible runs."
    ]
    assert Valkey("latest").source_image() == "valkey/valkey:bookworm"


def test_postgres_url_encodes_credentials():
    pg = Postgres("16", user="app$user", password="p@ss:w/rd#1", database="shop")
    assert pg.env() == {
        "DATABASE_URL": "postgresql://app%24user:p%40ss%3Aw%2Frd%231@127.0.0.1:5432/shop"
    }
    # The separate placeholders stay raw, e.g. for PGPASSWORD.
    raw = Postgres("16", password="p@ss", env={"PGPASSWORD": "{password}"})
    assert raw.env() == {"PGPASSWORD": "p@ss"}


@pytest.mark.parametrize("fails", [False, True])
def test_clickhouse_snapshot_fails_when_a_query_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "clickhouse"
    _ = fake.write_text(
        "#!/bin/sh\necho 'Code: 60. Table default.events does not exist' >&2; exit 60\n"
        if fails
        else "#!/bin/sh\necho '{\"id\":1}'; echo '{\"id\":2}'\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    ch = ClickHouse("25.8", snapshot=["events"])
    ctx = AgentContext(
        environment=LocalEnvironment(),
        case=make_case(tmp_path),
        runner_dir=tmp_path,
        host_logs_dir=tmp_path / "a",
    )
    out = tmp_path / "state" / "clickhouse"
    if fails:
        with pytest.raises(Exception, match="Snapshot of clickhouse"):
            asyncio.run(ch.snapshot(ctx, str(out)))
    else:
        asyncio.run(ch.snapshot(ctx, str(out)))
        assert json.loads((out / "events.json").read_text()) == [{"id": 1}, {"id": 2}]


def test_env_templates():
    pg = Postgres("16", database="shop", env={"DATABASE_URL": "{url}", "PGPORT": "{port}"})
    assert pg.env() == {
        "DATABASE_URL": "postgresql://postgres:postgres@127.0.0.1:5432/shop",
        "PGPORT": "5432",
    }
    assert ClickHouse("25.8", http_port=8124).env() == {"CLICKHOUSE_URL": "http://127.0.0.1:8124"}
    assert Valkey("8", env="CACHE_URL").env() == {"CACHE_URL": "redis://127.0.0.1:6379"}


def test_self_service_finds_declared_services():
    pg, mirror = Postgres("16"), Rystic("slack", version="1.4")
    instance = runner(services=[pg, Valkey("8"), mirror])()
    assert instance.service(Postgres) is pg
    assert instance.service("postgres") is pg
    assert instance.mirror("slack") is mirror
    with pytest.raises(EvalConfigError, match="declares no service 'ClickHouse'"):
        _ = instance.service(ClickHouse)
    with pytest.raises(EvalConfigError, match="not a mirror"):
        _ = instance.mirror("postgres")
    two = runner(services=[Valkey("8"), Valkey("8", name="cache", port=6380, env="CACHE_URL")])()
    with pytest.raises(EvalConfigError, match="several Valkey services"):
        _ = two.service(Valkey)


def test_compiling_is_deterministic(tmp_path: Path):
    """Same definition, same text: the Docker cache and lmnr-cli's dedup rely on it."""
    (tmp_path / "migrations").mkdir()
    services = [
        Postgres("16", migrations=str(tmp_path / "migrations"), init_sql=["select 1"]),
        Valkey("8"),
    ]
    image = Image.python("3.13")
    first = compile_image(image, services, fallback_dir=tmp_path)
    second = compile_image(image, services, fallback_dir=tmp_path)
    assert first.dockerfile_text == second.dockerfile_text


# Lifecycle


def test_services_start_migrate_seed_in_order(tmp_path: Path):
    calls: list[str] = []

    class Fake(Service):
        def __init__(self, name: str):
            super().__init__(name, "1", env={f"{name.upper()}_URL": "http://{host}"})

        async def start(self, ctx):
            calls.append(f"{self.name}:start")

        async def migrate(self, ctx):
            calls.append(f"{self.name}:migrate")

        async def seed(self, ctx):
            calls.append(f"{self.name}:seed")

        async def snapshot(self, ctx, out_dir):
            calls.append(f"{self.name}:snapshot {out_dir}")

    services: list[Service] = [Fake("a"), Fake("b")]
    ctx = AgentContext(
        environment=RecordingEnvironment(),
        case=make_case(tmp_path),
        runner_dir=tmp_path,
        host_logs_dir=tmp_path / "agent",
    )

    async def scenario() -> dict[str, str]:
        env = await start_services(services, ctx)
        await snapshot_services(services, ctx)
        return env

    env = asyncio.run(scenario())
    assert env == {"A_URL": "http://127.0.0.1", "B_URL": "http://127.0.0.1"}
    for name in ("a", "b"):
        assert (
            calls.index(f"{name}:start")
            < calls.index(f"{name}:migrate")
            < calls.index(f"{name}:seed")
        )
    assert calls[-2:] == [
        "a:snapshot /logs/artifacts/state/a",
        "b:snapshot /logs/artifacts/state/b",
    ]


def test_postgres_commands(tmp_path: Path):
    (tmp_path / "migrations").mkdir()
    pg = Postgres(
        "16",
        migrations=str(tmp_path / "migrations"),
        seed=case_fixture("seed.sql"),
        extensions=["pgcrypto"],
    )
    env = RecordingEnvironment()
    ctx = AgentContext(
        environment=env,
        case=make_case(tmp_path, {"seed.sql": "insert into t values (1);"}),
        runner_dir=tmp_path,
        host_logs_dir=tmp_path / "agent",
    )
    asyncio.run(start_services([pg], ctx))
    commands = [c for c, _ in env.commands]
    start = next(c for c in commands if "nohup" in c)
    assert "runuser -u postgres -- postgres -D /var/lib/postgresql/data" in start
    assert "-c listen_addresses=127.0.0.1" in start
    # Bootstrap, migrations and the seed go through temp scripts, removed after.
    uploads = [c for c in commands if c.startswith("upload ")]
    assert any("upload_dir" in c for c in commands)
    assert len(uploads) == 3
    assert all(
        c.startswith("bash /tmp/.") and "rm -f" in c
        for c in commands
        if c.startswith("bash /tmp/.")
    )
    # Everything as root; the agent's user never touches the service.
    assert all(user == "root" for c, user in env.commands if "psql" in c or "nohup" in c)


@pytest.mark.parametrize("migrator", ["clickhouse-migrations", "clickhouse-client"])
def test_clickhouse_migrators(tmp_path: Path, migrator: str):
    (tmp_path / "migrations").mkdir()
    _ = (tmp_path / "migrations" / "1_events.sql").write_text("CREATE TABLE events (id UInt64);")
    ch = ClickHouse("25.8", migrations=str(tmp_path / "migrations"), migrator=migrator)  # pyright: ignore[reportArgumentType]
    assert runner(services=[ch]).validate() == []
    env = RecordingEnvironment()
    ctx = AgentContext(
        environment=env, case=make_case(tmp_path), runner_dir=tmp_path, host_logs_dir=tmp_path / "a"
    )
    asyncio.run(start_services([ch], ctx))
    uploads = [c for c, _ in env.commands if c.startswith("upload ")]
    assert uploads
    stages = "\n".join(ch.bundle_stages(lambda p: str(p)))
    if migrator == "clickhouse-migrations":
        assert "FROM node:24-bookworm-slim AS svc-clickhouse-migrations" in stages
        assert "clickhouse-migrations@1.6.0" in stages
    else:
        assert "node" not in stages


def test_clickhouse_validation(tmp_path: Path):
    (tmp_path / "migrations").mkdir()
    _ = (tmp_path / "migrations" / "001.sql").write_text("SELECT 1;")
    _ = (tmp_path / "config.json").write_text("{}")
    with pytest.raises(
        EvalConfigError, match=r'has 001.sql. Rename them, or pass migrator="clickhouse-client"'
    ):
        _ = runner(
            services=[ClickHouse("25.8", migrations=str(tmp_path / "migrations"))]
        ).validate()
    plain = ClickHouse(
        "25.8", migrations=str(tmp_path / "migrations"), migrator="clickhouse-client"
    )
    assert runner(services=[plain]).validate() == []
    with pytest.raises(EvalConfigError, match="users file .* does not exist"):
        _ = runner(services=[ClickHouse("25.8", users=str(tmp_path / "users.xml"))]).validate()
    with pytest.raises(EvalConfigError, match="must be .xml or .yaml"):
        _ = runner(services=[ClickHouse("25.8", config=str(tmp_path / "config.json"))]).validate()


def test_clickhouse_config_overrides_are_merged_files(tmp_path: Path):
    _ = (tmp_path / "prod.xml").write_text("<clickhouse/>")
    _ = (tmp_path / "users.yaml").write_text("profiles: {}")
    ch = ClickHouse("25.8", config=str(tmp_path / "prod.xml"), users=str(tmp_path / "users.yaml"))
    assert ch.host_paths() == [tmp_path / "prod.xml", tmp_path / "users.yaml"]
    steps = ch.install_steps(lambda p: p.name)
    assert steps[-2:] == [
        'COPY ["prod.xml", "/etc/clickhouse-server/config.d/prod.xml"]',
        'COPY ["users.yaml", "/etc/clickhouse-server/users.d/users.yaml"]',
    ]


def test_live_helpers_fail_clearly_in_separate_verifier_mode(tmp_path: Path):
    pg = Postgres("16")
    ctx = VerifyContext(
        environment=RecordingEnvironment(),
        case=make_case(tmp_path),
        runner_dir=tmp_path,
        host_agent_dir=tmp_path / "agent",
        secrets=Secrets([], {}),
        verifier_mode="separate",
    )
    with pytest.raises(EvalConfigError, match=r"declare snapshot=\[\.\.\.\]"):
        asyncio.run(pg.sql(ctx, "select 1"))
    with pytest.raises(FileNotFoundError, match="add 'refunds' to the service's snapshot"):
        _ = ctx.state("postgres", "refunds")


def test_case_fixture_paths_are_checked():
    with pytest.raises(EvalConfigError):
        _ = case_fixture("../secrets.json")
    with pytest.raises(EvalConfigError):
        _ = case_fixture("/abs.json")


# Rystic against a fake simulator

FAKE_RYSTIC = textwrap.dedent(
    """\
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse

    args = sys.argv[1:]
    if args[0] == "pull":  # rystic pull <product> --version <v>
        print(f"pulled {args[1]} {args[3]}")
        sys.exit(0)
    # rystic run <product> --name <name> --port <port>
    product, port = args[1], int(args[args.index("--port") + 1])
    state = {"product": product, "messages": []}
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/_rystic/routes":
                return self.reply(200, {"routes": ["conversations.history"]})
            if url.path == "/_rystic/inspect":
                return self.reply(200, state)
            if url.path == "/_rystic/calls":
                query = parse_qs(url.query)
                assert query == {"summary": ["1"], "writes": ["1"]}, query
                return self.reply(200, {"calls": calls, "dropped": 0, "epoch": 1,
                                        "total": len(calls)})
            if url.path == "/_rystic/settle":
                quiet = int(parse_qs(url.query)["quiet_ms"][0])
                return self.reply(200, {"settled": True, "idle_ms": quiet, "last_call_seq": 0})
            if url.path == "/api/conversations.history":
                calls.append({"path": url.path, "ok": True})
                return self.reply(200, {"channel": parse_qs(url.query)["channel"][0],
                                        "messages": state["messages"]})
            self.reply(404, {"error": "not_found"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/_rystic/seed":  # merges; 204, no body
                state.update(json.loads(body))
                self.send_response(204)
                self.end_headers()
                return
            self.reply(404, {})

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """
)


# Resolved before a test narrows PATH.
PKILL = shutil.which("pkill") or "pkill"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("client", ["curl", "bash"])
def test_rystic_against_a_fake_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: str
):
    binary = tmp_path / "bin" / "rystic"
    binary.parent.mkdir()
    _ = binary.write_text(f"#!{sys.executable}\n{FAKE_RYSTIC}")
    binary.chmod(0o755)
    monkeypatch.setattr(rystic_module, "BINARY", str(binary))
    if client == "bash":
        # A PATH without curl: the /dev/tcp fallback.
        tools = tmp_path / "tools"
        tools.mkdir()
        for tool in ("bash", "sh", "mkdir", "cat", "wc", "rm", "nohup", "sleep", "grep", "chmod"):
            found = shutil.which(tool)
            assert found is not None
            (tools / tool).symlink_to(found)
        monkeypatch.setenv("PATH", str(tools))
    elif shutil.which("curl") is None:
        pytest.skip("no curl")

    # What Laminar's remote sandboxes set for every trial.
    monkeypatch.setenv("RYSTIC_TOKEN", "rys_" + "a" * 43)
    monkeypatch.setenv(
        "RYSTIC_RELEASE_ENDPOINT", "http://127.0.0.1:9/v1/hosted-evals/rystic/download"
    )
    port = free_port()
    slack = Rystic(
        "slack",
        version="1.4",
        port=port,
        seed=case_fixture("slack.json"),
        env={"SLACK_API_URL": "{url}/api"},
        snapshot=True,
    )
    assert runner(services=[slack]).validate() == []
    case = make_case(
        tmp_path, {"slack.json": json.dumps({"messages": [{"text": "refund please"}]})}
    )
    ctx = AgentContext(
        environment=LocalEnvironment(),
        case=case,
        runner_dir=tmp_path,
        host_logs_dir=tmp_path / "agent",
        logs_dir=str(tmp_path / "logs"),
    )

    async def scenario() -> None:
        env = await start_services([slack], ctx)
        assert env == {"SLACK_API_URL": f"http://127.0.0.1:{port}/api"}
        state = await slack.state(ctx)
        assert state == {"product": "slack-twin", "messages": [{"text": "refund please"}]}
        history = await slack.get(ctx, "/api/conversations.history", channel="C1")
        assert history == {"channel": "C1", "messages": [{"text": "refund please"}]}
        assert (await slack.settle(ctx, quiet_ms=10))["settled"] is True
        assert await slack.calls(ctx) == [{"path": "/api/conversations.history", "ok": True}]
        # Saved in collect, read back in a separate verifier.
        await slack.snapshot(ctx, str(tmp_path / "trial" / "artifacts" / "state" / "slack"))

    try:
        asyncio.run(scenario())
    finally:
        _ = subprocess.run([PKILL, "-f", str(binary)], check=False)

    saved = tmp_path / "trial" / "artifacts" / "state" / "slack" / "state.json"
    assert json.loads(saved.read_text())["messages"] == [{"text": "refund please"}]
    # The seed was uploaded to a temp dir that is gone again.
    assert not list(Path("/tmp").glob(".*/slack.json"))

    artifacts = tmp_path / "verify-artifacts" / "logs" / "artifacts" / "state" / "slack"
    artifacts.mkdir(parents=True)
    _ = (artifacts / "state.json").write_text(saved.read_text())
    verify_ctx = VerifyContext(
        environment=LocalEnvironment(),
        case=case,
        runner_dir=tmp_path,
        host_agent_dir=tmp_path / "agent",
        host_artifacts_dir=tmp_path / "verify-artifacts",
        secrets=Secrets([], {}),
        verifier_mode="separate",
    )
    _ = (artifacts / "calls.json").write_text((saved.parent / "calls.json").read_text())
    assert asyncio.run(slack.state(verify_ctx))["messages"] == [{"text": "refund please"}]
    assert asyncio.run(slack.calls(verify_ctx)) == [
        {"path": "/api/conversations.history", "ok": True}
    ]
    with pytest.raises(EvalConfigError, match="isn't running in separate verifier mode"):
        asyncio.run(slack.get(verify_ctx, "/api/x"))


@pytest.mark.parametrize("token", [False, True])
def test_rystic_is_remote_only_for_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, token: bool
):
    """No RYSTIC_TOKEN (a local run): fail before the pull, with the reason. A
    token without Laminar's endpoint (`--env RYSTIC_TOKEN`): warn and try."""
    monkeypatch.delenv("RYSTIC_RELEASE_ENDPOINT", raising=False)
    if token:
        monkeypatch.setenv("RYSTIC_TOKEN", "rys_" + "b" * 43)
    else:
        monkeypatch.delenv("RYSTIC_TOKEN", raising=False)
    binary = tmp_path / "rystic"
    _ = binary.write_text("#!/bin/sh\necho pulled-without-access >&2; exit 3\n")
    binary.chmod(0o755)
    monkeypatch.setattr(rystic_module, "BINARY", str(binary))
    ctx = AgentContext(
        environment=LocalEnvironment(),
        case=make_case(tmp_path),
        runner_dir=tmp_path,
        host_logs_dir=tmp_path / "agent",
    )
    warnings: list[str] = []
    monkeypatch.setattr(ctx.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
    message = "pulled-without-access" if token else "only in `lmnr-cli eval run --remote`"
    with pytest.raises(Exception, match=message):
        asyncio.run(Rystic("slack", version="1.4").start(ctx))
    assert bool(warnings) == token
    if token:
        assert "only in `lmnr-cli eval run --remote`" in warnings[0]


def test_rystic_build_and_start_assumptions():
    slack = Rystic("slack", version="1.4", clock="2026-01-05T09:00:00Z")
    assert isinstance(slack, Mirror)
    assert (slack.name, slack.product) == ("slack", "slack-twin")
    assert Rystic("github-pulls-twin", version="1").name == "github-pulls"
    assert slack.start_command() == (
        f"/usr/local/bin/rystic run slack-twin --name slack --port {slack.port} "
        + "--clock 2026-01-05T09:00:00Z"
    )
    assert 18000 <= slack.port < 20000
    # The image gets the CLI at a pinned version; twins need a key, so no twin.
    stages = "\n".join(slack.bundle_stages(lambda p: str(p)))
    assert "FROM curlimages/curl:8.17.0 AS svc-slack" in stages
    assert "RYSTIC_VERSION=0.0.93" in stages
    assert slack.image_title(None).startswith("service: Rystic slack-twin 1.4, CLI 0.0.93 (")


def test_mirror_snapshot_chmods_after_every_file(tmp_path: Path):
    class Twin(Mirror):
        @override
        async def read_state(self, ctx: Context) -> Any:
            return {"channels": []}

        @override
        async def snapshot_extras(self, ctx: AgentContext) -> dict[str, Any]:
            return {"calls.json": []}

    env = RecordingEnvironment()
    ctx = AgentContext(
        environment=env, case=make_case(tmp_path), runner_dir=tmp_path, host_logs_dir=tmp_path / "a"
    )
    asyncio.run(Twin("twin", "1", snapshot=True).snapshot(ctx, "/logs/artifacts/state/twin"))
    commands = [c for c, _ in env.commands]
    uploads = [i for i, c in enumerate(commands) if c.startswith("upload ")]
    targets = [commands[i].rsplit(" -> ", 1)[1] for i in uploads]
    assert targets == [
        "/logs/artifacts/state/twin/state.json",
        "/logs/artifacts/state/twin/calls.json",
    ]
    assert commands[-1] == "chmod -R a+rX /logs/artifacts/state/twin"


def test_custom_service(tmp_path: Path):
    mail = Service.custom(
        "mailpit",
        "1.21",
        source="axllent/mailpit:v1.21",
        bundle=["/mailpit"],
        start="/mailpit --listen 127.0.0.1:8025",
        ready=cmd("true"),
        port=8025,
        env={"MAIL_URL": "{url}"},
    )
    assert runner(services=[mail]).validate() == []
    assert mail.env() == {"MAIL_URL": "http://127.0.0.1:8025"}
    env = RecordingEnvironment()
    ctx = AgentContext(
        environment=env, case=make_case(tmp_path), runner_dir=tmp_path, host_logs_dir=tmp_path / "a"
    )
    asyncio.run(start_services([mail], ctx))
    assert "nohup sh -c '/mailpit --listen 127.0.0.1:8025'" in env.commands[0][0]
    # The ready check gets the bundled libraries too, not only the process.
    waits = [i for i, (c, _) in enumerate(env.commands) if c.startswith("bash -c")]
    assert waits
    assert all((env.envs[i] or {}).get("LD_LIBRARY_PATH") == mail.lib_dir for i in waits)

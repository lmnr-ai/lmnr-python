"""Images and services for tests/test_evals_images.py's golden Dockerfiles.

Declared here, not in the test, so provenance comments (file:line) stay stable.
"""

from lmnr.evals import ClickHouse, EvalRunner, Image, Postgres, Service, Valkey, cmd
from lmnr.evals.mirrors import Rystic

BUILDER = (
    Image.python("3.13")
    .apt_install("curl", "poppler-utils")
    .pip_install("httpx==0.28.1")
    .uv_pip_install("rich")
    .pip_install_from_requirements("../../requirements.txt")
    .uv_sync("../../pyproject.toml")
    .npm_ci("../../web", "/srv/web")
    .pnpm_install("../../web-pnpm", "/srv/web-pnpm")
    .add_file("../../requirements.txt", "/srv/app/requirements.txt")
    .add_dir("../../app", "/srv/app/app")
    .run_commands("echo one", "echo 'two words'")
    .env(APP_ENV="eval", GREETING='say "hi"')
    .workdir("/srv/app")
)

UV_BASE = Image.uv_python("3.13").uv_sync("../../pyproject.toml")

COMPOSED = Image.from_dockerfile("../support/agent/Dockerfile", context="../..").apt_install("jq")
COMPOSED_SERVICES = [Valkey("8")]


class ServicesRunner(EvalRunner):
    image = Image.debian_slim()
    services = [  # noqa: RUF012
        Postgres(
            "16",
            migrations="../../migrations",
            migrate_at="build",
            extensions=["vector"],
            source="pgvector/pgvector:pg16-bookworm",
        ),
        Valkey("8"),
        ClickHouse(
            "25.8",
            database="analytics",
            migrations="../../clickhouse/migrations",
            config="../../clickhouse/config.xml",
            users="../../clickhouse/users.xml",
        ),
        Service.custom(
            "mailpit",
            "1.21",
            source="axllent/mailpit:v1.21",
            bundle=["/mailpit"],
            start="/mailpit --listen 127.0.0.1:8025",
            ready=cmd("true"),
            port=8025,
            env={"MAIL_URL": "{url}"},
        ),
        Rystic("slack", version="1.4", env={"SLACK_API_URL": "{url}"}),
    ]

    async def run(self, ctx):
        pass

    async def verify(self, ctx):
        return 1

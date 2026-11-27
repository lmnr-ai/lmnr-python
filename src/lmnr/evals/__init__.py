"""One-file agent evaluations: an `EvalRunner` subclass in `agent/eval.py`.

    from lmnr.evals import EvalRunner, Image, Postgres, secret

    def get_customer(call, ctx):          # replaces observe(name="crm.get_customer")
        return ctx.case.fixture_json("customers.json").get(call.args[0])

    class SupportEval(EvalRunner):
        image = Image.python("3.13").uv_sync("../../../pyproject.toml").add_dir(
            "../../../app", "/srv/app/app"
        )
        services = [Postgres("16", migrations="../../../migrations")]
        secrets = ["ANTHROPIC_API_KEY"]
        env = {"ANTHROPIC_API_KEY": secret("ANTHROPIC_API_KEY")}
        mocks = {"crm.get_customer": get_customer}

        async def run(self, ctx):
            r = await ctx.container.sh("python -m app.cli /case/ticket.json")
            ctx.set_result(reply=r.stdout)

        async def verify(self, ctx):
            looked_up = len(ctx.mocks.calls("crm.get_customer")) > 0
            return {"replied": bool(ctx.result().get("reply")), "looked_up": looked_up}

`eval.py` runs on the host; `ctx.container` runs commands in the trial's
container. `lmnr-cli eval run` runs it through Harbor. This package never
imports Harbor.
"""

from lmnr.evals.artifacts import Artifact, Artifacts
from lmnr.evals.context import (
    AgentContext,
    ArtifactSaver,
    BackgroundProcess,
    Case,
    Container,
    Context,
    VerifyContext,
)
from lmnr.evals.errors import CommandError, CommandTimeout, EvalConfigError, ExecResult
from lmnr.evals.fixtures import CaseFixture, case_fixture
from lmnr.evals.image import Image
from lmnr.evals.mirrors import Mirror, Rystic
from lmnr.evals.mocks import mock
from lmnr.evals.readiness import Readiness, cmd, http, port
from lmnr.evals.runner import EvalRunner, Secret, secret
from lmnr.evals.services import ClickHouse, Postgres, Service, Valkey

__all__ = [
    "AgentContext",
    "Artifact",
    "ArtifactSaver",
    "Artifacts",
    "BackgroundProcess",
    "Case",
    "CaseFixture",
    "ClickHouse",
    "CommandError",
    "CommandTimeout",
    "Container",
    "Context",
    "EvalConfigError",
    "EvalRunner",
    "ExecResult",
    "Image",
    "Mirror",
    "Postgres",
    "Readiness",
    "Rystic",
    "Secret",
    "Service",
    "Valkey",
    "VerifyContext",
    "case_fixture",
    "cmd",
    "http",
    "mock",
    "port",
    "secret",
]

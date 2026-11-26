"""One-file agent evaluations: an `EvalRunner` subclass in `agent/eval.py`.

    from lmnr.evals import EvalRunner, Image, secret

    class SupportEval(EvalRunner):
        image = Image.from_dockerfile("Dockerfile", context="../../..")
        secrets = ["ANTHROPIC_API_KEY"]
        env = {"ANTHROPIC_API_KEY": secret("ANTHROPIC_API_KEY")}

        async def run(self, ctx):
            r = await ctx.sh("python -m app.cli /case/ticket.json")
            ctx.result(reply=r.stdout)

        async def verify(self, ctx):
            return {"replied": bool(ctx.result().get("reply"))}

`lmnr-cli eval run` runs it through Harbor. This package never imports Harbor.
"""

from lmnr.evals.artifacts import Artifact, Artifacts
from lmnr.evals.context import (
    AgentContext,
    BackgroundProcess,
    Case,
    Context,
    VerifyContext,
)
from lmnr.evals.errors import CommandError, EvalConfigError, ExecResult
from lmnr.evals.image import Image
from lmnr.evals.readiness import Readiness, cmd, http, port
from lmnr.evals.runner import EvalRunner, Secret, secret

__all__ = [
    "AgentContext",
    "Artifact",
    "Artifacts",
    "BackgroundProcess",
    "Case",
    "CommandError",
    "Context",
    "EvalConfigError",
    "EvalRunner",
    "ExecResult",
    "Image",
    "Readiness",
    "Secret",
    "VerifyContext",
    "cmd",
    "http",
    "port",
    "secret",
]

import json
import sqlite3

from lmnr.evals import Artifact, cmd, http

from _shared.billing import BillingRunner
from _shared.judge import judge

AUDIT_DIR = "/srv/app/audit"


class SupportApiEval(BillingRunner):
    """The agent as it runs in production: a ticket service started in setup,
    called over HTTP in run, graded in a fresh container from what the run left."""

    # Collected after the agent phase, before grading: the service's audit
    # records, under artifacts/audit/ on the host.
    artifacts = [Artifact(AUDIT_DIR, destination="audit")]
    # The task.toml lmnr-cli writes for new cases. "separate": verify() gets a
    # fresh container from the same image, which sees only the artifacts.
    new_task_toml = {
        "environment": {"workdir": "/case"},
        "agent": {"timeout_sec": 300},
        "verifier": {"environment_mode": "separate"},
    }

    async def setup(self, ctx):
        await super().setup(ctx)  # _shared/seed.py
        # A background process with its log in /logs/agent/api.log. setup
        # returns once /health answers, and fails with the log's tail if the
        # process exits first.
        await ctx.container.service(
            "api",
            "python -m app.ticket_server",
            ready=http("http://127.0.0.1:8080/health"),
            env={"AGENT_MODEL": self.agent_model(ctx)},
        )

    async def run(self, ctx):
        # BillingRunner's helper runs _shared/submit_ticket.py in the container.
        ticket_id = await self.submit_ticket(ctx)
        ctx.set_result(ticket_id=ticket_id)  # merged with what set_result records later
        # The service answers at once and works in the background: wait until
        # it writes the ticket's record.
        record_path = f"{AUDIT_DIR}/{ticket_id}.json"
        try:
            await ctx.container.wait_for(cmd(f"test -f {record_path}"), timeout=240)
        except TimeoutError:
            ctx.set_result(outcome="timeout")
            return
        record = await ctx.container.read_json(record_path)
        if record["status"] == "done":
            ctx.set_result(outcome="completed", reply=record["reply"])
        else:
            ctx.set_result(outcome="failed", error=record["error"])

    async def collect(self, ctx):
        # Runs after run(), also when it failed. The service is still running;
        # copy its database to /logs/artifacts/, which Harbor always collects.
        await ctx.artifacts.save("/srv/app/app.db")

    async def verify(self, ctx):
        # All on the host: the grading container (ctx.container) is fresh and
        # has nothing to ask, so this reads the collected artifacts directly.
        expected = ctx.case.expected()
        result = ctx.result()
        if result.get("outcome") != "completed":
            ctx.logger.info(f"agent run did not complete: {result}")
            return {"refund": 0, "reply": 0}

        # The database as the run left it, saved by collect().
        with sqlite3.connect(ctx.artifacts.path("/logs/artifacts/app.db")) as conn:
            refunded = [r[0] for r in conn.execute("SELECT charge_id FROM refunds")]
        ctx.logger.info(f"refunded: {refunded}")
        refund_ok = (
            len(refunded) == expected["refund_count"]
            and len(set(refunded)) == len(refunded)
            and set(refunded) <= set(expected["refundable_charge_ids"])
        )

        # The service's own record of the ticket, a declared artifact, read by
        # the container path it came from.
        record = ctx.artifacts.read_json(f"{AUDIT_DIR}/{result['ticket_id']}.json")
        ticket = json.loads(ctx.case.file("case.json").read_text())["ticket"]
        verdict = await judge(
            ticket, record["reply"], expected["reply_should"], ctx.secrets["ANTHROPIC_API_KEY"]
        )
        ctx.logger.info(f"judge: {'pass' if verdict.passed else 'fail'}: {verdict.reason}")
        return {"refund": refund_ok, "reply": verdict.passed}

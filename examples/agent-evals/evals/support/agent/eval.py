import json
import shlex

from lmnr.evals import EvalRunner, Image, secret

# Runs in the container in verify(): what actually happened in the database.
REFUNDS_QUERY = (
    "import json, sqlite3; "
    "print(json.dumps(sqlite3.connect('/srv/app/app.db').execute("
    "'SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM refunds').fetchone()))"
)


class SupportEval(EvalRunner):
    # The image every case starts from. The context is the example's root, so
    # the image copies the app's code.
    image = Image.from_dockerfile("Dockerfile", context="../../..")
    model = "anthropic/claude-sonnet-5"
    # Names only: lmnr-cli reads the values from your shell, .env.local or .env.
    secrets = ["ANTHROPIC_API_KEY"]
    env = {"ANTHROPIC_API_KEY": secret("ANTHROPIC_API_KEY")}
    # Written into the task.toml of new cases (lmnr-cli writes it when missing).
    task_template = {"environment": {"workdir": "/case"}, "agent": {"timeout_sec": 300}}

    async def setup(self, ctx):
        # This case's database, from its environment/seed.json.
        await ctx.run_file("seed.py")

    async def run(self, ctx):
        # "anthropic/claude-sonnet-5" -> "claude-sonnet-5"
        model = (ctx.model or "").split("/", 1)[-1]
        r = await ctx.sh(
            "python -m app.support_agent /case/case.json",
            env={"AGENT_MODEL": model},
            cwd="/srv/app",
            timeout=240,
            check=False,
        )
        if r.return_code == 0:
            ctx.result(outcome="completed", reply=r.stdout.strip())
        else:
            ctx.result(outcome="failed", error=r.stderr[-2000:])

    async def verify(self, ctx):
        expected = ctx.case.expected()  # tests/expected.json, never in the container
        result = ctx.result()
        if result.get("outcome") != "completed":
            ctx.logger.info(f"agent run did not complete: {result}")
            return {"refund": 0, "reply": 0}

        # 1. Hard check: what actually happened in the database.
        r = await ctx.sh(f"python -c {shlex.quote(REFUNDS_QUERY)}")
        count, total = json.loads(r.stdout)
        ctx.logger.info(f"refunds: {count} totalling {total}")
        refund_ok = count == expected["refund_count"] and total == expected["refund_amount"]

        # 2. LLM judge: is the reply to the customer any good?
        reply_ok = await ctx.judge(
            result["reply"],
            expected["reply_should"],
            model="anthropic/claude-haiku-4-5",
            api_key=ctx.secrets["ANTHROPIC_API_KEY"],
        )
        return {"refund": refund_ok, "reply": reply_ok}

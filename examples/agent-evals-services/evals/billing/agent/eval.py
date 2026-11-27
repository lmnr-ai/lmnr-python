from anthropic import AsyncAnthropic
from lmnr import observe
from lmnr.evals import ClickHouse, EvalRunner, Image, Postgres, case_fixture, secret
from pydantic import BaseModel


class Verdict(BaseModel):
    passed: bool
    reason: str


# An LLM judge is a plain function: verify() runs in the Harbor process, so it
# can call any SDK listed in evals/pyproject.toml. @observe records the
# arguments as the span's input, so the key is left out.
@observe(name="judge", ignore_inputs=["api_key"])
async def judge(reply: str, criteria: str, api_key: str) -> Verdict:
    response = await AsyncAnthropic(api_key=api_key).messages.parse(
        model="claude-haiku-5-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    "You grade a billing support agent's reply to a customer.\n\n"
                    f"Criteria: {criteria}\n\nReply:\n{reply}\n\n"
                    "Does the reply meet the criteria?"
                ),
            }
        ],
        output_format=Verdict,
    )
    if response.parsed_output is None:
        raise ValueError(f"The judge gave no verdict (stop reason {response.stop_reason})")
    return response.parsed_output


class BillingEval(EvalRunner):
    # No Dockerfile: lmnr-cli builds this. Paths are relative to this file, so
    # ../../.. is the example's root. Dependencies first, from the lockfile
    # (pnpm and the version in package.json's packageManager), then the code,
    # so a code change rebuilds one small layer.
    image = (
        Image.node("24")
        .pnpm_install("../../..", "/srv/app")
        .add_dir("../../../src", "/srv/app/src")
    )
    # Started, migrated and seeded in every trial before setup(), each with
    # this case's tests/fixtures/ (data the agent never sees as files). Their
    # URLs reach the app as DATABASE_URL / CLICKHOUSE_URL.
    services = [
        Postgres(
            "17",
            database="billing",
            migrations="../../../db/migrations",  # *.sql in natural order
            seed=case_fixture("accounts.sql", required=True),
        ),
        ClickHouse(
            "25.8",
            database="usage",
            # Applied with the clickhouse-migrations CLI, like production does.
            migrations="../../../clickhouse/migrations",
            # Production's settings, merged over the defaults.
            config="../../../clickhouse/config.xml",
            users="../../../clickhouse/users.xml",
            seed=case_fixture("usage.sql", required=True),
            env={"CLICKHOUSE_URL": "{url}", "CLICKHOUSE_DB": "{database}"},
        ),
    ]
    model = "anthropic/claude-sonnet-5-5"
    # Names only: lmnr-cli reads the values from your shell, .env.local or .env.
    secrets = ["ANTHROPIC_API_KEY"]
    env = {"ANTHROPIC_API_KEY": secret("ANTHROPIC_API_KEY")}
    new_task_toml = {"environment": {"workdir": "/case"}, "agent": {"timeout_sec": 300}}

    # No setup(): the services already hold this case's data.

    async def run(self, ctx):
        # "anthropic/claude-sonnet-5-5" -> "claude-sonnet-5-5"
        model = (ctx.model or "").split("/", 1)[-1]
        r = await ctx.container.sh(
            "node src/agent.ts /case/case.json",
            # Laminar's TS SDK logs to stdout, where the reply goes.
            env={"AGENT_MODEL": model, "LMNR_LOG_LEVEL": "warn"},
            cwd="/srv/app",
            timeout=240,
            check=False,
        )
        if r.return_code == 0:
            ctx.set_result(outcome="completed", reply=r.stdout.strip())
        else:
            ctx.set_result(outcome="failed", error=r.stderr[-2000:])

    async def verify(self, ctx):
        expected = ctx.case.expected()  # tests/expected.json, never in the container
        result = ctx.result()
        if result.get("outcome") != "completed":
            ctx.logger.info(f"agent run did not complete: {result}")
            return {"credit": 0, "reply": 0}

        # 1. Hard check: the credits the agent issued, straight from Postgres.
        [credits] = await self.service(Postgres).sql(
            ctx, "SELECT count(*)::int AS n, coalesce(sum(amount_cents), 0)::int AS cents FROM credits"
        )
        ctx.logger.info(f"credits: {credits['n']} totalling {credits['cents']} cents")
        credit_ok = credits["cents"] == expected["credit_cents"]

        # 2. LLM judge: is the reply to the customer right?
        verdict = await judge(
            result["reply"], expected["reply_should"], ctx.secrets["ANTHROPIC_API_KEY"]
        )
        ctx.logger.info(f"judge: {'pass' if verdict.passed else 'fail'}: {verdict.reason}")
        return {"credit": credit_ok, "reply": verdict.passed}

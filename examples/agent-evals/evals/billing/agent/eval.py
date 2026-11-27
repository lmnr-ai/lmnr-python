from datetime import datetime, timedelta, timezone

from anthropic import AsyncAnthropic
from lmnr import observe
from lmnr.evals import ClickHouse, EvalRunner, Image, Postgres, Rystic, case_fixture, secret
from pydantic import BaseModel

CHANNEL = "C0BILLING"  # #billing-help, from each case's tests/fixtures/slack.json
CUSTOMER_MESSAGE_TS = "1767225600.000100"


class Verdict(BaseModel):
    passed: bool
    reason: str


# An LLM judge is a plain function: verify() runs in the process that runs
# Harbor, so it can call any SDK listed in evals/pyproject.toml. @observe
# records the arguments as the span's input, so the key is left out.
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


# A mock: the app's observed `incidents.list_recent` (src/incidents.ts) is
# answered by this function instead of calling the status service. It runs
# in the container, in the mock server; ctx.fixture_json reads this case's
# tests/fixtures/, which the agent never sees as files.
def list_incidents(call, ctx):
    [days] = call.args  # a TypeScript function's arguments, in order
    now = datetime.now(timezone.utc)
    return [
        {
            "id": incident["id"],
            "title": incident["title"],
            "started_at": (now - timedelta(days=incident["started_days_ago"])).isoformat(),
            "resolved_at": now.isoformat() if incident["resolved"] else None,
            "summary": incident["summary"],
        }
        for incident in ctx.fixture_json("incidents.json")
        if incident["started_days_ago"] <= days
    ]


class BillingEval(EvalRunner):
    # No Dockerfile: lmnr-cli builds this. Paths are relative to this file, so
    # ../../.. is the example's root. Dependencies first, from the lockfile
    # (pnpm at the version in package.json's packageManager), then the code,
    # so a code change rebuilds one small layer.
    image = (
        Image.node("24")
        .pnpm_install("../../..", "/srv/app")
        .add_dir("../../../src", "/srv/app/src")
    )

    # Started in every trial before setup(), each loaded with this case's
    # tests/fixtures/ (data the agent never sees as files), and reachable by
    # the app through the env vars they set.
    services = [
        Postgres(
            "17",
            database="billing",
            migrations="../../../db/migrations",  # *.sql in natural order
            seed=case_fixture("accounts.sql", required=True),
        ),  # DATABASE_URL
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
        # A Slack workspace simulated by Rystic: the customer's message is in
        # the case's slack.json, and the app talks to it with the official
        # Slack SDK. Needs a Rystic API key (see the README).
        Rystic(
            "slack",
            version="0.0.31",
            seed=case_fixture("slack.json", required=True),
            env={"SLACK_API_URL": "{url}/api/"},
        ),
    ]

    mocks = {"incidents.list_recent": list_incidents}
    # The mock server imports this file in the container, and the file
    # imports anthropic (for the judge): install it there too.
    mock_dependencies = ["anthropic>=2.4.0"]

    model = "anthropic/claude-sonnet-5-5"
    # Names only: lmnr-cli reads the values from your shell, .env.local or .env.
    secrets = ["ANTHROPIC_API_KEY"]
    env = {
        "ANTHROPIC_API_KEY": secret("ANTHROPIC_API_KEY"),
        "SLACK_BOT_TOKEN": "xoxb-eval",  # any xoxb- token: the simulator acts as the bot
        "SLACK_CHANNEL_ID": CHANNEL,
    }
    new_task_toml = {"agent": {"timeout_sec": 300}}

    # No setup(): the services already hold this case's data.

    async def run(self, ctx):
        # "anthropic/claude-sonnet-5-5" -> "claude-sonnet-5-5"
        model = (ctx.model or "").split("/", 1)[-1]
        r = await ctx.container.sh(
            "node src/agent.ts slack",
            # Laminar's TS SDK logs to stdout, where the reply goes.
            env={"AGENT_MODEL": model, "LMNR_LOG_LEVEL": "warn"},
            cwd="/srv/app",
            timeout=240,
            check=False,
        )
        if r.return_code == 0:
            ctx.set_result(outcome="completed")
        else:
            # Local runs return stderr inside stdout; remote runs keep them apart.
            ctx.set_result(outcome="failed", error=(r.stdout + r.stderr)[-2000:])

    async def verify(self, ctx):
        expected = ctx.case.expected()  # tests/expected.json, never in the container
        if ctx.result().get("outcome") != "completed":
            ctx.logger.info(f"agent run did not complete: {ctx.result()}")
            return {"credit": 0, "reply": 0, "in_thread": 0}

        # 1. The credits the agent issued, straight from Postgres.
        [credits] = await self.service(Postgres).sql(
            ctx,
            "SELECT count(*)::int AS n, coalesce(sum(amount_cents), 0)::int AS cents FROM credits",
        )
        ctx.logger.info(f"credits: {credits['n']} totalling {credits['cents']} cents")

        # 2. What the agent posted to Slack: every call the app made to the
        # simulated API, in order.
        posts = [
            call["args"]
            for call in await self.mirror("slack").calls(ctx)
            if call["path"] == "/api/chat.postMessage"
        ]
        ctx.logger.info(f"Slack posts: {posts}")
        in_thread = len(posts) == 1 and posts[0].get("thread_ts") == CUSTOMER_MESSAGE_TS

        # 3. LLM judge: is the reply to the customer right?
        reply = posts[-1]["text"] if posts else ""
        verdict = await judge(reply, expected["reply_should"], ctx.secrets["ANTHROPIC_API_KEY"])
        ctx.logger.info(f"judge: {'pass' if verdict.passed else 'fail'}: {verdict.reason}")
        return {
            "credit": credits["cents"] == expected["credit_cents"],
            "reply": verdict.passed,
            "in_thread": in_thread,
        }

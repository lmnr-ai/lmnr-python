import json
import shlex

from anthropic import AsyncAnthropic
from lmnr import observe
from lmnr.evals import EvalRunner, Image, mock, secret
from pydantic import BaseModel

# Runs in the container in verify(): what actually happened in the database.
REFUNDS_QUERY = (
    "import json, sqlite3; "
    "print(json.dumps(sqlite3.connect('/srv/app/app.db').execute("
    "'SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM refunds').fetchone()))"
)


class Verdict(BaseModel):
    passed: bool
    reason: str


# An LLM judge is a plain function: verify() runs in the Harbor process, so it
# can call any SDK listed in evals/pyproject.toml (lmnr-cli installs them). The call is
# traced under the trial's verifier span; @observe records the arguments as the
# span's input, so the key is left out.
@observe(name="judge", ignore_inputs=["api_key"])
async def judge(reply: str, criteria: str, api_key: str) -> Verdict:
    client = AsyncAnthropic(api_key=api_key)
    response = await client.messages.parse(
        model="claude-haiku-5-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    "You grade a support agent's reply to a customer.\n\n"
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


# Mocks: plain functions that run inside the container, in the mock server, for
# every trial. ctx.fixture_json reads this case's tests/fixtures/, which the
# agent never sees as files.
def get_customer(call, ctx):
    """Answers the app's observed `crm.get_customer` instead of running it
    (function mocks need lmnr >= 0.9.3 in the app; an older SDK runs the real
    function, which calls CRM_API_URL). `call.args` are the function's
    arguments by name; a `mock.Error` raises `lmnr.MockedError` in the app."""
    customer = ctx.fixture_json("crm.json").get(call.args["email"])
    if customer is None:
        raise mock.Error("CustomerNotFound", f"no customer {call.args['email']}")
    return customer


def helpdesk(req, ctx):
    """The helpdesk API, which the app reaches through HELPDESK_API_URL."""
    if req.method == "POST" and req.path.endswith("/replies"):
        return mock.Response(201, json={"id": f"reply-{req.index}"})
    return None  # anything else: 501, logged as unmatched


class SupportEval(EvalRunner):
    # The image every case starts from. The context is the example's root, so
    # the image copies the app's code.
    image = Image.from_dockerfile("Dockerfile", context="../../..")
    model = "anthropic/claude-sonnet-5"
    # Names only: lmnr-cli reads the values from your shell, .env.local or .env.
    secrets = ["ANTHROPIC_API_KEY"]
    env = {"ANTHROPIC_API_KEY": secret("ANTHROPIC_API_KEY")}
    mocks = {
        "crm.get_customer": get_customer,
        "https://helpdesk.example.com/api": mock.http(helpdesk, base_url_env="HELPDESK_API_URL"),
    }
    # The mock server imports this file in the container, and the file imports
    # anthropic (for the judge): install it there too.
    mock_dependencies = ["anthropic>=2.4.0"]
    # The task.toml lmnr-cli writes for new cases (tasks without one).
    new_task_toml = {"environment": {"workdir": "/case"}, "agent": {"timeout_sec": 300}}

    async def setup(self, ctx):
        # This case's database, from its environment/seed.json.
        await ctx.container.run_file("seed.py")

    async def run(self, ctx):
        # "anthropic/claude-sonnet-5" -> "claude-sonnet-5"
        model = (ctx.model or "").split("/", 1)[-1]
        r = await ctx.container.sh(
            "python -m app.support_agent /case/case.json",
            env={"AGENT_MODEL": model},
            cwd="/srv/app",
            timeout=240,
            check=False,
        )
        if r.return_code == 0:
            ctx.set_result(outcome="completed", reply=r.stdout.strip())
        else:
            ctx.set_result(outcome="failed", error=r.stderr[-2000:])

    async def verify(self, ctx):
        # verify() runs on the host. ctx.container is the grading container
        # (here the agent's, after it finished); the rest of ctx is read here.
        expected = ctx.case.expected()  # tests/expected.json, never in the container
        result = ctx.result()
        if result.get("outcome") != "completed":
            ctx.logger.info(f"agent run did not complete: {result}")
            return {"refund": 0, "reply": 0, "posted": 0}

        # 1. Hard check: what actually happened in the database.
        r = await ctx.container.sh(f"python -c {shlex.quote(REFUNDS_QUERY)}")
        count, total = json.loads(r.stdout)
        ctx.logger.info(f"refunds: {count} totalling {total}")
        refund_ok = count == expected["refund_count"] and total == expected["refund_amount"]

        # 2. LLM judge: is the reply to the customer any good?
        verdict = await judge(
            result["reply"], expected["reply_should"], ctx.secrets["ANTHROPIC_API_KEY"]
        )
        ctx.logger.info(f"judge: {'pass' if verdict.passed else 'fail'}: {verdict.reason}")

        # 3. The mocks' call log: the reply went out through the helpdesk, once.
        posts = [
            c
            for c in ctx.mocks.calls("https://helpdesk.example.com/api")
            if c["request"]["method"] == "POST"
        ]
        return {"refund": refund_ok, "reply": verdict.passed, "posted": len(posts) == 1}

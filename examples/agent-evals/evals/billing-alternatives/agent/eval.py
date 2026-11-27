"""The billing agent run and graded the other ways lmnr.evals supports. The
`billing` dataset is the recommended setup; each choice here notes what it
replaces there and when you'd pick it."""

from datetime import datetime, timedelta, timezone

from lmnr.evals import Artifact, ClickHouse, EvalRunner, Image, Postgres, mock, secret

STATUS_API = "https://status.meterly.dev"


# An HTTP mock instead of a function mock: the runner points the app's
# INCIDENTS_API_URL at the mock server, so the app's real HTTP code runs. Use
# it when the code to skip isn't an observed function, or the app isn't
# instrumented with Laminar.
def status_api(req, ctx):
    if req.method == "GET" and req.path.endswith("/v1/incidents"):
        days = int(req.query.get("since_days", ["30"])[0])
        now = datetime.now(timezone.utc)
        return [
            {
                "id": incident["id"],
                "title": incident["title"],
                "started_at": (now - timedelta(days=incident["started_days_ago"])).isoformat(),
                "resolved_at": now.isoformat() if incident["resolved"] else None,
                "summary": incident["summary"],
            }
            for incident in ctx.case.fixture_json("incidents.json")
            if incident["started_days_ago"] <= days
        ]
    return None  # anything else: 501, logged as unmatched


class BillingAlternativesEval(EvalRunner):
    # An existing Dockerfile instead of Image's methods. Its context is the
    # example's root; lmnr-cli adds the services and the Laminar layer on top.
    image = Image.from_dockerfile("Dockerfile", context="../../..")

    services = [
        # No seed=: setup() loads the rows. snapshot= saves the table into
        # /logs/artifacts/ after the run, for a grader that can't query the
        # live database (the separate verifier below).
        Postgres(
            "17",
            database="billing",
            migrations="../../migrations/postgres",
            snapshot=["credits"],
        ),
        ClickHouse(
            "25.8",
            database="usage",
            migrations="../../migrations/clickhouse",
            config="../../../clickhouse/config.xml",
            users="../../../clickhouse/users.xml",
            env={"CLICKHOUSE_URL": "{url}", "CLICKHOUSE_DB": "{database}"},
        ),
    ]
    mocks = {STATUS_API: mock.http(status_api, base_url_env="INCIDENTS_API_URL")}

    env = {"ANTHROPIC_API_KEY": secret("ANTHROPIC_API_KEY"), "LMNR_LOG_LEVEL": "warn"}

    # One command instead of run(). It gets no ctx, so the app uses its
    # default model: define run() to pass ctx.model (and honor --model).
    run_command = "node /srv/app/src/agent.ts /case/ticket.json > /srv/app/out/reply.txt"
    # Collected after the run, before grading, into the trial's artifacts/out/.
    artifacts = [Artifact("/srv/app/out", destination="out")]

    # Harbor's default verifier instead of verify(): it runs each case's own
    # tests/test.sh, which writes the rewards. Imported Harbor datasets work
    # this way; each case can check something different.
    grade_with_test_sh = True
    new_task_toml = {
        "environment": {"workdir": "/case"},
        "agent": {"timeout_sec": 300},
        # A fresh container for grading, which sees only the artifacts.
        "verifier": {"environment_mode": "separate"},
    }

    async def setup(self, ctx):
        # Instead of seed=case_fixture(...): load the case's rows yourself,
        # e.g. when they need computing or come from several files.
        await self.service(Postgres).execute(ctx, ctx.case.fixture_text("accounts.sql"))
        await self.service(ClickHouse).execute(ctx, ctx.case.fixture_text("usage.sql"))

from _shared.billing import BillingRunner


class SupportTestShEval(BillingRunner):
    """Graded like an imported Harbor dataset: each case's own tests/test.sh
    checks the outcome, so cases can test different things without verify()."""

    # One command instead of run(). It gets no ctx, so the app uses its default
    # model: define run() instead to pass ctx.model (and honor --model).
    run_command = "python -m app.support_agent /case/case.json > /logs/agent/reply.txt"
    # Harbor's default verifier runs tests/test.sh in the container after the
    # agent; it writes the reward to /logs/verifier/reward.txt.
    grade_with_test_sh = True
    new_task_toml = {"environment": {"workdir": "/case"}, "agent": {"timeout_sec": 300}}

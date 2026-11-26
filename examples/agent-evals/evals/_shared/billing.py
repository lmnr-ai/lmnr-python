"""The runner every billing dataset shares; each dataset's agent/eval.py
subclasses it. lmnr puts the evals folder on the import path, so eval.py
imports this as `from _shared.billing import BillingRunner`.

Relative paths resolve against the file that declares them: the Dockerfile
and seed.py below are the ones next to this file, whichever dataset runs.
"""

from lmnr.evals import EvalRunner, Image, secret


class BillingRunner(EvalRunner):
    # The context is the example's root (two levels up from this file), so the
    # image copies the app's code. Each dataset builds it as <folder>-evals.
    image = Image.from_dockerfile("Dockerfile", context="../..")
    model = "anthropic/claude-sonnet-5"
    secrets = ["ANTHROPIC_API_KEY"]
    env = {"ANTHROPIC_API_KEY": secret("ANTHROPIC_API_KEY")}

    async def setup(self, ctx):
        # _shared/seed.py: this case's database, from its environment/seed.json.
        await ctx.container.run_file("seed.py")

    async def submit_ticket(self, ctx) -> str:
        """Queue the case's ticket with the ticket service; returns its id.

        A helper here rather than in a dataset's eval.py: run_file resolves a
        relative path against the file whose code calls it, so this finds
        _shared/submit_ticket.py for every dataset.
        """
        r = await ctx.container.run_file("submit_ticket.py", "/case/case.json")
        return r.stdout.strip()

    def agent_model(self, ctx) -> str:
        """`anthropic/claude-sonnet-5` -> `claude-sonnet-5`, as the app takes it."""
        return (ctx.model or "").split("/", 1)[-1]

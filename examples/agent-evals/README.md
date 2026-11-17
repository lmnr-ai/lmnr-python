# Agent evaluation example

A support agent that handles billing tickets, with an [agent evaluation](https://laminar.sh/docs/agent-evaluations/introduction) you run with `lmnr-cli eval run`. The agent looks up a customer's charges in a SQLite database and can refund one. Each case starts with its own copy of the database, and a verifier checks both the database and the reply.

The [quickstart](https://laminar.sh/docs/agent-evaluations/quickstart#get-started-manually) walks through the files.

## Layout

```
app/
  support_agent.py            # the agent: Anthropic tool loop, instrumented with Laminar
pyproject.toml, uv.lock       # the app's dependencies, installed into the image
evals/
  billing/                    # a task set; everything but tasks/ is its scripts
    eval.yaml                 # the image tasks use, the Harbor job config, the version pin
    adapter.py                # Harbor agent that runs run_case.py in the container
    image/
      Dockerfile
      run_case.py             # seeds the database, calls the agent, writes result.json
    tasks/                    # the cases, versioned in Laminar
      duplicate-charge/       # a task (case)
        task.toml
        instruction.md
        environment/
          case.json           # the ticket
          seed.json           # the database rows for this case
        tests/
          test.sh             # the verifier
          verify.py
          expected.json       # the reference answer
```

## Run it

You need [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/), and Node.js.

1. Run every command from this directory. `lmnr-cli` looks for `evals/` and `.env` in the current directory.
2. Link it to a Laminar project. This writes `LMNR_PROJECT_API_KEY` to `.env`:

   ```bash
   npx lmnr-cli@latest setup
   ```

3. Add `ANTHROPIC_API_KEY` to `.env` (see `.env.example`).
4. Run the task set:

   ```bash
   npx lmnr-cli@latest eval run billing
   ```

5. Publish the task set to Laminar, so later runs (yours, CI's, or Laminar's agent) start from it:

   ```bash
   npx lmnr-cli@latest eval push billing
   ```

6. Pin the version it printed in `evals/billing/eval.yaml` (`version: 1`), so runs without a local copy, such as CI, use it.

Each trial shows up in Laminar as a datapoint with a `refund` score and a `reply` score, next to the trace of the agent's model and tool calls.

To add a case, copy `evals/billing/tasks/duplicate-charge`, then change `case.json`, `seed.json`, and `expected.json`, and `eval push billing` to publish it. Bump `version` in `eval.yaml` to the new version to run it in CI.

In your own repo, tasks live in Laminar rather than in git: `lmnr-cli eval pull` and `eval push` write an `evals/<set>/.gitignore` for `tasks/`. Task set names are unique per Laminar project. The scripts must be committed, since `eval run` only uses the repo's copy, and task sets can share files from a sibling folder such as `evals/common/`. This example ships its tasks so it runs out of the box.

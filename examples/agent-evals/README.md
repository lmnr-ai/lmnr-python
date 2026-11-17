# Agent evaluation example

A support agent that handles billing tickets, with an [agent evaluation](https://laminar.sh/docs/agent-evaluations/introduction) you run with `lmnr-cli eval run`. The agent looks up a customer's charges in a SQLite database and can refund one. Each case starts with its own copy of the database, and a verifier checks both the database and the reply.

The [quickstart](https://laminar.sh/docs/agent-evaluations/quickstart#get-started-manually) walks through the files.

## Layout

```
app/
  support_agent.py            # the agent: Anthropic tool loop, instrumented with Laminar
pyproject.toml, uv.lock       # the app's dependencies, installed into the image
evals/
  support/                    # an agent; everything but task_sets/ is its scripts
    agent.yaml                # the image tasks use, the Harbor job config, task set pins
    adapter.py                # Harbor agent that runs run_case.py in the container
    image/
      Dockerfile
      run_case.py             # seeds the database, calls the agent, writes result.json
    task_sets/
      billing/                # a task set, versioned in Laminar
        duplicate-charge/     # a task (case)
          task.toml
          instruction.md
          environment/
            case.json         # the ticket
            seed.json         # the database rows for this case
          tests/
            test.sh           # the verifier
            verify.py
            expected.json     # the reference answer
```

## Run it

You need [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/), and Node.js.

1. Run every command from this directory. `lmnr-cli` looks for `evals/` and `.env` in the current directory.
2. Link it to a Laminar project. This writes `LMNR_PROJECT_API_KEY` to `.env`:

   ```bash
   npx lmnr-cli@latest setup
   ```

3. Add `ANTHROPIC_API_KEY` to `.env` (see `.env.example`).
4. Run the agent's task sets:

   ```bash
   npx lmnr-cli@latest eval run support
   ```

5. Publish the task set to Laminar, so later runs (yours, CI's, or Laminar's agent) start from it:

   ```bash
   npx lmnr-cli@latest eval push support
   ```

6. Pin the version it printed in `evals/support/agent.yaml` (`task_sets: {billing: 1}`), so runs without a local copy, such as CI, use it.

Each trial shows up in Laminar as a datapoint with a `refund` score and a `reply` score, next to the trace of the agent's model and tool calls.

To add a case, copy `evals/support/task_sets/billing/duplicate-charge`, then change `case.json`, `seed.json`, and `expected.json`, and `eval push support/billing` to publish it. Bump the pin in `agent.yaml` to the new version to run it in CI.

In your own repo, task sets live in Laminar rather than in git: `lmnr-cli eval pull` writes an `evals/<agent>/.gitignore` for `task_sets/`. The agent scripts must be committed, since `eval run` only uses the repo's copy. This example ships its task set so it runs out of the box.

# Agent evaluation example

A support agent that handles billing tickets, with an [agent evaluation](https://laminar.sh/docs/agent-evaluations/introduction) suite you run with `lmnr-cli eval run`. The agent looks up a customer's charges in a SQLite database and can refund one. Each case starts with its own copy of the database, and a verifier checks both the database and the reply.

The [quickstart](https://laminar.sh/docs/agent-evaluations/quickstart#get-started-manually) walks through every file.

## Layout

```
app/
  support_agent.py            # the agent: Anthropic tool loop, instrumented with Laminar
pyproject.toml, uv.lock       # the app's dependencies, installed into the image
evals/
  lmnr.yaml                   # the image tasks use, and how to build it
  image/
    Dockerfile
    shared/
      run_case.py             # seeds the database, calls the agent, writes result.json
  support/                    # a suite
    harbor.yaml
    agent/
      __init__.py
      agent.py                # Harbor agent that runs run_case.py in the container
    tasks/
      duplicate-charge/       # a case
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
4. Run the suite:

   ```bash
   npx lmnr-cli@latest eval run
   ```

Each trial shows up in Laminar as a datapoint with a `refund` score and a `reply` score, next to the trace of the agent's model and tool calls.

To add a case, copy `evals/support/tasks/duplicate-charge`, then change `case.json`, `seed.json`, and `expected.json`.

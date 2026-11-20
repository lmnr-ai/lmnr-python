# Agent evaluation example

A support agent that handles billing tickets, with an [agent evaluation](https://laminar.sh/docs/agent-evaluations/introduction) dataset you run with `lmnr-cli eval run`. The agent looks up a customer's charges in a SQLite database and can refund one. Each case starts with its own copy of the database, and a verifier checks both the database and the reply.

The [quickstart](https://laminar.sh/docs/agent-evaluations/quickstart#get-started-manually) walks through the files.

## Layout

```
app/
  support_agent.py            # the agent: Anthropic tool loop, instrumented with Laminar
pyproject.toml, uv.lock       # the app's dependencies, installed into the image
evals/
  support/                    # a dataset
    harbor.yaml               # Harbor config, plus the image tasks use and how to build it
    agent/                    # everything the dataset runs, one file each
      Dockerfile              # the image every case starts from
      agent.py                # Harbor agent: runs seed.py, then run.py, in the container
      seed.py                 # creates this case's database from its seed.json
      run.py                  # calls the agent on the ticket, writes result.json
      verify.py               # the verifier every case runs
    tasks/
      dataset.toml            # the dataset's name in Laminar, and its task versions
      duplicate-charge/       # a case
        task.toml             # [task] name, the image, timeouts
        instruction.md
        environment/
          case.json           # the ticket
          seed.json           # the database rows for this case
        tests/
          test.sh             # runs verify.py
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
4. Run the dataset:

   ```bash
   npx lmnr-cli@latest eval run
   ```

Each trial shows up in Laminar as a datapoint with a `refund` score and a `reply` score, next to the trace of the agent's model and tool calls.

To add a case, copy `evals/support/tasks/duplicate-charge`, give it a new `name` in its `task.toml`, then change `case.json`, `seed.json`, and `expected.json`.

## Store the cases in Laminar

The case folders are committed here so the example runs as it is. In your own project they live in Laminar instead, versioned by content, and git only keeps `tasks/dataset.toml`:

```bash
npx lmnr-cli@latest eval dataset push
```

This uploads each case, pins its version in `tasks/dataset.toml`, and adds `*/tasks/*` (except `dataset.toml`) to `evals/.gitignore`. Teammates get the pinned cases with `npx lmnr-cli@latest eval dataset pull` before running them: `eval run` runs the cases in `tasks/` as they are and never pulls. The project's **Eval datasets** page lists each case and its versions.
